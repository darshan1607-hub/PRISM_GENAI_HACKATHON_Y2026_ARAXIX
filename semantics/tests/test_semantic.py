"""
Unit test suite for Semantic Code Search (Member 2).

Tests:
1. CodeChunk creation and dictionary serialization.
2. AST-based chunking (functions, classes, methods, interfaces).
3. Line-based chunking fallback (malformed syntax, unsupported files).
4. Preservation of original source code and accurate 1-indexed line numbers.
5. GeminiEmbeddingProvider API key validation and clear error reporting.
6. MockEmbeddingProvider deterministic behavior, dimensionality, and normalization.
7. FAISSIndex vector storage, cosine similarity search, and top-K ranking.
8. FAISSIndex persistence: saving index + metadata to disk and reloading.
9. Metadata 1-to-1 linkage: verifying retrieved chunks maintain file, lines, and code.
10. SemanticCodeSearch end-to-end search pipeline integration.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.semantic import (
    CodeChunk,
    CodeChunker,
    FAISSIndex,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    SemanticCodeSearch,
    create_embedding_provider,
)
from backend.structural import RepositoryScanner, SourceFile, TreeSitterParser


class TestCodeChunker(unittest.TestCase):
    """Test suite for CodeChunk and CodeChunker."""

    def setUp(self) -> None:
        self.chunker = CodeChunker()

    def test_chunk_data_structure(self) -> None:
        """Verify CodeChunk preserves file, start_line, end_line, and code."""
        chunk = CodeChunk(
            file="src/example.ts",
            start_line=10,
            end_line=25,
            code="function example() {\n  return true;\n}",
            chunk_type="function",
            name="example",
        )
        d = chunk.to_dict()
        self.assertEqual(d["file"], "src/example.ts")
        self.assertEqual(d["start_line"], 10)
        self.assertEqual(d["end_line"], 25)
        self.assertIn("function example", d["code"])
        self.assertEqual(d["chunk_type"], "function")
        self.assertEqual(d["name"], "example")

    def test_ast_function_chunking(self) -> None:
        """Verify AST chunker extracts functions with start/end lines."""
        code = (
            "// Top comment\n"
            "function add(a, b) {\n"
            "  return a + b;\n"
            "}\n"
        )
        chunks = self.chunker.chunk_code(code, "math.js", "javascript")
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        self.assertGreaterEqual(len(func_chunks), 1)
        fc = func_chunks[0]
        self.assertEqual(fc.file, "math.js")
        self.assertEqual(fc.name, "add")
        self.assertIn("function add", fc.code)
        self.assertGreaterEqual(fc.start_line, 1)
        self.assertLessEqual(fc.end_line, 4)

    def test_ast_class_and_method_chunking(self) -> None:
        """Verify AST chunker extracts classes and individual methods."""
        code = (
            "export class Calculator {\n"
            "  multiply(x, y) {\n"
            "    return x * y;\n"
            "  }\n"
            "}\n"
        )
        chunks = self.chunker.chunk_code(code, "calc.js", "javascript")
        types = [c.chunk_type for c in chunks]
        self.assertIn("class", types)
        self.assertIn("method", types)

        method_chunk = next(c for c in chunks if c.chunk_type == "method")
        self.assertIn("multiply", method_chunk.name)
        self.assertIn("return x * y;", method_chunk.code)

    def test_ast_interface_chunking(self) -> None:
        """Verify TypeScript interfaces are chunked properly."""
        code = (
            "export interface UserProfile {\n"
            "  id: string;\n"
            "  username: string;\n"
            "}\n"
        )
        chunks = self.chunker.chunk_code(code, "user.ts", "typescript")
        interface_chunks = [c for c in chunks if c.chunk_type == "interface"]
        self.assertEqual(len(interface_chunks), 1)
        self.assertEqual(interface_chunks[0].name, "UserProfile")

    def test_line_chunking_fallback(self) -> None:
        """Verify line-based chunking fallback handles broken syntax safely."""
        broken_code = "function broken( {\nlet a = ;\nif ("
        chunks = self.chunker.chunk_code(broken_code, "broken.js", "javascript")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "lines")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 3)
        self.assertEqual(chunks[0].code, broken_code)

    def test_line_chunking_window_and_overlap(self) -> None:
        """Verify line-based chunking correctly partitions multi-line files."""
        lines = [f"line {i}" for i in range(1, 51)]
        code = "\n".join(lines)
        chunks = self.chunker.chunk_lines("test.txt", code, chunk_size=20, overlap=5)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 20)
        self.assertEqual(chunks[1].start_line, 16)  # 20 - 5 + 1
        self.assertEqual(chunks[1].end_line, 35)

    def test_docstring_attachment(self) -> None:
        """Verify JSDoc comments are preserved with the declaration."""
        code = (
            "/**\n"
            " * Computes discount.\n"
            " */\n"
            "function getDiscount(price) {\n"
            "  return price * 0.1;\n"
            "}\n"
        )
        chunks = self.chunker.chunk_code(code, "discount.js", "javascript")
        func_chunk = next(c for c in chunks if c.chunk_type == "function")
        self.assertIn("Computes discount", func_chunk.code)
        self.assertEqual(func_chunk.start_line, 1)


class TestEmbeddings(unittest.TestCase):
    """Test suite for Gemini and Mock embedding providers."""

    def test_gemini_missing_key_fails_clearly(self) -> None:
        """Verify GeminiEmbeddingProvider fails with actionable message when key is missing."""
        import os

        old_key = os.environ.pop("GEMINI_API_KEY", None)
        old_gkey = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            with self.assertRaises(ValueError) as ctx:
                GeminiEmbeddingProvider(api_key=None)
            self.assertIn("GEMINI_API_KEY is not configured", str(ctx.exception))
        finally:
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key
            if old_gkey:
                os.environ["GOOGLE_API_KEY"] = old_gkey

    def test_mock_provider_embedding_properties(self) -> None:
        """Verify MockEmbeddingProvider generates normalized float32 vectors."""
        mock = MockEmbeddingProvider(dimension=512)
        self.assertEqual(mock.dimension, 512)

        texts = ["function authenticate() {}", "class Database {}"]
        vecs = mock.embed_texts(texts)
        self.assertEqual(vecs.shape, (2, 512))
        self.assertEqual(vecs.dtype, np.float32)

        # Verify unit normalization
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)

    def test_mock_query_embedding(self) -> None:
        """Verify embed_query produces shape (1, dimension)."""
        mock = MockEmbeddingProvider(dimension=256)
        q_vec = mock.embed_query("test query")
        self.assertEqual(q_vec.shape, (1, 256))
        self.assertEqual(q_vec.dtype, np.float32)


class TestFAISSIndex(unittest.TestCase):
    """Test suite for FAISSIndex."""

    def setUp(self) -> None:
        self.dim = 64
        self.index = FAISSIndex(dimension=self.dim, metric="cosine")

    def test_add_and_search(self) -> None:
        """Verify vectors and metadata can be added and searched."""
        v1 = np.zeros(self.dim, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(self.dim, dtype=np.float32)
        v2[1] = 1.0
        vecs = np.vstack([v1, v2])

        metadata = [
            {"file": "src/auth.ts", "start_line": 1, "end_line": 10, "code": "code1"},
            {"file": "src/db.ts", "start_line": 1, "end_line": 20, "code": "code2"},
        ]

        self.index.add(vecs, metadata)
        self.assertEqual(self.index.total, 2)

        # Query close to v1
        q = np.zeros(self.dim, dtype=np.float32)
        q[0] = 1.0

        results = self.index.search(q, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["file"], "src/auth.ts")
        self.assertAlmostEqual(results[0]["score"], 1.0, places=4)
        self.assertEqual(results[1]["file"], "src/db.ts")

    def test_save_and_load(self) -> None:
        """Verify index and metadata persist to disk and reload identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            index_path = tmp_path / "test.faiss"

            v = np.random.randn(5, self.dim).astype(np.float32)
            meta = [
                {"file": f"src/f{i}.ts", "start_line": i, "end_line": i + 5, "code": f"code_{i}"}
                for i in range(5)
            ]
            self.index.add(v, meta)

            self.index.save(index_path)
            self.assertTrue(index_path.exists())
            self.assertTrue(index_path.with_suffix(".meta.json").exists())

            loaded = FAISSIndex.load(index_path)
            self.assertEqual(loaded.total, 5)
            self.assertEqual(loaded.dimension, self.dim)
            self.assertEqual(len(loaded.metadata), 5)
            self.assertEqual(loaded.metadata[0]["file"], "src/f0.ts")


class TestSemanticCodeSearchIntegration(unittest.TestCase):
    """End-to-end integration test connecting Scanner, Chunker, and Search."""

    def test_end_to_end_search_pipeline(self) -> None:
        """Verify complete pipeline from SourceFiles to search results with scores."""
        sf1 = SourceFile(
            path="src/auth.ts",
            language="typescript",
            content=(
                "export async function authenticateUser(email: string): Promise<boolean> {\n"
                "  return true;\n"
                "}\n"
            ),
            line_count=3,
        )
        sf2 = SourceFile(
            path="src/tax.js",
            language="javascript",
            content=(
                "function calculateTax(amount, rate) {\n"
                "  return amount * rate;\n"
                "}\n"
            ),
            line_count=3,
        )

        provider = MockEmbeddingProvider(dimension=256)
        searcher = SemanticCodeSearch(embedding_provider=provider)
        index = searcher.index_repository([sf1, sf2])
        self.assertGreaterEqual(index.total, 2)

        results = searcher.search("Where is authentication implemented?", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["file"], "src/auth.ts")
        self.assertIn("score", results[0])
        self.assertIn("code", results[0])
        self.assertIn("start_line", results[0])
        self.assertIn("end_line", results[0])


if __name__ == "__main__":
    unittest.main()
