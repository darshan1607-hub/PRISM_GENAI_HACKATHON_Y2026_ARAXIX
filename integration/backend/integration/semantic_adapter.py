"""Thin adapter over Member 2 (SemanticCodeSearch). Does not reimplement search."""
import os
from pathlib import Path
from typing import List
from backend.integration.evidence import NormalizedEvidence, SourceType, from_semantic_result


class SemanticUnavailable(RuntimeError):
    pass


class MockSemanticAdapter:
    def search(self, query: str, top_k: int = 3) -> List[NormalizedEvidence]:
        return [NormalizedEvidence(
            "[MOCK] src/settings/bluetooth.js", 10, 16,
            "// MOCK DATA: not from a real repository\nexport const BT_LINK = 'app://settings/bluetooth';",
            SourceType.SEMANTIC, "BT_LINK", 0.9, True, {"note": "development mock"})]


class RealSemanticAdapter:
    def __init__(self, repo_path: str):
        self.repo = Path(repo_path)
        self._engine = None

    def _build(self):
        if not self.repo.is_dir():
            raise SemanticUnavailable(f"Invalid repository path: {self.repo}")
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SemanticUnavailable("GEMINI_API_KEY not set; cannot build embeddings.")
        try:
            from backend.structural import RepositoryScanner, TreeSitterParser
            from backend.semantic import CodeChunker, GeminiEmbeddingProvider, SemanticCodeSearch
            files = RepositoryScanner(self.repo).scan()
            if not files:
                raise SemanticUnavailable("No supported JS/TS source files in repository.")
            parser = TreeSitterParser()
            chunker = CodeChunker(parser=parser)
            engine = SemanticCodeSearch(embedding_provider=GeminiEmbeddingProvider(),
                                        chunker=chunker, metric="cosine")
            engine.index_chunks(chunker.chunk_repository(files, parser=parser))
            self._engine = engine
        except SemanticUnavailable:
            raise
        except Exception as e:  # noqa
            raise SemanticUnavailable(f"Semantic index build failed: {e}") from e

    def search(self, query: str, top_k: int = 3) -> List[NormalizedEvidence]:
        if self._engine is None:
            self._build()
        try:
            return [from_semantic_result(r) for r in self._engine.search(query, top_k=top_k)]
        except Exception as e:
            raise SemanticUnavailable(f"Semantic search failed: {e}") from e
