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

    def _resolve_repo(self) -> Path:
        if self.repo.is_dir():
            return self.repo
        for candidate in ("semantics/demo_repo", "structural/demo_repo", "demo_repo"):
            cp = Path(candidate)
            if cp.is_dir():
                return cp
        return self.repo

    def _build(self):
        target_repo = self._resolve_repo()
        if not target_repo.is_dir():
            raise SemanticUnavailable(f"Invalid repository path: {self.repo}")

        try:
            from backend.structural import RepositoryScanner, TreeSitterParser
            from backend.semantic import (
                CodeChunker,
                SemanticCodeSearch,
                GeminiEmbeddingProvider,
                MockEmbeddingProvider,
            )

            files = RepositoryScanner(target_repo).scan()
            if not files:
                raise SemanticUnavailable(f"No supported JS/TS source files in repository: {target_repo}")

            parser = TreeSitterParser()
            chunker = CodeChunker(parser=parser)

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key and api_key.strip():
                provider = GeminiEmbeddingProvider(api_key=api_key.strip())
            else:
                provider = MockEmbeddingProvider()

            engine = SemanticCodeSearch(
                embedding_provider=provider,
                chunker=chunker,
                metric="cosine",
            )
            chunks = chunker.chunk_repository(files, parser=parser)
            engine.index_chunks(chunks)
            self._engine = engine
        except SemanticUnavailable:
            raise
        except Exception as e:
            raise SemanticUnavailable(f"Semantic index build failed: {e}") from e

    def search(self, query: str, top_k: int = 3) -> List[NormalizedEvidence]:
        if self._engine is None:
            self._build()
        try:
            return [from_semantic_result(r) for r in self._engine.search(query, top_k=top_k)]
        except Exception as e:
            raise SemanticUnavailable(f"Semantic search failed: {e}") from e