"""
Semantic Code Search Module (Member 2).

Provides:
- CodeChunker & CodeChunk for AST and line-based code segmentation.
- GeminiEmbeddingProvider & MockEmbeddingProvider for vector generation.
- FAISSIndex & SemanticCodeSearch for vector indexing and semantic retrieval.
"""

from .chunker import CodeChunk, CodeChunker
from .embeddings import (
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODEL,
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    MockEmbeddingProvider,
    create_embedding_provider,
)
from .faiss_search import FAISSIndex, SemanticCodeSearch

__all__ = [
    "CodeChunk",
    "CodeChunker",
    "BaseEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "MockEmbeddingProvider",
    "create_embedding_provider",
    "DEFAULT_GEMINI_MODEL",
    "FALLBACK_GEMINI_MODEL",
    "FAISSIndex",
    "SemanticCodeSearch",
]
