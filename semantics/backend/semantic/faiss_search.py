"""
FAISS Vector Search Module (Member 2 - Semantic Code Search).

Provides vector indexing, persistence, natural language querying,
and metadata management for semantic code retrieval.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import logging
import os

import faiss
import numpy as np

from .chunker import CodeChunk, CodeChunker
from .embeddings import BaseEmbeddingProvider, GeminiEmbeddingProvider, MockEmbeddingProvider

logger = logging.getLogger(__name__)


class FAISSIndex:
    """Manages a FAISS vector index linked 1-to-1 with source code metadata.

    Maintains vector-to-metadata association so that every search result
    returns the exact file path, start line, end line, and code snippet.
    """

    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
    ) -> None:
        """Initialize FAISS index.

        Args:
            dimension: Dimensionality of vector space.
            metric: Distance metric ('cosine' or 'l2'). Cosine similarity
                    uses IndexFlatIP with L2 normalized vectors.
        """
        self.dimension = dimension
        self.metric = metric.lower()

        if self.metric == "cosine":
            self.index: faiss.Index = faiss.IndexFlatIP(dimension)
        elif self.metric == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported metric: '{metric}'. Choose 'cosine' or 'l2'.")

        self.metadata: List[Dict[str, Any]] = []

    @property
    def total(self) -> int:
        """Total number of vectors indexed."""
        return self.index.ntotal

    def add(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Add vectors and their associated code metadata to the index.

        Args:
            vectors: 2D numpy array of shape (N, dimension) with dtype float32.
            metadata: List of N metadata dictionaries corresponding to vectors.

        Raises:
            ValueError: If vector count does not match metadata length or dimension mismatch.
        """
        if len(vectors) == 0:
            return

        if len(vectors) != len(metadata):
            raise ValueError(
                f"Mismatch: received {len(vectors)} vectors but {len(metadata)} metadata items."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}."
            )

        # Ensure contiguous float32
        vecs_to_add = np.ascontiguousarray(vectors, dtype=np.float32)

        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(vecs_to_add)

        self.index.add(vecs_to_add)
        self.metadata.extend(metadata)
        logger.info("Added %d vectors to FAISS index. Total: %d", len(vectors), self.total)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search index for the top-K most semantically similar code chunks.

        Args:
            query_vector: 2D numpy array of shape (1, dimension) with dtype float32.
            top_k: Maximum number of results to return.

        Returns:
            List of result dictionaries containing:
                file: File path
                start_line: Start line number
                end_line: End line number
                code: Code snippet
                score: Similarity score (cosine similarity or L2 distance)
        """
        if self.total == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        if query_vector.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query_vector.shape[1]} does not match index {self.dimension}."
            )

        q_vec = np.ascontiguousarray(query_vector, dtype=np.float32)

        if self.metric == "cosine":
            faiss.normalize_L2(q_vec)

        k = min(max(1, top_k), self.total)
        distances, indices = self.index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(dist)
            results.append(item)

        return results

    def save(
        self,
        index_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Save FAISS index and metadata to disk.

        Args:
            index_path: Path to write the FAISS binary index (.faiss).
            metadata_path: Path to write the JSON metadata. If None, derived from index_path.
        """
        idx_p = Path(index_path)
        idx_p.parent.mkdir(parents=True, exist_ok=True)

        meta_p = (
            Path(metadata_path)
            if metadata_path is not None
            else idx_p.with_suffix(".meta.json")
        )

        faiss.write_index(self.index, str(idx_p))

        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "metric": self.metric,
                    "count": self.total,
                    "metadata": self.metadata,
                },
                f,
                indent=2,
            )

        logger.info("Saved FAISS index to '%s' and metadata to '%s'", idx_p, meta_p)

    @classmethod
    def load(
        cls,
        index_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
    ) -> "FAISSIndex":
        """Load a saved FAISS index and its metadata from disk.

        Args:
            index_path: Path to the FAISS binary index.
            metadata_path: Path to the JSON metadata file.

        Returns:
            Reconstructed FAISSIndex instance.
        """
        idx_p = Path(index_path)
        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found: {idx_p}")

        meta_p = (
            Path(metadata_path)
            if metadata_path is not None
            else idx_p.with_suffix(".meta.json")
        )
        if not meta_p.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_p}")

        raw_index = faiss.read_index(str(idx_p))

        with open(meta_p, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        dimension = meta_data.get("dimension", raw_index.d)
        metric = meta_data.get("metric", "cosine")

        instance = cls(dimension=dimension, metric=metric)
        instance.index = raw_index
        instance.metadata = meta_data.get("metadata", [])

        logger.info(
            "Loaded FAISS index with %d items from '%s'",
            instance.total,
            idx_p,
        )
        return instance


class SemanticCodeSearch:
    """High-level semantic search coordinator for code repositories.

    Integrates:
    - CodeChunker (AST + line fallback)
    - BaseEmbeddingProvider (Gemini Embedding 2 or Mock for testing)
    - FAISSIndex (vector search and metadata preservation)
    """

    def __init__(
        self,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        chunker: Optional[CodeChunker] = None,
        metric: str = "cosine",
    ) -> None:
        """Initialize SemanticCodeSearch.

        Args:
            embedding_provider: Provider instance (defaults to Gemini or Mock if unavailable).
            chunker: CodeChunker instance (instantiated if None).
            metric: Distance metric for FAISS ('cosine' or 'l2').
        """
        self.chunker = chunker or CodeChunker()
        self.metric = metric
        self.index: Optional[FAISSIndex] = None

        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            # Auto-detect whether GEMINI_API_KEY is available
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if api_key:
                self.embedding_provider = GeminiEmbeddingProvider(api_key=api_key)
            else:
                self.embedding_provider = MockEmbeddingProvider()

    def index_chunks(self, chunks: List[CodeChunk]) -> FAISSIndex:
        """Generate embeddings for chunks and build a new FAISS index.

        Args:
            chunks: List of CodeChunk objects.

        Returns:
            Populated FAISSIndex.
        """
        if not chunks:
            raise ValueError("No code chunks provided for indexing.")

        dim = self.embedding_provider.dimension
        self.index = FAISSIndex(dimension=dim, metric=self.metric)

        # Generate embeddings
        vectors = self.embedding_provider.embed_chunks(chunks)

        # Extract metadata
        metadata = [c.to_dict() for c in chunks]

        # Add to FAISS index
        self.index.add(vectors=vectors, metadata=metadata)
        return self.index

    def index_repository(
        self,
        source_files: List[Any],
        parser: Optional[Any] = None,
    ) -> FAISSIndex:
        """Ingest repository source files, chunk them, and build the FAISS index.

        Directly reuses Member 1 SourceFile list and Member 3 TreeSitterParser.

        Args:
            source_files: List of SourceFile objects from Member 1 RepositoryScanner.
            parser: Optional TreeSitterParser from Member 3.

        Returns:
            Populated FAISSIndex.
        """
        chunks = self.chunker.chunk_repository(source_files, parser=parser)
        return self.index_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search for code chunks matching the natural language query.

        Args:
            query: Natural language question or search phrase.
            top_k: Number of relevant code chunks to return.

        Returns:
            List of result dictionaries with file, start_line, end_line, code, score.
        """
        if self.index is None or self.index.total == 0:
            raise RuntimeError(
                "Search index is empty. Call index_repository() or index_chunks() first."
            )

        query_vec = self.embedding_provider.embed_query(query)
        results = self.index.search(query_vector=query_vec, top_k=top_k)
        return results

    def save(self, storage_dir: Union[str, Path], name: str = "code_index") -> Tuple[Path, Path]:
        """Save search index and metadata to a directory."""
        if self.index is None:
            raise RuntimeError("Cannot save: No index has been built.")

        out_dir = Path(storage_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        index_file = out_dir / f"{name}.faiss"
        meta_file = out_dir / f"{name}.meta.json"

        self.index.save(index_file, meta_file)
        return index_file, meta_file

    def load(self, storage_dir: Union[str, Path], name: str = "code_index") -> FAISSIndex:
        """Load a previously saved index from a directory."""
        in_dir = Path(storage_dir)
        index_file = in_dir / f"{name}.faiss"
        meta_file = in_dir / f"{name}.meta.json"

        self.index = FAISSIndex.load(index_file, meta_file)
        return self.index
