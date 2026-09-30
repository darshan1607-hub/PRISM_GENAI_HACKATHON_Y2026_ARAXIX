"""
Embeddings Provider Module (Member 2 - Semantic Code Search).

Provides vector embeddings for code chunks and search queries using
Google's Gemini Embedding 2 model, with support for batching, FAISS-compatible
numpy representations, and a clearly labeled offline test provider.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import hashlib
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)

# Default model per specification
DEFAULT_GEMINI_MODEL = "gemini-embedding-2"
FALLBACK_GEMINI_MODEL = "text-embedding-004"


class BaseEmbeddingProvider(ABC):
    """Abstract base class for semantic embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of text strings.

        Args:
            texts: List of strings to embed.

        Returns:
            2D numpy array of shape (len(texts), dimension) with dtype float32.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single search query.

        Args:
            query: Natural language query string.

        Returns:
            2D numpy array of shape (1, dimension) with dtype float32.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of vectors produced by this provider."""
        pass

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Convenience method to embed a list of CodeChunk objects or strings.

        Args:
            chunks: List of CodeChunk objects (or strings).

        Returns:
            2D numpy array of shape (len(chunks), dimension) with dtype float32.
        """
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)

        texts: List[str] = []
        for ch in chunks:
            if hasattr(ch, "code"):
                # Prepend file path hint to code for richer contextual embedding
                prefix = f"// File: {ch.file}\n" if hasattr(ch, "file") and ch.file else ""
                texts.append(f"{prefix}{ch.code}")
            elif isinstance(ch, dict) and "code" in ch:
                prefix = f"// File: {ch.get('file', '')}\n" if ch.get("file") else ""
                texts.append(f"{prefix}{ch['code']}")
            else:
                texts.append(str(ch))

        return self.embed_texts(texts)


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Generates vector embeddings using Google's Gemini Embedding 2 model.

    Features:
    - Securely retrieves API key from GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable.
    - NEVER hard-codes or persists API keys.
    - Supports batching for high throughput on large repositories.
    - Produces FAISS-compatible float32 2D numpy arrays.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        batch_size: int = 32,
        output_dimensionality: Optional[int] = None,
    ) -> None:
        """Initialize the Gemini Embedding 2 provider.

        Args:
            api_key: Optional explicit API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env vars).
            model: Model name ('gemini-embedding-2' or 'text-embedding-004').
            batch_size: Number of texts per API request (default 32).
            output_dimensionality: Optional Matryoshka dimension constraint.

        Raises:
            ValueError: If no API key is found in environment or arguments.
            ImportError: If google-genai is not installed.
        """
        # Resolve API key securely
        resolved_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )

        if not resolved_key or not resolved_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is not configured.\n"
                "To enable Gemini Embedding 2, set the environment variable:\n"
                "  PowerShell:  $env:GEMINI_API_KEY = 'your_gemini_api_key'\n"
                "  Linux/macOS: export GEMINI_API_KEY='your_gemini_api_key'\n"
                "Alternatively, for offline testing without an API key, use MockEmbeddingProvider."
            )

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for Gemini embeddings. "
                "Please run: pip install google-genai"
            ) from exc

        self.api_key = resolved_key.strip()
        self.model = model
        self.batch_size = max(1, batch_size)
        self.output_dimensionality = output_dimensionality
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=self.api_key)
        self._cached_dimension: Optional[int] = output_dimensionality

    @property
    def dimension(self) -> int:
        """Vector dimensionality, inferred dynamically from model if not predefined."""
        if self._cached_dimension is not None:
            return self._cached_dimension
        # Probe dimension using a minimal query
        probe = self.embed_query("code")
        self._cached_dimension = probe.shape[1]
        return self._cached_dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of text strings in batches using Gemini Embedding 2."""
        if not texts:
            dim = self._cached_dimension or 768
            return np.empty((0, dim), dtype=np.float32)

        config = None
        if self.output_dimensionality is not None:
            config = self._types.EmbedContentConfig(
                output_dimensionality=self.output_dimensionality
            )

        all_vectors: List[List[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = self._client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=config,
                )
            except Exception as err:
                # Handle model name fallback if gemini-embedding-2 is not enabled on account
                if self.model == DEFAULT_GEMINI_MODEL and "not found" in str(err).lower():
                    logger.warning(
                        "Model '%s' not accessible, falling back to '%s'",
                        self.model,
                        FALLBACK_GEMINI_MODEL,
                    )
                    self.model = FALLBACK_GEMINI_MODEL
                    response = self._client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=config,
                    )
                else:
                    logger.error("Gemini embed_content failed: %s", err)
                    raise

            # Extract vectors from EmbedContentResponse
            if hasattr(response, "embeddings") and response.embeddings:
                for emb in response.embeddings:
                    if hasattr(emb, "values") and emb.values:
                        all_vectors.append(emb.values)
                    elif isinstance(emb, list):
                        all_vectors.append(emb)
            elif hasattr(response, "embedding") and response.embedding:
                emb = response.embedding
                values = emb.values if hasattr(emb, "values") else emb
                all_vectors.append(values)
            else:
                raise RuntimeError(f"Unexpected response format from Gemini API: {response}")

        arr = np.array(all_vectors, dtype=np.float32)
        if self._cached_dimension is None and arr.shape[0] > 0:
            self._cached_dimension = arr.shape[1]

        return arr

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for semantic similarity search."""
        return self.embed_texts([query])


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic hash-based embedding provider strictly for offline unit tests.

    [TEST MODE ONLY]
    This class is clearly designated for offline verification of the pipeline
    when GEMINI_API_KEY is not available. It does NOT make network calls and does
    NOT pretend to be real Gemini embeddings. It generates reproducible, normalized
    pseudo-semantic vectors using subword token hashing.
    """

    def __init__(self, dimension: int = 768) -> None:
        """Initialize the MockEmbeddingProvider.

        Args:
            dimension: Dimensionality of the synthetic vector space (default 768).
        """
        self._dimension = dimension
        print(
            "\n[TEST MODE ONLY] Initialized MockEmbeddingProvider.\n"
            "Generating deterministic synthetic hash embeddings for offline test execution.\n"
            "NOT real Gemini embeddings — set GEMINI_API_KEY to use Gemini Embedding 2.\n"
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokenize(self, text: str) -> List[str]:
        """Split text into words, camelCase tokens, and character n-grams."""
        # Split on non-alphanumeric
        raw_words = re.findall(r"[A-Za-z0-9]+", text)
        tokens: List[str] = []

        for word in raw_words:
            tokens.append(word.lower())
            # Split camelCase e.g. authenticateUser -> authenticate, user
            sub_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", word)
            for part in sub_parts:
                p_lower = part.lower()
                if p_lower != word.lower():
                    tokens.append(p_lower)

            # Character 3-grams for fuzzy subword matching (e.g. auth, db, log)
            w_lower = word.lower()
            if len(w_lower) >= 3:
                for k in range(len(w_lower) - 2):
                    tokens.append(w_lower[k : k + 3])

        return tokens

    def _embed_single(self, text: str) -> np.ndarray:
        """Generate a single normalized vector from tokens."""
        vec = np.zeros(self._dimension, dtype=np.float32)
        tokens = self._tokenize(text)

        if not tokens:
            # Deterministic non-zero vector
            vec[0] = 1.0
            return vec

        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self._dimension
            vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts using deterministic hashing."""
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        vectors = [self._embed_single(t) for t in texts]
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query using deterministic hashing."""
        return self.embed_texts([query])


def create_embedding_provider(
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    use_mock: bool = False,
    dimension: int = 768,
) -> BaseEmbeddingProvider:
    """Factory function to instantiate the appropriate embedding provider.

    Args:
        api_key: Optional explicit Gemini API key.
        model: Gemini model identifier.
        use_mock: If True, explicitly returns MockEmbeddingProvider.
        dimension: Dimension for MockEmbeddingProvider (default 768).

    Returns:
        Configured BaseEmbeddingProvider instance.
    """
    if use_mock:
        return MockEmbeddingProvider(dimension=dimension)

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        # Actionable error message
        raise ValueError(
            "GEMINI_API_KEY environment variable is not configured.\n"
            "To use real Gemini Embedding 2, set GEMINI_API_KEY:\n"
            "  PowerShell:  $env:GEMINI_API_KEY = 'your_api_key'\n"
            "  Linux/macOS: export GEMINI_API_KEY='your_api_key'\n"
            "For offline test mode, pass use_mock=True."
        )

    return GeminiEmbeddingProvider(api_key=key, model=model)
