"""Semantic sentence similarity using Gemini multilingual text embeddings."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

# Default latest Gemini multilingual text embedding model
DEFAULT_MODEL = os.environ.get(
    "GEMINI_MODEL",
    os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-002"),
)
DEFAULT_EMBEDDING_MODEL = DEFAULT_MODEL


class GeminiAuthError(ValueError):
    """Raised when no valid Gemini API key is found."""


def _pseudo_embed(text: str, dim: int = 64) -> np.ndarray:
    """Deterministic hash-based pseudo embedding for offline testing / mock clients."""
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()
    if not words:
        return vec
    for w in words:
        idx = abs(hash(w)) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-9)


class GeminiEmbedder:
    """Computes semantic multilingual embeddings using Google Gemini's Embedding API.

    Directly embeds cross-lingual input pairs into Gemini's shared multilingual vector
    space, allowing direct semantic similarity calculation without translation overhead.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = 64,
        max_retries: int = 4,
        backoff_factor: float = 2.0,
        enable_cache: bool = True,
        client: Optional[Any] = None,
    ) -> None:
        """Initializes the Gemini embedder.

        Parameters
        ----------
        api_key: str, optional
            Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY from environment.
        model: str
            Gemini embedding model (default: 'gemini-embedding-002').
        batch_size: int
            Number of sentences per embed_content request.
        max_retries: int
            Maximum retry attempts on transient / rate limit errors.
        backoff_factor: float
            Exponential backoff factor for retries.
        enable_cache: bool
            Whether to cache sentence embeddings in memory.
        client: Any, optional
            Pre-configured Google GenAI client instance.
        """
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor
        self.enable_cache = enable_cache
        self._cache: Dict[str, np.ndarray] = {}

        if client is not None:
            self.client = client
        else:
            resolved_key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not resolved_key:
                raise GeminiAuthError(
                    "No Gemini API key found. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                    "in your environment, or pass `api_key='...'`."
                )
            self.client = self._init_client(resolved_key)

    @staticmethod
    def _init_client(api_key: str) -> Any:
        """Initializes the GenAI client."""
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except ImportError:  # pragma: no cover
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                return genai_legacy
            except ImportError:
                raise ImportError(
                    "Google GenAI SDK is required. "
                    "Please install `google-genai` using `pip install google-genai`."
                )

    def _call_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Calls Gemini embed_content API with retry logic."""
        wait_time = 1.0

        for attempt in range(self.max_retries + 1):
            try:
                # Modern google.genai Client
                if hasattr(self.client, "models") and hasattr(self.client.models, "embed_content"):
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=texts,
                    )
                    vectors: List[np.ndarray] = []
                    for item in response.embeddings:
                        vals = getattr(item, "values", item)
                        vectors.append(np.asarray(vals, dtype=np.float32))
                    return vectors

                # Legacy google.generativeai fallback
                elif hasattr(self.client, "embed_content"):
                    response = self.client.embed_content(
                        model=self.model if "/" in self.model else f"models/{self.model}",
                        content=texts,
                    )
                    raw_embs = response.get("embedding", [])
                    return [np.asarray(v, dtype=np.float32) for v in raw_embs]

                # Mock / callable client
                elif callable(self.client):
                    import inspect
                    sig = inspect.signature(self.client)
                    if "is_embed" in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    ):
                        res = self.client(texts, is_embed=True)
                        return [np.asarray(v, dtype=np.float32) for v in res]
                    else:
                        return [_pseudo_embed(t) for t in texts]

                raise TypeError(f"Unrecognized client for embeddings: {type(self.client)}")

            except Exception as exc:
                err_msg = str(exc).lower()
                is_rate_limit = any(
                    token in err_msg
                    for token in ["429", "rate", "resource_exhausted", "quota", "overloaded"]
                )
                is_transient = is_rate_limit or any(
                    token in err_msg for token in ["503", "timeout", "unavailable"]
                )

                if attempt < self.max_retries and is_transient:
                    logger.warning(
                        "Gemini Embedding API failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    wait_time *= self.backoff_factor
                else:
                    logger.error("Gemini Embedding API failed permanently: %s", exc)
                    raise RuntimeError(f"Failed to generate Gemini embeddings: {exc}") from exc

        raise RuntimeError("Gemini Embedding call exceeded max retries.")

    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generates multilingual embeddings for a sequence of texts.

        Parameters
        ----------
        texts: Sequence[str]
            Sentences to encode.
        batch_size: int, optional
            Batch size chunking. Defaults to self.batch_size.
        normalize: bool
            Whether to L2-normalize vectors for cosine similarity.

        Returns
        -------
        np.ndarray
            Array of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        bs = batch_size or self.batch_size
        cleaned = [str(t) if t is not None else "" for t in texts]
        results: List[Optional[np.ndarray]] = [None] * len(cleaned)
        uncached_indices: List[int] = []

        for i, text in enumerate(cleaned):
            if not text.strip():
                uncached_indices.append(i)
            elif self.enable_cache and text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_texts = [cleaned[i] for i in uncached_indices]
            for start in range(0, len(uncached_texts), bs):
                chunk_indices = uncached_indices[start : start + bs]
                chunk_texts = uncached_texts[start : start + bs]

                valid_chunk_texts = [t if t.strip() else " " for t in chunk_texts]
                raw_vectors = self._call_embed(valid_chunk_texts)

                for orig_idx, text_str, vec in zip(chunk_indices, chunk_texts, raw_vectors):
                    if not text_str.strip():
                        vec = np.zeros_like(vec)
                    results[orig_idx] = vec
                    if self.enable_cache and text_str.strip():
                        self._cache[text_str] = vec

        matrix = np.asarray([r for r in results if r is not None], dtype=np.float32)

        if normalize and len(matrix) > 0:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            zero_mask = norms < 1e-9
            norms = np.where(zero_mask, 1.0, norms)
            matrix = matrix / norms
            matrix[zero_mask.squeeze(1)] = 0.0

        return matrix

    def compute_similarity(
        self,
        predictions: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
        scale: str = "percent",
    ) -> np.ndarray:
        """Computes semantic cosine similarity between predictions and references.

        Parameters
        ----------
        predictions: Sequence[str]
            Target hypothesis translations.
        references: Sequence[str | Sequence[str]]
            Reference translations (or source texts). Supports multiple references.
        scale: str
            Output scale: 'percent' (0-100, default), 'unit' (0-1), or 'raw' (-1 to 1).

        Returns
        -------
        np.ndarray
            Per-sample similarity scores.
        """
        n = len(predictions)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        if len(references) != n:
            raise ValueError(
                f"Mismatch: {n} predictions provided but {len(references)} references given."
            )

        is_multi_ref = any(isinstance(r, (list, tuple)) for r in references)

        if not is_multi_ref:
            ref_list = [str(r) for r in references]
            p_embs = self.encode(predictions, normalize=True)
            r_embs = self.encode(ref_list, normalize=True)
            scores = np.sum(p_embs * r_embs, axis=1)
        else:
            p_embs = self.encode(predictions, normalize=True)
            ref_slices: list[slice] = []
            flat_refs: list[str] = []
            for r in references:
                if isinstance(r, (list, tuple)):
                    r_strs = [str(item) for item in r]
                    start = len(flat_refs)
                    flat_refs.extend(r_strs if r_strs else [""])
                    ref_slices.append(slice(start, len(flat_refs)))
                else:
                    start = len(flat_refs)
                    flat_refs.append(str(r))
                    ref_slices.append(slice(start, len(flat_refs)))

            flat_r_embs = self.encode(flat_refs, normalize=True)
            scores = np.zeros(n, dtype=np.float32)
            for i, sl in enumerate(ref_slices):
                sample_r = flat_r_embs[sl]
                sims = np.dot(sample_r, p_embs[i])
                scores[i] = float(np.max(sims)) if len(sims) > 0 else 0.0

        return self._scale_scores(scores, scale=scale)

    @staticmethod
    def _scale_scores(scores: np.ndarray, scale: str) -> np.ndarray:
        """Scales cosine similarity values."""
        scale = scale.lower()
        if scale == "raw":
            return np.clip(scores, -1.0, 1.0)
        elif scale == "unit":
            return np.clip(scores, 0.0, 1.0)
        elif scale == "percent":
            clipped = np.clip(scores, 0.0, 1.0)
            return clipped * 100.0
        else:
            raise ValueError(
                f"Unknown scale '{scale}'. Options are 'percent', 'unit', or 'raw'."
            )


# Backward-compatibility aliases
GeminiEmbeddings = GeminiEmbedder
SentenceSimilarity = GeminiEmbedder
