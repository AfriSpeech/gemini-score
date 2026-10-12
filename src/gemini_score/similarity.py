"""Fast CPU sentence similarity using static Model2Vec embeddings (BGE / potion)."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    from model2vec import StaticModel
except ImportError:  # pragma: no cover
    StaticModel = None

logger = logging.getLogger(__name__)

# Default model: BAAI/bge-base-en-v1.5 distilled to static Model2Vec
DEFAULT_SIMILARITY_MODEL = "minishlab/M2V_base_output"


class SentenceSimilarity:
    """Computes fast CPU-optimized semantic sentence similarity using Model2Vec.

    Model2Vec produces high-quality static token/sentence embeddings with zero
    heavy neural network overhead, running hundreds of times faster on CPU than
    standard transformer models.
    """

    _cache: dict[str, "SentenceSimilarity"] = {}

    def __init__(self, model_name: str = DEFAULT_SIMILARITY_MODEL) -> None:
        """Initializes the similarity model.

        Parameters
        ----------
        model_name: str
            HuggingFace model ID or local directory for a Model2Vec model.
            Defaults to 'minishlab/M2V_base_output' (distilled BGE-base-en-v1.5).
        """
        if StaticModel is None:  # pragma: no cover
            raise ImportError(
                "model2vec is required for sentence similarity. "
                "Install it using `pip install model2vec`."
            )

        self.model_name = model_name
        logger.info("Loading static sentence embedding model: %s", model_name)
        self.model = StaticModel.from_pretrained(model_name)
        # Probe embedding dimension
        probe = self.model.encode(["probe"])
        self.dim = int(probe.shape[1])

    @classmethod
    def get_instance(cls, model_name: str = DEFAULT_SIMILARITY_MODEL) -> "SentenceSimilarity":
        """Returns a cached instance for the given model_name to avoid reloading."""
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name=model_name)
        return cls._cache[model_name]

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encodes a sequence of strings into static sentence embeddings.

        Parameters
        ----------
        texts: Sequence[str]
            List of sentences to encode.
        batch_size: int
            Batch size for chunked processing.
        normalize: bool
            If True, L2-normalizes the vectors so dot products equal cosine similarity.

        Returns
        -------
        np.ndarray
            Array of shape (len(texts), embedding_dim) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        # Sanitize texts (replace None or non-str with empty string)
        cleaned = [str(t) if t is not None else "" for t in texts]

        # Model2Vec supports batch encode directly
        if len(cleaned) <= batch_size:
            embeddings = self.model.encode(list(cleaned))
        else:
            chunks = []
            for i in range(0, len(cleaned), batch_size):
                chunk = cleaned[i : i + batch_size]
                chunks.append(self.model.encode(list(chunk)))
            embeddings = np.concatenate(chunks, axis=0)

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero
            zero_mask = norms < 1e-9
            norms = np.where(zero_mask, 1.0, norms)
            embeddings = embeddings / norms
            embeddings[zero_mask.squeeze(1)] = 0.0

        return embeddings

    def compute_similarity(
        self,
        translations: Sequence[str],
        references: Sequence[Union[str, Sequence[str]]],
        scale: str = "percent",
    ) -> np.ndarray:
        """Computes cosine similarity between translations and references.

        Supports multi-reference evaluation: if a sample has multiple references,
        the highest similarity across all references is taken (standard in MT metrics).

        Parameters
        ----------
        translations: Sequence[str]
            List of generated/back-translated English sentences.
        references: Sequence[str | Sequence[str]]
            List of reference English sentences (single or multiple per candidate).
        scale: str
            Output scale:
            - 'percent': Scores in range [0.0, 100.0] (default, like BLEU)
            - 'unit': Scores in range [0.0, 1.0]
            - 'raw': Raw cosine similarity in range [-1.0, 1.0]

        Returns
        -------
        np.ndarray
            Per-sample similarity scores of shape (len(translations),).
        """
        n = len(translations)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        if len(references) != n:
            raise ValueError(
                f"Mismatch: {n} translations provided but {len(references)} references given."
            )

        # Check if references are 1-to-1 or multi-reference
        is_multi_ref = any(isinstance(r, (list, tuple)) for r in references)

        if not is_multi_ref:
            # 1-to-1 matching
            ref_list = [str(r) for r in references]
            t_embs = self.encode(translations, normalize=True)
            r_embs = self.encode(ref_list, normalize=True)
            # Dot product of normalized vectors
            scores = np.sum(t_embs * r_embs, axis=1)
        else:
            # Multi-reference matching: encode all references efficiently
            t_embs = self.encode(translations, normalize=True)
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
                sims = np.dot(sample_r, t_embs[i])
                scores[i] = float(np.max(sims)) if len(sims) > 0 else 0.0

        scores = self._scale_scores(scores, scale=scale)
        return scores

    @staticmethod
    def _scale_scores(scores: np.ndarray, scale: str) -> np.ndarray:
        """Scales cosine similarity values based on the requested format."""
        scale = scale.lower()
        if scale == "raw":
            return np.clip(scores, -1.0, 1.0)
        elif scale == "unit":
            return np.clip(scores, 0.0, 1.0)
        elif scale == "percent":
            # Scale to 0-100 (clipping negative similarities to 0)
            clipped = np.clip(scores, 0.0, 1.0)
            return clipped * 100.0
        else:
            raise ValueError(
                f"Unknown scale '{scale}'. Options are 'percent', 'unit', or 'raw'."
            )
