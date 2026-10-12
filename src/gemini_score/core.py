"""Core evaluation API for GeminiScore."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .models import GeminiScoreResult
from .similarity import DEFAULT_MODEL, GeminiEmbedder

logger = logging.getLogger(__name__)


class GeminiScore:
    """Evaluates Machine Translation outputs against references using Gemini multilingual embeddings.

    Directly maps candidate predictions and reference translations (or source sentences)
    into Gemini's shared cross-lingual embedding space (default: 'gemini-embedding-002')
    and computes semantic cosine similarity without translation overhead or extra libraries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        scale: str = "percent",
        batch_size: int = 64,
        max_retries: int = 4,
        enable_cache: bool = True,
        client: Optional[Any] = None,
    ) -> None:
        """Initializes the GeminiScore evaluator.

        Parameters
        ----------
        api_key: str, optional
            Gemini API key. Defaults to GEMINI_API_KEY / GOOGLE_API_KEY from environment.
        model: str
            Gemini multilingual text embedding model (default: 'gemini-embedding-002').
        scale: str
            Output scoring scale: 'percent' (0-100, default), 'unit' (0-1), or 'raw' (-1 to 1).
        batch_size: int
            Number of sentences per embed_content request.
        max_retries: int
            Maximum retry attempts on rate limits / network errors.
        enable_cache: bool
            Whether to cache embeddings in memory.
        client: Any, optional
            Pre-configured GenAI client or mock client.
        """
        self.model = model
        self.scale = scale

        self.embedder = GeminiEmbedder(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_retries=max_retries,
            enable_cache=enable_cache,
            client=client,
        )

    def score(
        self,
        predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[Union[str, Sequence[str]]]],
        scale: Optional[str] = None,
    ) -> GeminiScoreResult:
        """Computes GeminiScore for a set of predictions and references.

        Parameters
        ----------
        predictions: str or Sequence[str]
            Target hypothesis sentence(s).
        references: str, Sequence[str], or Sequence[Sequence[str]]
            Reference sentence(s).
        scale: str, optional
            Override output scale ('percent', 'unit', 'raw'). Defaults to self.scale.

        Returns
        -------
        GeminiScoreResult
        """
        if isinstance(predictions, str):
            pred_list = [predictions]
        else:
            pred_list = list(predictions)

        n = len(pred_list)
        if n == 0:
            return GeminiScoreResult(
                score=0.0,
                scores=[],
                predictions=[],
                references=[],
                scale=scale or self.scale,
                model=self.model,
            )

        if isinstance(references, str):
            ref_list: List[Union[str, Sequence[str]]] = [references]
        elif isinstance(references, (list, tuple)):
            if n == 1 and len(references) > 1 and all(isinstance(r, str) for r in references):
                ref_list = [list(references)]
            else:
                ref_list = list(references)
        else:
            raise TypeError(f"Unsupported references type: {type(references).__name__}")

        if len(ref_list) != n:
            raise ValueError(
                f"Number of predictions ({n}) must match number of references ({len(ref_list)})."
            )

        active_scale = scale or self.scale

        # Compute semantic cosine similarity directly using Gemini multilingual embeddings
        raw_scores = self.embedder.compute_similarity(
            predictions=pred_list,
            references=ref_list,
            scale=active_scale,
        )

        sentence_scores = [float(s) for s in raw_scores]
        system_score = float(np.mean(sentence_scores)) if sentence_scores else 0.0

        return GeminiScoreResult(
            score=system_score,
            scores=sentence_scores,
            predictions=pred_list,
            references=ref_list,
            scale=active_scale,
            model=self.model,
        )

    def compute(
        self,
        predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[Union[str, Sequence[str]]]],
        scale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """HF Evaluate-compatible method returning a metrics dictionary."""
        res = self.score(
            predictions=predictions,
            references=references,
            scale=scale,
        )
        return {
            "gemini_score": res.score,
            "scores": res.scores,
        }

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Utility method to directly generate embeddings for a list of texts."""
        return self.embedder.encode(texts, normalize=normalize)


def score(
    predictions: Union[str, Sequence[str]],
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    scale: str = "percent",
    batch_size: int = 64,
    enable_cache: bool = True,
    client: Optional[Any] = None,
) -> GeminiScoreResult:
    """Convenience functional interface to evaluate machine translations.

    Parameters
    ----------
    predictions: str or Sequence[str]
        Target hypothesis sentence(s).
    references: str or Sequence[str | Sequence[str]]
        Reference sentence(s).
    api_key: str, optional
        Gemini API key. Defaults to GEMINI_API_KEY in environment.
    model: str
        Gemini embedding model (default: 'gemini-embedding-002').
    scale: str
        Scoring scale ('percent', 'unit', 'raw').
    batch_size: int
        Batch size per embed_content call.
    enable_cache: bool
        Whether to cache embeddings.
    client: Any, optional
        Pre-configured or mock GenAI client.

    Returns
    -------
    GeminiScoreResult
    """
    scorer = GeminiScore(
        api_key=api_key,
        model=model,
        scale=scale,
        batch_size=batch_size,
        enable_cache=enable_cache,
        client=client,
    )
    return scorer.score(predictions=predictions, references=references)


def compute(
    predictions: Union[str, Sequence[str]],
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hugging Face Evaluate-style functional interface."""
    res = score(predictions=predictions, references=references, **kwargs)
    return {
        "gemini_score": res.score,
        "scores": res.scores,
    }
