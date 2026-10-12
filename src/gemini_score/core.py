"""Core evaluation API for GeminiScore."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .languages import LanguageInfo, resolve_language
from .models import GeminiScoreResult
from .similarity import DEFAULT_SIMILARITY_MODEL, SentenceSimilarity
from .translator import DEFAULT_GEMINI_MODEL, GeminiTranslator

logger = logging.getLogger(__name__)


class GeminiScore:
    """Evaluates African-language Machine Translation models against English references.

    It works by:
    1. Translating candidate African language text to English using Google Gemini API.
    2. Measuring semantic similarity between the back-translated English and the reference
       English using a CPU-fast static Model2Vec sentence embedding model (BGE / potion).

    This provides a semantically grounded, robust alternative to surface n-gram overlap
    metrics like BLEU, which struggle with morphology, dialect variation, and low-resource
    African language nuances.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        similarity_model: str = DEFAULT_SIMILARITY_MODEL,
        scale: str = "percent",
        batch_size: int = 16,
        max_workers: int = 4,
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
            Gemini model name (e.g. 'gemini-2.5-flash').
        similarity_model: str
            Model2Vec static embedding model name or path.
            Defaults to 'minishlab/M2V_base_output' (distilled BGE-base-en-v1.5).
        scale: str
            Output scoring scale: 'percent' (0-100, default), 'unit' (0-1), or 'raw' (-1 to 1).
        batch_size: int
            Number of sentences per prompt batch.
        max_workers: int
            Worker thread concurrency for translation.
        max_retries: int
            Maximum retry attempts on rate limits / network errors.
        enable_cache: bool
            Whether to cache translations in memory.
        client: Any, optional
            Pre-configured GenAI client or mock client.
        """
        self.model = model
        self.similarity_model_name = similarity_model
        self.scale = scale

        self.translator = GeminiTranslator(
            api_key=api_key,
            model=model,
            batch_size=batch_size,
            max_workers=max_workers,
            max_retries=max_retries,
            enable_cache=enable_cache,
            client=client,
        )
        self.similarity_engine = SentenceSimilarity.get_instance(model_name=similarity_model)

    def score(
        self,
        predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[Union[str, Sequence[str]]]],
        language: Union[str, LanguageInfo, Sequence[Union[str, LanguageInfo]]],
        scale: Optional[str] = None,
    ) -> GeminiScoreResult:
        """Computes GeminiScore for a set of predictions and English references.

        Parameters
        ----------
        predictions: str or Sequence[str]
            Target hypothesis sentence(s) generated in an African language.
        references: str, Sequence[str], or Sequence[Sequence[str]]
            Reference sentence(s) in English (e.g. original source or reference translations).
        language: str, LanguageInfo, or Sequence[str | LanguageInfo]
            African language name, ISO 639-3 code (e.g. 'yor', 'swh', 'twi'), or a list of
            languages corresponding to each sample in predictions.
        scale: str, optional
            Override output scale ('percent', 'unit', 'raw'). Defaults to self.scale.

        Returns
        -------
        GeminiScoreResult
            Comprehensive evaluation result with system-level and sentence-level scores.
        """
        # Normalize predictions to a list
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
                translations=[],
                languages=[],
                scale=scale or self.scale,
                model=self.model,
                similarity_model=self.similarity_model_name,
            )

        # Normalize references to a list
        if isinstance(references, str):
            ref_list: List[Union[str, Sequence[str]]] = [references]
        elif isinstance(references, (list, tuple)):
            if n == 1 and len(references) > 1 and all(isinstance(r, str) for r in references):
                # Single prediction evaluated against multiple references: references = [ref1, ref2]
                ref_list = [list(references)]
            else:
                ref_list = list(references)
        else:
            raise TypeError(f"Unsupported references type: {type(references).__name__}")

        if len(ref_list) != n:
            raise ValueError(
                f"Number of predictions ({n}) must match number of references ({len(ref_list)})."
            )

        # Normalize languages
        if isinstance(language, (str, LanguageInfo)):
            resolved_lang_objs = [resolve_language(language)] * n
        elif isinstance(language, (list, tuple)):
            if len(language) != n:
                raise ValueError(
                    f"Number of languages ({len(language)}) must match number of predictions ({n})."
                )
            resolved_lang_objs = [resolve_language(l) for l in language]
        else:
            raise TypeError(f"Unsupported language type: {type(language).__name__}")

        active_scale = scale or self.scale

        # Step 1: Translate African predictions to English via Gemini
        translations = self.translator.translate_batch(
            texts=pred_list,
            languages=resolved_lang_objs,
        )

        # Step 2: Compute semantic sentence similarity via Model2Vec
        raw_scores = self.similarity_engine.compute_similarity(
            translations=translations,
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
            translations=translations,
            languages=[l.prompt_name for l in resolved_lang_objs],
            scale=active_scale,
            model=self.model,
            similarity_model=self.similarity_model_name,
        )

    def compute(
        self,
        predictions: Union[str, Sequence[str]],
        references: Union[str, Sequence[Union[str, Sequence[str]]]],
        language: Union[str, LanguageInfo, Sequence[Union[str, LanguageInfo]]],
        scale: Optional[str] = None,
    ) -> Dict[str, Any]:
        """HF Evaluate-compatible method returning a metrics dictionary."""
        res = self.score(
            predictions=predictions,
            references=references,
            language=language,
            scale=scale,
        )
        return {
            "gemini_score": res.score,
            "scores": res.scores,
            "translations": res.translations,
        }


def score(
    predictions: Union[str, Sequence[str]],
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    language: Union[str, LanguageInfo, Sequence[Union[str, LanguageInfo]]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    similarity_model: str = DEFAULT_SIMILARITY_MODEL,
    scale: str = "percent",
    batch_size: int = 16,
    max_workers: int = 4,
    enable_cache: bool = True,
    client: Optional[Any] = None,
) -> GeminiScoreResult:
    """Convenience functional interface to evaluate machine translations.

    Parameters
    ----------
    predictions: str or Sequence[str]
        Target hypothesis sentence(s) in African language.
    references: str or Sequence[str | Sequence[str]]
        Reference English sentences.
    language: str, LanguageInfo, or Sequence[str | LanguageInfo]
        African language name(s) or ISO code(s) (resolved via afriso).
    api_key: str, optional
        Gemini API key. Defaults to GEMINI_API_KEY in environment.
    model: str
        Gemini model name.
    similarity_model: str
        Model2Vec static embedding model.
    scale: str
        Scoring scale ('percent', 'unit', 'raw').
    batch_size: int
        Batch size per Gemini prompt call.
    max_workers: int
        Thread concurrency.
    enable_cache: bool
        Whether to cache translations.
    client: Any, optional
        Pre-configured or mock GenAI client.

    Returns
    -------
    GeminiScoreResult
    """
    scorer = GeminiScore(
        api_key=api_key,
        model=model,
        similarity_model=similarity_model,
        scale=scale,
        batch_size=batch_size,
        max_workers=max_workers,
        enable_cache=enable_cache,
        client=client,
    )
    return scorer.score(
        predictions=predictions,
        references=references,
        language=language,
    )


def compute(
    predictions: Union[str, Sequence[str]],
    references: Union[str, Sequence[Union[str, Sequence[str]]]],
    language: Union[str, LanguageInfo, Sequence[Union[str, LanguageInfo]]],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Hugging Face Evaluate-style functional interface."""
    res = score(
        predictions=predictions,
        references=references,
        language=language,
        **kwargs,
    )
    return {
        "gemini_score": res.score,
        "scores": res.scores,
        "translations": res.translations,
    }
