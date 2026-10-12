"""GeminiScore: Semantic MT evaluation metric for African languages.

Evaluates Machine Translation hypotheses by translating African language predictions
to English with Google Gemini and computing semantic sentence similarity with
fast CPU-optimized static Model2Vec embeddings (BGE / potion).
"""

from .core import GeminiScore, compute, score
from .languages import (
    LanguageInfo,
    is_supported_language,
    list_supported_languages,
    resolve_language,
    search_languages,
)
from .models import GeminiScoreResult, get_performance_tier
from .similarity import DEFAULT_SIMILARITY_MODEL, SentenceSimilarity
from .translator import (
    DEFAULT_GEMINI_MODEL,
    MINIMUM_GEMINI_VERSION,
    GeminiAuthError,
    GeminiTranslationError,
    GeminiTranslator,
    validate_gemini_model,
)

__version__ = "0.1.0"

__all__ = [
    "GeminiScore",
    "GeminiScoreResult",
    "get_performance_tier",
    "LanguageInfo",
    "SentenceSimilarity",
    "GeminiTranslator",
    "GeminiAuthError",
    "GeminiTranslationError",
    "score",
    "compute",
    "resolve_language",
    "list_supported_languages",
    "search_languages",
    "is_supported_language",
    "validate_gemini_model",
    "DEFAULT_GEMINI_MODEL",
    "MINIMUM_GEMINI_VERSION",
    "DEFAULT_SIMILARITY_MODEL",
    "__version__",
]
