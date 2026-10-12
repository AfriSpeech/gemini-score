"""GeminiScore: Semantic MT evaluation metric using Gemini multilingual text embeddings."""

from .core import GeminiScore, compute, score
from .models import GeminiScoreResult, get_performance_tier
from .similarity import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    GeminiAuthError,
    GeminiEmbedder,
    GeminiEmbeddings,
    SentenceSimilarity,
)

__version__ = "0.1.0"

__all__ = [
    "GeminiScore",
    "GeminiScoreResult",
    "get_performance_tier",
    "GeminiEmbedder",
    "GeminiEmbeddings",
    "SentenceSimilarity",
    "GeminiAuthError",
    "score",
    "compute",
    "DEFAULT_MODEL",
    "DEFAULT_EMBEDDING_MODEL",
    "__version__",
]
