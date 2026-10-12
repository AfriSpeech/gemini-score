"""Data structures, aggregations, and performance tiers for GeminiScore."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def get_performance_tier(score: float, scale: str = "percent") -> Dict[str, str]:
    """Maps a GeminiScore to its qualitative performance tier and explanation.

    Parameters
    ----------
    score: float
        Aggregate or sentence-level GeminiScore.
    scale: str
        Scale used ('percent', 'unit', or 'raw').

    Returns
    -------
    dict
        Dictionary containing 'tier', 'grade', and 'description'.
    """
    scale_lower = scale.lower()
    if scale_lower == "percent":
        pct = score
    elif scale_lower == "unit":
        pct = score * 100.0
    elif scale_lower == "raw":
        pct = max(0.0, score) * 100.0
    else:
        pct = score

    if pct >= 85.0:
        return {
            "tier": "Excellent",
            "grade": "A",
            "description": "High-fidelity semantic match. Full meaning, entities, and nuances are preserved with high precision.",
        }
    elif pct >= 70.0:
        return {
            "tier": "Good",
            "grade": "B",
            "description": "High semantic quality. Core meaning is accurately conveyed with minor stylistic or acceptable lexical variation.",
        }
    elif pct >= 55.0:
        return {
            "tier": "Moderate",
            "grade": "C",
            "description": "Adequate translation. The overall gist is clear, but secondary clauses or specific terminology may be imprecise.",
        }
    elif pct >= 40.0:
        return {
            "tier": "Poor",
            "grade": "D",
            "description": "Fragmentary translation. Significant semantic drift; only partial keywords or disconnected fragments are preserved.",
        }
    else:
        return {
            "tier": "Severely Deficient",
            "grade": "F",
            "description": "Unusable translation. The original meaning is largely lost, contradictory, or completely hallucinated.",
        }


@dataclass
class GeminiScoreResult:
    """Evaluation result returned by GeminiScore.

    Attributes
    ----------
    score: float
        The overall aggregated system-level score (mean across all samples).
    scores: List[float]
        Per-sample semantic similarity scores.
    predictions: List[str]
        Input candidate hypothesis strings evaluated.
    references: List[Union[str, List[str]]]
        Reference sentences evaluated against.
    model: str
        Gemini multilingual embedding model used.
    scale: str
        Scoring scale ('percent', 'unit', or 'raw').
    """

    score: float
    scores: List[float]
    predictions: List[str]
    references: List[Union[str, List[str]]]
    model: str = "gemini-embedding-002"
    scale: str = "percent"

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "prediction": self.predictions[idx],
            "reference": self.references[idx],
            "score": self.scores[idx],
            "tier": get_performance_tier(self.scores[idx], self.scale)["tier"],
        }

    @property
    def tier(self) -> str:
        """Returns the qualitative performance tier of the aggregate system score."""
        return get_performance_tier(self.score, self.scale)["tier"]

    @property
    def grade(self) -> str:
        """Returns the letter grade (A, B, C, D, F) of the aggregate score."""
        return get_performance_tier(self.score, self.scale)["grade"]

    @property
    def tier_description(self) -> str:
        """Returns the explanatory interpretation of the aggregate score tier."""
        return get_performance_tier(self.score, self.scale)["description"]

    def __repr__(self) -> str:
        return (
            f"GeminiScoreResult(score={self.score:.2f}, "
            f"tier='{self.tier}', "
            f"num_samples={len(self.scores)}, "
            f"scale='{self.scale}', "
            f"model='{self.model}')"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result to a dictionary."""
        return {
            "score": self.score,
            "tier": self.tier,
            "grade": self.grade,
            "tier_description": self.tier_description,
            "scores": self.scores,
            "predictions": self.predictions,
            "references": self.references,
            "scale": self.scale,
            "model": self.model,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the result dictionary to a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_dataframe(self) -> Any:
        """Converts sample-level results into a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to convert results to a DataFrame. "
                "Install it with `pip install pandas`."
            )

        data = {
            "prediction": self.predictions,
            "reference": self.references,
            "score": self.scores,
            "tier": [get_performance_tier(s, self.scale)["tier"] for s in self.scores],
        }
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Returns a formatted textual summary of the evaluation."""
        if not self.scores:
            return "GeminiScore: 0 samples evaluated."

        import numpy as np

        arr = np.array(self.scores)
        unit_sym = "%" if self.scale == "percent" else ""
        tier_info = get_performance_tier(self.score, self.scale)

        lines = [
            "===========================================================",
            "               GeminiScore Evaluation Summary              ",
            "===========================================================",
            f"Final Aggregated Score : {self.score:.2f}{unit_sym}",
            f"Performance Tier       : {tier_info['tier']} (Grade {tier_info['grade']})",
            f"Interpretation         : {tier_info['description']}",
            "-----------------------------------------------------------",
            f"Sample Count           : {len(self.scores)}",
            f"Median Score           : {float(np.median(arr)):.2f}{unit_sym}",
            f"Std Deviation          : {float(np.std(arr)):.2f}",
            f"Min Score              : {float(np.min(arr)):.2f}{unit_sym}",
            f"Max Score              : {float(np.max(arr)):.2f}{unit_sym}",
            f"Embedding Model        : {self.model}",
            f"Scale                  : {self.scale}",
            "===========================================================",
        ]
        return "\n".join(lines)
