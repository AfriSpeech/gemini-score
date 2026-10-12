"""Tests for the core GeminiScore API and result structures."""

import pytest
from gemini_score import GeminiScore, GeminiScoreResult, compute, get_performance_tier, score
from gemini_score.similarity import _pseudo_embed


def make_mock_client():
    def mock_embed(texts, is_embed=True):
        return [_pseudo_embed(t) for t in texts]

    return mock_embed


def test_core_score_basic():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions=["Ologbo naa wa lori akete.", "Good morning my friend."],
        references=["The cat is on the mat.", "Good morning my friend."],
    )
    assert isinstance(res, GeminiScoreResult)
    assert len(res) == 2
    assert res.score >= 50.0
    assert len(res.scores) == 2
    assert res.model == "gemini-embedding-002"


def test_core_score_single_string():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions="The cat is resting on the mat.",
        references="The cat is resting on the mat.",
    )
    assert len(res) == 1
    assert res.score >= 99.0


def test_core_score_empty():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(predictions=[], references=[])
    assert res.score == 0.0
    assert len(res) == 0


def test_core_score_multi_reference():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions=["The cat is on the mat."],
        references=[["A feline sits upon the rug.", "The cat is on the mat."]],
    )
    assert len(res) == 1
    assert res.score >= 99.0


def test_core_score_mismatched_lengths():
    scorer = GeminiScore(client=make_mock_client())
    with pytest.raises(ValueError, match="match"):
        scorer.score(
            predictions=["Pred 1", "Pred 2"],
            references=["Ref 1"],
        )


def test_core_compute():
    scorer = GeminiScore(client=make_mock_client())
    out = scorer.compute(
        predictions=["The cat is on the mat."],
        references=["The cat is on the mat."],
    )
    assert "gemini_score" in out
    assert "scores" in out
    assert isinstance(out["gemini_score"], float)


def test_functional_score_and_compute():
    mock = make_mock_client()
    res = score(
        predictions=["Hello world"],
        references=["Hello world"],
        client=mock,
    )
    assert isinstance(res, GeminiScoreResult)
    assert res.score >= 99.0

    out = compute(
        predictions=["Hello world"],
        references=["Hello world"],
        client=mock,
    )
    assert "gemini_score" in out


def test_result_methods():
    mock = make_mock_client()
    scorer = GeminiScore(client=mock)
    res = scorer.score(
        predictions=["The cat is on the mat."],
        references=["The cat is on the mat."],
    )

    # __getitem__
    item = res[0]
    assert item["prediction"] == "The cat is on the mat."
    assert item["reference"] == "The cat is on the mat."
    assert "score" in item
    assert "tier" in item

    # to_dict & to_json
    d = res.to_dict()
    assert d["score"] == res.score
    j = res.to_json()
    assert isinstance(j, str)
    assert "gemini-embedding-002" in j

    # to_dataframe
    df = res.to_dataframe()
    assert len(df) == 1
    assert "prediction" in df.columns
    assert "score" in df.columns

    # summary
    summary = res.summary()
    assert "GeminiScore Evaluation Summary" in summary
    assert "Final Aggregated Score" in summary
    assert "Performance Tier" in summary
    assert res.tier in ["Excellent", "Good", "Moderate", "Poor", "Severely Deficient"]


def test_performance_tiers():
    t1 = get_performance_tier(95.0, scale="percent")
    assert t1["tier"] == "Excellent"
    assert t1["grade"] == "A"

    t2 = get_performance_tier(75.0, scale="percent")
    assert t2["tier"] == "Good"
    assert t2["grade"] == "B"

    t3 = get_performance_tier(60.0, scale="percent")
    assert t3["tier"] == "Moderate"

    t4 = get_performance_tier(45.0, scale="percent")
    assert t4["tier"] == "Poor"

    t5 = get_performance_tier(20.0, scale="percent")
    assert t5["tier"] == "Severely Deficient"
