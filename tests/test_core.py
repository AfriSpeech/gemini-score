"""Tests for the core GeminiScore API and result structures."""

import pytest
from gemini_score import GeminiScore, GeminiScoreResult, compute, score


def make_mock_client():
    def mock_client(prompt, is_json=False):
        if is_json:
            return '["The cat is on the mat.", "Good morning my friend."]'
        return "The cat is on the mat."

    return mock_client


def test_core_score_basic():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions=["Ologbo naa wa lori akete.", "E kaaro ore mi."],
        references=["The cat is resting on the mat.", "Good morning to my dear friend."],
        language="Yoruba",
    )
    assert isinstance(res, GeminiScoreResult)
    assert len(res) == 2
    assert res.score > 70.0
    assert len(res.scores) == 2
    assert len(res.translations) == 2
    assert res.languages == ["Yoruba", "Yoruba"]


def test_core_score_single_string():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions="Ologbo naa wa lori akete.",
        references="The cat is resting on the mat.",
        language="yor",
    )
    assert len(res) == 1
    assert res.score > 70.0


def test_core_score_empty():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(predictions=[], references=[], language="yor")
    assert res.score == 0.0
    assert len(res) == 0


def test_core_score_multi_reference():
    scorer = GeminiScore(client=make_mock_client())
    res = scorer.score(
        predictions=["Ologbo naa wa lori akete."],
        references=[["A feline sits upon the rug.", "The cat is on the mat."]],
        language="yor",
    )
    assert len(res) == 1
    # Because one reference is identical to the mock translation:
    assert res.score >= 99.0


def test_core_score_mismatched_lengths():
    scorer = GeminiScore(client=make_mock_client())
    with pytest.raises(ValueError, match="match"):
        scorer.score(
            predictions=["Pred 1", "Pred 2"],
            references=["Ref 1"],
            language="yor",
        )

    with pytest.raises(ValueError, match="match"):
        scorer.score(
            predictions=["Pred 1"],
            references=["Ref 1"],
            language=["yor", "hau"],
        )


def test_core_compute():
    scorer = GeminiScore(client=make_mock_client())
    out = scorer.compute(
        predictions=["Ologbo naa wa lori akete."],
        references=["The cat is resting on the mat."],
        language="yor",
    )
    assert "gemini_score" in out
    assert "scores" in out
    assert "translations" in out
    assert isinstance(out["gemini_score"], float)


def test_functional_score_and_compute():
    mock = make_mock_client()
    res = score(
        predictions=["Bawo ni"],
        references=["How are you?"],
        language="yor",
        client=mock,
    )
    assert isinstance(res, GeminiScoreResult)

    out = compute(
        predictions=["Bawo ni"],
        references=["How are you?"],
        language="yor",
        client=mock,
    )
    assert "gemini_score" in out


def test_result_methods():
    mock = make_mock_client()
    scorer = GeminiScore(client=mock)
    res = scorer.score(
        predictions=["Ologbo naa wa lori akete."],
        references=["The cat is on the mat."],
        language="yor",
    )

    # __getitem__
    item = res[0]
    assert item["prediction"] == "Ologbo naa wa lori akete."
    assert item["reference"] == "The cat is on the mat."
    assert "translation" in item
    assert "score" in item

    # to_dict & to_json
    d = res.to_dict()
    assert d["score"] == res.score
    j = res.to_json()
    assert isinstance(j, str)
    assert "gemini-3.8-flash" in j

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


def test_performance_tiers_and_multilingual_breakdown():
    from gemini_score import get_performance_tier

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

    # Multilingual aggregation test
    def multi_client(prompt, is_json=False):
        if is_json:
            return '["Good morning", "Thank you"]'
        return "Good morning"

    scorer = GeminiScore(client=multi_client)
    res = scorer.score(
        predictions=["E kaaro", "Asante sana"],
        references=["Good morning", "Thank you very much"],
        language=["yor", "swh"],
    )

    by_lang = res.by_language()
    assert "Yoruba" in by_lang
    assert "Swahili" in by_lang
    assert by_lang["Yoruba"]["count"] == 1
    assert by_lang["Swahili"]["count"] == 1
    assert "Language Breakdown:" in res.summary()
