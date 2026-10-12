"""Tests for Gemini translator module."""

import os
import pytest
from gemini_score.languages import resolve_language
from gemini_score.translator import (
    GeminiAuthError,
    GeminiTranslationError,
    GeminiTranslator,
    _clean_json_response,
)


def test_clean_json_response():
    assert _clean_json_response('["Hello", "World"]') == ["Hello", "World"]
    assert _clean_json_response('```json\n["One", "Two"]\n```') == ["One", "Two"]
    assert _clean_json_response('Some prefix ["A", "B"] some suffix') == ["A", "B"]
    assert _clean_json_response("not valid json at all") is None


def test_auth_error_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(GeminiAuthError):
        GeminiTranslator(api_key=None, client=None)


def test_translate_single_empty():
    translator = GeminiTranslator(client=lambda p, is_json=False: "Never called")
    assert translator.translate_single("", "yor") == ""
    assert translator.translate_single("   ", "yor") == ""


def test_translate_single_with_mock():
    calls = []

    def mock_client(prompt, is_json=False):
        calls.append((prompt, is_json))
        return "Good morning"

    translator = GeminiTranslator(client=mock_client)
    res = translator.translate_single("E kaaro", "yor")
    assert res == "Good morning"
    assert len(calls) == 1
    assert "Yoruba" in calls[0][0]
    assert "E kaaro" in calls[0][0]


def test_caching():
    call_count = 0

    def mock_client(prompt, is_json=False):
        nonlocal call_count
        call_count += 1
        return "Thank you"

    translator = GeminiTranslator(client=mock_client, enable_cache=True)
    res1 = translator.translate_single("E se", "yor")
    res2 = translator.translate_single("E se", "yor")
    assert res1 == "Thank you"
    assert res2 == "Thank you"
    # Second call must hit cache
    assert call_count == 1


def test_translate_batch_with_mock_json():
    def mock_client(prompt, is_json=False):
        assert is_json is True
        return '["Good morning", "How are you?"]'

    translator = GeminiTranslator(client=mock_client, batch_size=10)
    res = translator.translate_batch(["E kaaro", "Bawo ni"], "Yoruba")
    assert res == ["Good morning", "How are you?"]


def test_translate_batch_fallback_on_json_error():
    call_log = []

    def mock_client(prompt, is_json=False):
        call_log.append(is_json)
        if is_json:
            # Return broken json or wrong length
            return '["Only one item"]'
        # Individual fallback calls
        return "Fallback translation"

    translator = GeminiTranslator(client=mock_client, batch_size=10)
    res = translator.translate_batch(["Text 1", "Text 2"], "Yoruba")
    assert len(res) == 2
    assert res == ["Fallback translation", "Fallback translation"]
    assert True in call_log  # batch JSON attempted
    assert False in call_log  # individual fallback attempted


def test_translate_batch_multilingual():
    translations_map = {
        "E kaaro": "Good morning",
        "Habari za asubuhi": "Good morning",
    }

    def mock_client(prompt, is_json=False):
        if is_json:
            if "Yoruba" in prompt:
                return '["Good morning"]'
            elif "Swahili" in prompt:
                return '["Good morning"]'
        return "Good morning"

    translator = GeminiTranslator(client=mock_client)
    res = translator.translate_batch(
        ["E kaaro", "Habari za asubuhi"],
        languages=["yor", "swh"],
    )
    assert res == ["Good morning", "Good morning"]


def test_retry_on_rate_limit():
    attempts = 0

    def mock_flaky_client(prompt, is_json=False):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded")
        return "Recovered after retry"

    translator = GeminiTranslator(client=mock_flaky_client, max_retries=2, backoff_factor=1.1)
    res = translator.translate_single("Bawo ni", "yor")
    assert res == "Recovered after retry"
    assert attempts == 2


def test_gemini_model_validation():
    from gemini_score.translator import validate_gemini_model

    # Valid Flash 3+ models
    assert validate_gemini_model("gemini-3.8-flash") == "gemini-3.8-flash"
    assert validate_gemini_model("gemini-3.0-flash") == "gemini-3.0-flash"
    assert validate_gemini_model("gemini-3-flash") == "gemini-3-flash"
    assert validate_gemini_model("models/gemini-3.8-flash") == "models/gemini-3.8-flash"
    assert validate_gemini_model("gemini-4.0-flash") == "gemini-4.0-flash"

    # Reject versions below 3.0
    with pytest.raises(ValueError, match="below the minimum supported version 3.0"):
        validate_gemini_model("gemini-2.5-flash")

    with pytest.raises(ValueError, match="below the minimum supported version 3.0"):
        validate_gemini_model("gemini-2.0-flash")

    with pytest.raises(ValueError, match="below the minimum supported version 3.0"):
        validate_gemini_model("gemini-1.5-flash")

    # Reject non-flash models
    with pytest.raises(ValueError, match="not a Flash variant"):
        validate_gemini_model("gemini-3.8-pro")

    # Reject invalid model name format
    with pytest.raises(ValueError, match="Invalid Gemini model"):
        validate_gemini_model("gpt-4o")
