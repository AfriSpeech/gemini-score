"""Tests for language resolution module using afriso."""

import pytest
from gemini_score.languages import (
    LanguageInfo,
    clean_prompt_name,
    is_supported_language,
    list_supported_languages,
    resolve_language,
    search_languages,
)


def test_clean_prompt_name():
    assert clean_prompt_name("Swahili (macrolanguage)") == "Swahili"
    assert clean_prompt_name("Swahili (individual language)") == "Swahili"
    assert clean_prompt_name("Yoruba") == "Yoruba"
    assert clean_prompt_name("Boko (Benin)") == "Boko (Benin)"


def test_resolve_language_by_iso3():
    yor = resolve_language("yor")
    assert yor.name == "Yoruba"
    assert yor.iso639_3 == "yor"
    assert yor.prompt_name == "Yoruba"
    assert yor.is_african is True
    assert "NG" in yor.countries

    hau = resolve_language("hau")
    assert hau.name == "Hausa"
    assert hau.iso639_3 == "hau"

    twi = resolve_language("twi")
    assert twi.name == "Twi"
    assert twi.iso639_3 == "twi"


def test_resolve_language_by_iso1():
    yo = resolve_language("yo")
    assert yo.name == "Yoruba"
    assert yo.iso639_3 == "yor"

    sw = resolve_language("sw")
    assert "Swahili" in sw.name
    assert sw.prompt_name == "Swahili"


def test_resolve_language_by_primary_name():
    yor = resolve_language("Yoruba")
    assert yor.iso639_3 == "yor"

    swh = resolve_language("Swahili")
    assert swh.prompt_name == "Swahili"

    amh = resolve_language("Amharic")
    assert amh.iso639_3 == "amh"


def test_resolve_language_by_alt_name():
    # 'Yariba' is an alternative name for Yoruba
    lang = resolve_language("Yariba")
    assert lang.iso639_3 == "yor"

    # 'Kiswahili'
    kiswahili = resolve_language("Kiswahili")
    assert "Swahili" in kiswahili.name


def test_resolve_language_fallback_for_unknown():
    lang = resolve_language("AlienLanguage123")
    assert lang.name == "AlienLanguage123"
    assert lang.prompt_name == "AlienLanguage123"
    assert lang.is_african is False
    assert lang.iso639_3 is None


def test_resolve_language_already_info():
    info = LanguageInfo(name="Custom", prompt_name="Custom", is_african=True)
    assert resolve_language(info) is info


def test_resolve_language_invalid_inputs():
    with pytest.raises(ValueError):
        resolve_language("")

    with pytest.raises(ValueError):
        resolve_language("   ")

    with pytest.raises(TypeError):
        resolve_language(12345)


def test_list_supported_languages():
    langs = list_supported_languages()
    assert len(langs) > 2000
    assert "Yoruba" in langs
    assert "Hausa" in langs


def test_is_supported_language():
    assert is_supported_language("yor") is True
    assert is_supported_language("Yoruba") is True
    assert is_supported_language("hau") is True
    assert is_supported_language("twi") is True
    assert is_supported_language("non_existent_xyz") is False


def test_search_languages():
    results = search_languages("Swahili")
    assert len(results) > 0
    assert any("Swahili" in r.prompt_name for r in results)
