"""Language resolution and metadata handling powered by AfriSpeech/afriso."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

try:
    import afriso
except ImportError:  # pragma: no cover
    afriso = None

logger = logging.getLogger(__name__)

# Pattern to strip technical disambiguation tags from SIL/ISO names
_DISAMBIGUATION_PATTERN = re.compile(
    r"\s*\((individual language|macrolanguage|special)\)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class LanguageInfo:
    """Standardized representation of a language with African language metadata."""

    name: str
    prompt_name: str
    iso639_3: Optional[str] = None
    iso639_1: Optional[str] = None
    family: Optional[str] = None
    regions: Tuple[str, ...] = field(default_factory=tuple)
    countries: Tuple[str, ...] = field(default_factory=tuple)
    alt_names: Tuple[str, ...] = field(default_factory=tuple)
    is_african: bool = True

    def __str__(self) -> str:
        code_str = f" [{self.iso639_3}]" if self.iso639_3 else ""
        return f"{self.prompt_name}{code_str}"


def clean_prompt_name(name: str) -> str:
    """Strips technical SIL tags such as '(macrolanguage)' or '(individual language)'."""
    return _DISAMBIGUATION_PATTERN.sub("", name).strip()


def resolve_language(lang_input: Union[str, LanguageInfo]) -> LanguageInfo:
    """Resolves any African language identifier (name, alt-name, ISO code) using afriso.

    Parameters
    ----------
    lang_input: str or LanguageInfo
        Language code (e.g. 'yor', 'swh', 'sw', 'twi', 'hau'),
        language name (e.g. 'Yoruba', 'Swahili', 'Hausa', 'Kiswahili'),
        or an existing LanguageInfo instance.

    Returns
    -------
    LanguageInfo
        Resolved language details. If not found in afriso, returns a fallback
        LanguageInfo with `is_african=False`.
    """
    if isinstance(lang_input, LanguageInfo):
        return lang_input

    if not isinstance(lang_input, str):
        raise TypeError(f"Language must be str or LanguageInfo, got {type(lang_input).__name__}")

    query = lang_input.strip()
    if not query:
        raise ValueError("Language identifier cannot be empty.")

    if afriso is None:  # pragma: no cover
        # Fallback if afriso is unexpectedly not installed
        cleaned = clean_prompt_name(query)
        return LanguageInfo(
            name=query,
            prompt_name=cleaned,
            is_african=False,
        )

    # 1. Direct query in afriso (handles ISO 639-3, ISO 639-1, primary name, alt names)
    record = afriso.get(query)

    if record is not None:
        p_name = clean_prompt_name(record.name)
        return LanguageInfo(
            name=record.name,
            prompt_name=p_name,
            iso639_3=record.iso639_3,
            iso639_1=record.iso639_1,
            family=record.family,
            regions=tuple(record.regions or ()),
            countries=tuple(record.countries or ()),
            alt_names=tuple(record.alt_names or ()),
            is_african=True,
        )

    # 2. Try fuzzy / prefix search with afriso if direct lookup fails
    try:
        search_res = afriso.search(query)
        if search_res and len(search_res) > 0:
            match = search_res[0]
            logger.info("Fuzzy matched '%s' to '%s' (%s)", query, match.name, match.iso639_3)
            p_name = clean_prompt_name(match.name)
            return LanguageInfo(
                name=match.name,
                prompt_name=p_name,
                iso639_3=match.iso639_3,
                iso639_1=match.iso639_1,
                family=match.family,
                regions=tuple(match.regions or ()),
                countries=tuple(match.countries or ()),
                alt_names=tuple(match.alt_names or ()),
                is_african=True,
            )
    except Exception as exc:  # pragma: no cover
        logger.debug("afriso.search error on '%s': %s", query, exc)

    # 3. Graceful fallback for non-African languages or uncataloged dialects
    logger.warning(
        "Language '%s' was not found in the afriso African language database. "
        "Proceeding with provided name for translation.",
        query,
    )
    cleaned = clean_prompt_name(query)
    return LanguageInfo(
        name=query,
        prompt_name=cleaned,
        is_african=False,
    )


def list_supported_languages() -> List[str]:
    """Returns a list of all African language names in the afriso database."""
    if afriso is None:  # pragma: no cover
        return []
    return [lang.name for lang in afriso.languages()]


def search_languages(query: str) -> List[LanguageInfo]:
    """Searches afriso for African languages matching a query."""
    if afriso is None:  # pragma: no cover
        return []
    results = afriso.search(query)
    out = []
    for r in results:
        p_name = clean_prompt_name(r.name)
        out.append(
            LanguageInfo(
                name=r.name,
                prompt_name=p_name,
                iso639_3=r.iso639_3,
                iso639_1=r.iso639_1,
                family=r.family,
                regions=tuple(r.regions or ()),
                countries=tuple(r.countries or ()),
                alt_names=tuple(r.alt_names or ()),
                is_african=True,
            )
        )
    return out


def is_supported_language(query: str) -> bool:
    """Checks whether the given name or code is in the afriso African language database."""
    if afriso is None:  # pragma: no cover
        return False
    return afriso.get(query) is not None
