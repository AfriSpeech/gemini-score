"""Gemini translation client with batching, concurrency, retries, and caching."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .languages import LanguageInfo, resolve_language

logger = logging.getLogger(__name__)

# Default Gemini model
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiAuthError(ValueError):
    """Raised when no valid Gemini API key is found."""


class GeminiTranslationError(RuntimeError):
    """Raised when translation fails after all retry attempts."""


def _clean_json_response(text: str) -> Optional[List[str]]:
    """Extracts a list of strings from a Gemini JSON response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except Exception:
        pass

    # Regex search for JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except Exception:
            pass

    return None


class GeminiTranslator:
    """Translates text from African languages to English using Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        batch_size: int = 16,
        max_workers: int = 4,
        max_retries: int = 4,
        backoff_factor: float = 2.0,
        enable_cache: bool = True,
        client: Optional[Any] = None,
    ) -> None:
        """Initializes the Gemini translator.

        Parameters
        ----------
        api_key: str, optional
            Gemini API key. Defaults to GEMINI_API_KEY or GOOGLE_API_KEY from environment.
        model: str
            Gemini model name. Defaults to 'gemini-2.5-flash' (or env GEMINI_MODEL).
        batch_size: int
            Number of sentences per structured prompt call.
        max_workers: int
            Number of concurrent worker threads.
        max_retries: int
            Maximum number of retry attempts on rate limits or network errors.
        backoff_factor: float
            Exponential backoff factor for retries.
        enable_cache: bool
            Whether to cache translated sentences in memory.
        client: Any, optional
            Pre-configured GenAI client instance (useful for testing and dependency injection).
        """
        self.model = model
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self.max_retries = max(0, max_retries)
        self.backoff_factor = backoff_factor
        self.enable_cache = enable_cache
        self._cache: Dict[Tuple[str, str], str] = {}

        if client is not None:
            self.client = client
        else:
            resolved_key = (
                api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            if not resolved_key:
                raise GeminiAuthError(
                    "No Gemini API key found. Please set the GEMINI_API_KEY (or GOOGLE_API_KEY) "
                    "environment variable, or pass `api_key='...'` to the constructor."
                )
            self.client = self._init_client(resolved_key)

    @staticmethod
    def _init_client(api_key: str) -> Any:
        """Initializes the Google GenAI SDK client."""
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except ImportError:  # pragma: no cover
            try:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=api_key)
                return genai_legacy
            except ImportError:
                raise ImportError(
                    "Google GenAI SDK is required. "
                    "Please install `google-genai` using `pip install google-genai`."
                )

    def _call_gemini(self, prompt: str, is_json: bool = False) -> str:
        """Sends a single request to Gemini with retry logic."""
        wait_time = 1.0

        for attempt in range(self.max_retries + 1):
            try:
                # Modern google.genai Client
                if hasattr(self.client, "models") and hasattr(self.client.models, "generate_content"):
                    from google.genai import types

                    config_kwargs: Dict[str, Any] = {"temperature": 0.0}
                    if is_json:
                        config_kwargs["response_mime_type"] = "application/json"

                    config = types.GenerateContentConfig(**config_kwargs)
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                    text = response.text or ""
                    return text.strip()

                # Legacy google.generativeai fallback
                elif hasattr(self.client, "GenerativeModel"):
                    generation_config = {"temperature": 0.0}
                    if is_json:
                        generation_config["response_mime_type"] = "application/json"

                    model_instance = self.client.GenerativeModel(
                        model_name=self.model,
                        generation_config=generation_config,
                    )
                    response = model_instance.generate_content(prompt)
                    text = response.text or ""
                    return text.strip()

                # Generic callable client or mock object
                elif callable(self.client):
                    res = self.client(prompt, is_json=is_json)
                    return str(res).strip()

                raise TypeError(f"Unrecognized Gemini client type: {type(self.client)}")

            except Exception as exc:
                err_msg = str(exc).lower()
                is_rate_limit = any(
                    token in err_msg
                    for token in ["429", "rate", "resource_exhausted", "quota", "overloaded"]
                )
                is_transient = is_rate_limit or any(
                    token in err_msg for token in ["503", "timeout", "unavailable"]
                )

                if attempt < self.max_retries and is_transient:
                    logger.warning(
                        "Gemini API request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self.max_retries,
                        exc,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    wait_time *= self.backoff_factor
                else:
                    logger.error("Gemini API call failed permanently: %s", exc)
                    raise GeminiTranslationError(f"Failed to call Gemini API: {exc}") from exc

        raise GeminiTranslationError("Gemini API call exceeded max retries.")

    def translate_single(self, text: str, language: Union[str, LanguageInfo]) -> str:
        """Translates a single African language sentence to English."""
        if not text or not text.strip():
            return ""

        lang_info = resolve_language(language)
        cache_key = (lang_info.prompt_name, text)

        if self.enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"You are a professional linguist and translator. "
            f"Translate the following {lang_info.prompt_name} text into natural English. "
            f"Output ONLY the exact English translation without any quotes, notes, or explanations.\n\n"
            f"Source text: {text}"
        )

        response = self._call_gemini(prompt, is_json=False)
        # Clean any accidental outer quotes
        cleaned = response.strip().strip('"').strip("'")

        if self.enable_cache:
            self._cache[cache_key] = cleaned

        return cleaned

    def _translate_batch_chunk(
        self,
        texts: List[str],
        lang_info: LanguageInfo,
    ) -> List[str]:
        """Translates a chunk of sentences in a single prompt using structured JSON."""
        # Check cache for any already-translated items
        uncached_indices = []
        translations: List[Optional[str]] = [None] * len(texts)

        for i, t in enumerate(texts):
            if not t or not t.strip():
                translations[i] = ""
            elif self.enable_cache and (lang_info.prompt_name, t) in self._cache:
                translations[i] = self._cache[(lang_info.prompt_name, t)]
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [t or "" for t in translations]

        uncached_texts = [texts[i] for i in uncached_indices]

        # Structured prompt
        prompt = (
            f"You are a professional linguist and translator. "
            f"Translate each of the following {lang_info.prompt_name} sentences into natural English.\n"
            f"You must return ONLY a JSON array of strings containing the translations in the exact same order.\n\n"
            f"Input sentences:\n{json.dumps(uncached_texts, ensure_ascii=False)}"
        )

        try:
            response_text = self._call_gemini(prompt, is_json=True)
            parsed = _clean_json_response(response_text)
            if parsed and len(parsed) == len(uncached_texts):
                for idx, trans in zip(uncached_indices, parsed):
                    translations[idx] = trans
                    if self.enable_cache:
                        self._cache[(lang_info.prompt_name, texts[idx])] = trans
                return [t or "" for t in translations]
            else:
                logger.warning(
                    "JSON array length mismatch (expected %d, got %s). Falling back to individual translation.",
                    len(uncached_texts),
                    len(parsed) if parsed else "None",
                )
        except Exception as exc:
            logger.warning("Batch translation failed: %s. Falling back to individual translation.", exc)

        # Fallback: translate remaining individually
        for idx in uncached_indices:
            trans = self.translate_single(texts[idx], lang_info)
            translations[idx] = trans

        return [t or "" for t in translations]

    def translate_batch(
        self,
        texts: Sequence[str],
        languages: Union[str, LanguageInfo, Sequence[Union[str, LanguageInfo]]],
    ) -> List[str]:
        """Translates a batch of texts to English with batching and thread concurrency.

        Parameters
        ----------
        texts: Sequence[str]
            Texts in African languages to translate.
        languages: str, LanguageInfo, or Sequence[str | LanguageInfo]
            Single language applied to all texts, or a sequence specifying the language
            for each individual text in the batch.

        Returns
        -------
        List[str]
            List of translated English texts.
        """
        n = len(texts)
        if n == 0:
            return []

        # Normalize languages
        if isinstance(languages, (str, LanguageInfo)):
            resolved_langs = [resolve_language(languages)] * n
        else:
            if len(languages) != n:
                raise ValueError(
                    f"Mismatch between number of texts ({n}) and languages ({len(languages)})."
                )
            resolved_langs = [resolve_language(lang) for lang in languages]

        # Group contiguous chunks by language for efficient prompt batching
        chunks: List[Tuple[List[int], List[str], LanguageInfo]] = []
        current_indices: List[int] = []
        current_texts: List[str] = []
        current_lang: Optional[LanguageInfo] = None

        for idx, (text, lang_info) in enumerate(zip(texts, resolved_langs)):
            if (
                current_lang is not None
                and (lang_info.name != current_lang.name or len(current_texts) >= self.batch_size)
            ):
                chunks.append((current_indices, current_texts, current_lang))
                current_indices = []
                current_texts = []

            current_lang = lang_info
            current_indices.append(idx)
            current_texts.append(str(text))

        if current_texts and current_lang is not None:
            chunks.append((current_indices, current_texts, current_lang))

        results: List[Optional[str]] = [None] * n

        if len(chunks) == 1 or self.max_workers == 1:
            for indices, chunk_texts, lang_info in chunks:
                chunk_res = self._translate_batch_chunk(chunk_texts, lang_info)
                for idx, trans in zip(indices, chunk_res):
                    results[idx] = trans
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(chunks))) as executor:
                future_to_indices = {
                    executor.submit(self._translate_batch_chunk, chunk_texts, lang_info): indices
                    for indices, chunk_texts, lang_info in chunks
                }
                for future in as_completed(future_to_indices):
                    indices = future_to_indices[future]
                    chunk_res = future.result()
                    for idx, trans in zip(indices, chunk_res):
                        results[idx] = trans

        return [r or "" for r in results]
