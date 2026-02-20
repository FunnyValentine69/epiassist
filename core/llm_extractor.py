"""LLM-based extraction module — provider-agnostic.

Provides a second-pass extraction that runs after regex to catch
non-standard formatting, tabular layouts, and unusual stat presentations.
Results are deduplicated and merged with regex results.

Delegates to providers in core.llm_providers (Gemini, Ollama).
"""

from typing import Optional

from core.llm_providers import detect_provider, get_provider_functions
from core.llm_providers._parse import (
    CATEGORIES,
    _empty_results,
    _safe_float,
    _safe_int,
)


def is_llm_available() -> tuple[bool, Optional[str]]:
    """Check if any LLM provider is available.

    Returns:
        Tuple of (available: bool, provider_name: str | None).
    """
    provider = detect_provider()
    return (provider is not None, provider)


def extract_with_llm(text: str, page: int = 1) -> dict[str, list[dict]]:
    """Extract epidemiological statistics from text using the best available LLM.

    Args:
        text: Page text to extract from.
        page: Page number for attribution.

    Returns:
        Dict with 8 category keys, each containing a list of extracted dicts.
        Returns empty results if no provider available or on any failure.
    """
    provider = detect_provider()
    if provider is None:
        return _empty_results()

    try:
        funcs = get_provider_functions(provider)
        return funcs["extract_stats"](text, page)
    except Exception:
        return _empty_results()


def _dedup_key(item: dict, category: str) -> Optional[tuple]:
    """Build a deduplication key for a result item.

    Args:
        item: A result dict from either regex or LLM extraction.
        category: The category key (e.g., 'effect_measures').

    Returns:
        A tuple used for equality comparison, or None if unparseable.
    """
    try:
        if category == "effect_measures":
            return (
                item.get("type"),
                _safe_float(str(item.get("value", ""))),
                _safe_float(str(item.get("ci_lower", "") or "")),
                _safe_float(str(item.get("ci_upper", "") or "")),
                item.get("page"),
            )
        elif category == "confidence_intervals":
            return (
                item.get("level"),
                _safe_float(str(item.get("lower", ""))),
                _safe_float(str(item.get("upper", ""))),
                item.get("page"),
            )
        elif category == "p_values":
            return (
                _safe_float(str(item.get("value", ""))),
                item.get("operator"),
                item.get("page"),
            )
        elif category == "sample_sizes":
            return (
                _safe_int(str(item.get("value", ""))),
                item.get("page"),
            )
        elif category == "beta_coefficients":
            return (
                _safe_float(str(item.get("value", ""))),
                _safe_float(str(item.get("ci_lower", "") or "")),
                _safe_float(str(item.get("ci_upper", "") or "")),
                item.get("page"),
            )
        elif category == "mean_differences":
            return (
                _safe_float(str(item.get("value", ""))),
                _safe_float(str(item.get("ci_lower", "") or "")),
                _safe_float(str(item.get("ci_upper", "") or "")),
                item.get("page"),
            )
        elif category == "standard_deviations":
            return (
                _safe_float(str(item.get("value", ""))),
                _safe_float(str(item.get("mean", "") or "")),
                item.get("type"),
                item.get("page"),
            )
        elif category == "weighted_statistics":
            return (
                item.get("stat_type"),
                _safe_float(str(item.get("value", ""))),
                item.get("page"),
            )
    except (ValueError, TypeError):
        pass
    return None


def merge_results(
    regex_results: dict[str, list[dict]],
    llm_results: dict[str, list[dict]],
) -> dict[str, list[dict]]:
    """Merge regex and LLM extraction results with deduplication.

    Tags each result with source='regex' or source='llm'. LLM results
    that duplicate a regex result (by float-equal key comparison) are dropped.

    Args:
        regex_results: Results from regex extraction.
        llm_results: Results from LLM extraction.

    Returns:
        Merged dict with all 8 category keys.
    """
    merged = _empty_results()

    for category in CATEGORIES:
        # Tag and collect regex results, build dedup set
        seen_keys: set[tuple] = set()
        for item in regex_results.get(category, []):
            tagged = {**item, "source": "regex"}
            merged[category].append(tagged)
            key = _dedup_key(tagged, category)
            if key is not None:
                seen_keys.add(key)

        # Add LLM results that aren't duplicates
        for item in llm_results.get(category, []):
            tagged = {**item, "source": "llm"}
            key = _dedup_key(tagged, category)
            if key is None or key not in seen_keys:
                merged[category].append(tagged)
                if key is not None:
                    seen_keys.add(key)

    return merged
