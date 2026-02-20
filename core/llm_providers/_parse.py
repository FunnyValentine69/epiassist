"""Shared JSON response parser for all LLM providers.

Converts raw JSON strings from any provider into the dict schema
that merge_results() expects.
"""

import json
from typing import Optional


CATEGORIES = [
    "effect_measures",
    "confidence_intervals",
    "p_values",
    "sample_sizes",
    "beta_coefficients",
    "mean_differences",
    "standard_deviations",
    "weighted_statistics",
]


def _empty_results() -> dict[str, list[dict]]:
    """Return an empty results dict with all 8 category keys."""
    return {cat: [] for cat in CATEGORIES}


def _safe_float(val) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    """Safely convert a value to int, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _parse_effect_measure(item: dict, page: int) -> Optional[dict]:
    """Parse a single effect measure entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "type": item.get("type", "OR"),
        "value": value,
        "ci_lower": _safe_float(item.get("ci_lower")),
        "ci_upper": _safe_float(item.get("ci_upper")),
        "adjusted": None,
        "adjusted_for": None,
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_confidence_interval(item: dict, page: int) -> Optional[dict]:
    """Parse a single confidence interval entry."""
    lower = _safe_float(item.get("lower"))
    upper = _safe_float(item.get("upper"))
    if lower is None or upper is None:
        return None
    return {
        "level": _safe_int(item.get("level", 95)) or 95,
        "lower": lower,
        "upper": upper,
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_p_value(item: dict, page: int) -> Optional[dict]:
    """Parse a single p-value entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "operator": item.get("operator", "="),
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_sample_size(item: dict, page: int) -> Optional[dict]:
    """Parse a single sample size entry."""
    value = _safe_int(item.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "page": page,
    }


def _parse_beta_coefficient(item: dict, page: int) -> Optional[dict]:
    """Parse a single beta coefficient entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "ci_lower": _safe_float(item.get("ci_lower")),
        "ci_upper": _safe_float(item.get("ci_upper")),
        "se": _safe_float(item.get("se")),
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_mean_difference(item: dict, page: int) -> Optional[dict]:
    """Parse a single mean difference entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "ci_lower": _safe_float(item.get("ci_lower")),
        "ci_upper": _safe_float(item.get("ci_upper")),
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_standard_deviation(item: dict, page: int) -> Optional[dict]:
    """Parse a single standard deviation entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "value": value,
        "mean": _safe_float(item.get("mean")),
        "type": item.get("type", "SD"),
        "context": item.get("context", ""),
        "page": page,
    }


def _parse_weighted_statistic(item: dict, page: int) -> Optional[dict]:
    """Parse a single weighted statistic entry."""
    value = _safe_float(item.get("value"))
    if value is None:
        return None
    return {
        "stat_type": item.get("stat_type", ""),
        "value": value,
        "weight_method": item.get("weight_method") or None,
        "context": item.get("context", ""),
        "page": page,
    }


_PARSERS = {
    "effect_measures": _parse_effect_measure,
    "confidence_intervals": _parse_confidence_interval,
    "p_values": _parse_p_value,
    "sample_sizes": _parse_sample_size,
    "beta_coefficients": _parse_beta_coefficient,
    "mean_differences": _parse_mean_difference,
    "standard_deviations": _parse_standard_deviation,
    "weighted_statistics": _parse_weighted_statistic,
}


def parse_extraction_response(raw_json: str, page: int) -> dict[str, list[dict]]:
    """Parse a JSON string from any LLM provider into the 8-category results dict.

    Args:
        raw_json: Raw JSON string from LLM response.
        page: Page number for attribution.

    Returns:
        Dict with 8 category keys, each containing a list of parsed dicts.
        Returns empty results on any parse failure.
    """
    if not raw_json:
        return _empty_results()

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return _empty_results()

    if not isinstance(data, dict):
        return _empty_results()

    results = _empty_results()
    for category in CATEGORIES:
        items = data.get(category, [])
        if not isinstance(items, list):
            continue
        parser = _PARSERS[category]
        for item in items:
            if not isinstance(item, dict):
                continue
            parsed = parser(item, page)
            if parsed is not None:
                results[category].append(parsed)

    return results
