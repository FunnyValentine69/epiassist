"""LLM-based extraction module using LangExtract + Ollama.

Provides a second-pass extraction that runs after regex to catch
non-standard formatting, tabular layouts, and unusual stat presentations.
Results are deduplicated and merged with regex results.
"""

from typing import Optional

import requests


# Category keys matching paper_parser output structure
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

_PROMPT = (
    "Extract ALL epidemiological statistics from this text. "
    "For each statistic found, classify it as one of: "
    "effect_measure, confidence_interval, p_value, sample_size, "
    "beta_coefficient, mean_difference, standard_deviation, weighted_statistic. "
    "Extract the numeric values as attributes. "
    "For effect_measure: type (OR/HR/RR/PR/IRR), value, ci_lower, ci_upper. "
    "For confidence_interval: lower, upper, level (default 95). "
    "For p_value: value, operator (= or < or >). "
    "For sample_size: value. "
    "For beta_coefficient: value, ci_lower, ci_upper, se. "
    "For mean_difference: value, ci_lower, ci_upper. "
    "For standard_deviation: value, mean, sd_type (SD or SE). "
    "For weighted_statistic: stat_type, value, weight_method."
)


def _empty_results() -> dict[str, list[dict]]:
    """Return an empty results dict with all 8 category keys."""
    return {cat: [] for cat in CATEGORIES}


def is_llm_available(model_id: str = "llama3.1:8b") -> bool:
    """Check if LangExtract and Ollama are available.

    Args:
        model_id: Ollama model identifier (unused in check, kept for API parity).

    Returns:
        True if langextract is importable and Ollama server is reachable.
    """
    try:
        import langextract  # noqa: F401
    except ImportError:
        return False

    try:
        resp = requests.get("http://localhost:11434", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _build_examples() -> list:
    """Build few-shot examples for LangExtract.

    Returns:
        List of lx.data.ExampleData objects covering common epi stat patterns.
    """
    import langextract as lx

    return [
        lx.data.ExampleData(
            text="The adjusted odds ratio was 2.45 (95% CI: 1.12-5.34, p=0.024).",
            extractions=[
                lx.data.Extraction(
                    extraction_class="effect_measure",
                    extraction_text="adjusted odds ratio was 2.45 (95% CI: 1.12-5.34",
                    attributes={
                        "type": "OR",
                        "value": "2.45",
                        "ci_lower": "1.12",
                        "ci_upper": "5.34",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="p_value",
                    extraction_text="p=0.024",
                    attributes={"value": "0.024", "operator": "="},
                ),
            ],
        ),
        lx.data.ExampleData(
            text="HR = 1.78 (1.23-2.56) after adjusting for age and sex. "
            "Beta coefficient: 0.34 (95% CI: 0.12, 0.56).",
            extractions=[
                lx.data.Extraction(
                    extraction_class="effect_measure",
                    extraction_text="HR = 1.78 (1.23-2.56)",
                    attributes={
                        "type": "HR",
                        "value": "1.78",
                        "ci_lower": "1.23",
                        "ci_upper": "2.56",
                    },
                ),
                lx.data.Extraction(
                    extraction_class="beta_coefficient",
                    extraction_text="Beta coefficient: 0.34 (95% CI: 0.12, 0.56)",
                    attributes={
                        "value": "0.34",
                        "ci_lower": "0.12",
                        "ci_upper": "0.56",
                        "se": "",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Mean difference was 3.2 (SD 1.5), n=450 participants.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="mean_difference",
                    extraction_text="Mean difference was 3.2",
                    attributes={"value": "3.2", "ci_lower": "", "ci_upper": ""},
                ),
                lx.data.Extraction(
                    extraction_class="standard_deviation",
                    extraction_text="SD 1.5",
                    attributes={"value": "1.5", "mean": "3.2", "sd_type": "SD"},
                ),
                lx.data.Extraction(
                    extraction_class="sample_size",
                    extraction_text="n=450",
                    attributes={"value": "450"},
                ),
            ],
        ),
    ]


def _safe_float(val: str) -> Optional[float]:
    """Safely convert a string to float, returning None on failure."""
    if not val or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: str) -> Optional[int]:
    """Safely convert a string to int, returning None on failure."""
    if not val or val.strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _extraction_to_dict(
    extraction, page: int
) -> Optional[tuple[str, dict]]:
    """Convert a LangExtract Extraction to our dict schema.

    Args:
        extraction: An lx.data.Extraction object.
        page: Page number where the text was found.

    Returns:
        Tuple of (category_key, dict) or None if unparseable.
    """
    cls = extraction.extraction_class
    attrs = extraction.attributes or {}
    context = extraction.extraction_text or ""

    if cls == "effect_measure":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "effect_measures", {
            "type": attrs.get("type", "OR"),
            "value": value,
            "ci_lower": _safe_float(attrs.get("ci_lower", "")),
            "ci_upper": _safe_float(attrs.get("ci_upper", "")),
            "adjusted": None,
            "adjusted_for": None,
            "context": context,
            "page": page,
        }

    elif cls == "confidence_interval":
        lower = _safe_float(attrs.get("lower", ""))
        upper = _safe_float(attrs.get("upper", ""))
        if lower is None or upper is None:
            return None
        return "confidence_intervals", {
            "level": _safe_int(attrs.get("level", "95")) or 95,
            "lower": lower,
            "upper": upper,
            "context": context,
            "page": page,
        }

    elif cls == "p_value":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "p_values", {
            "value": value,
            "operator": attrs.get("operator", "="),
            "context": context,
            "page": page,
        }

    elif cls == "sample_size":
        value = _safe_int(attrs.get("value", ""))
        if value is None:
            return None
        return "sample_sizes", {
            "value": value,
            "page": page,
        }

    elif cls == "beta_coefficient":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "beta_coefficients", {
            "value": value,
            "ci_lower": _safe_float(attrs.get("ci_lower", "")),
            "ci_upper": _safe_float(attrs.get("ci_upper", "")),
            "se": _safe_float(attrs.get("se", "")),
            "context": context,
            "page": page,
        }

    elif cls == "mean_difference":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "mean_differences", {
            "value": value,
            "ci_lower": _safe_float(attrs.get("ci_lower", "")),
            "ci_upper": _safe_float(attrs.get("ci_upper", "")),
            "context": context,
            "page": page,
        }

    elif cls == "standard_deviation":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "standard_deviations", {
            "value": value,
            "mean": _safe_float(attrs.get("mean", "")),
            "type": attrs.get("sd_type", "SD"),
            "context": context,
            "page": page,
        }

    elif cls == "weighted_statistic":
        value = _safe_float(attrs.get("value", ""))
        if value is None:
            return None
        return "weighted_statistics", {
            "stat_type": attrs.get("stat_type", ""),
            "value": value,
            "weight_method": attrs.get("weight_method") or None,
            "context": context,
            "page": page,
        }

    return None


def extract_with_llm(
    text: str, page: int = 1, model_id: str = "llama3.1:8b"
) -> dict[str, list[dict]]:
    """Extract epidemiological statistics from text using LangExtract + Ollama.

    Args:
        text: Page text to extract from.
        page: Page number for attribution.
        model_id: Ollama model identifier.

    Returns:
        Dict with 8 category keys, each containing a list of extracted dicts.
        Returns empty results on any failure.
    """
    try:
        import langextract as lx

        examples = _build_examples()

        result = lx.extract(
            text_or_documents=text,
            prompt_description=_PROMPT,
            examples=examples,
            model_id=model_id,
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False,
            extraction_passes=2,
            max_char_buffer=4000,
        )

        results = _empty_results()
        for extraction in result.extractions:
            parsed = _extraction_to_dict(extraction, page)
            if parsed is not None:
                category, item_dict = parsed
                results[category].append(item_dict)

        return results

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
