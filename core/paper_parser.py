"""PDF text extraction and statistics parsing.

This module provides functions to extract text from PDFs and identify
epidemiological statistics using regular expressions.
"""

import re

import fitz  # PyMuPDF


def _normalize_text(text: str) -> str:
    """Normalize text for consistent regex matching.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text with standardized whitespace and dashes.
    """
    # Replace en-dash, em-dash, and Unicode minus with hyphen
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    # Replace non-breaking space with regular space
    text = text.replace("\u00a0", " ")
    # Remove soft hyphens (U+00AD) - used for optional line breaks
    text = text.replace("\u00ad", "")
    # Fix hyphenated word breaks (e.g., "confi-\ndence" → "confidence")
    text = re.sub(r"-\s*\n\s*", "", text)
    # Standardize whitespace (collapse multiple spaces, normalize newlines)
    text = re.sub(r"\s+", " ", text)
    return text


def _detect_adjustment_status(
    text: str, match_start: int, match_end: int
) -> tuple[bool | None, str | None]:
    """Detect if an effect measure is adjusted and extract adjustment variables.

    Args:
        text: Full normalized text.
        match_start: Start position of the effect measure match.
        match_end: End position of the effect measure match.

    Returns:
        Tuple of (is_adjusted, adjusted_for_text).
        is_adjusted: True if adjusted, False if crude/unadjusted, None if unknown.
        adjusted_for_text: Extracted adjustment variables or None.
    """
    # Expand context window for adjustment detection (±150 chars)
    context_start = max(0, match_start - 150)
    context_end = min(len(text), match_end + 150)
    context = text[context_start:context_end].lower()

    # Position of match within context
    match_pos_in_context = match_start - context_start

    is_adjusted = None
    adjusted_for = None

    # Check for adjusted indicators BEFORE the match (within 50 chars)
    prefix_start = max(0, match_pos_in_context - 50)
    prefix = context[prefix_start:match_pos_in_context]

    # Adjusted indicators in prefix
    if re.search(r"\b(adjusted|aor|ahr|arr|apr|airr)\b", prefix):
        is_adjusted = True
    # Crude/unadjusted indicators in prefix
    elif re.search(r"\b(crude|unadjusted)\b", prefix):
        is_adjusted = False

    # Also check for adjustment phrases in wider context
    if is_adjusted is None:
        if re.search(r"after\s+adjust", context):
            is_adjusted = True
        elif re.search(r"after\s+controll", context):
            is_adjusted = True
        elif re.search(r"before\s+adjust", context):
            is_adjusted = False
        elif re.search(r"unadjusted\s+model", context):
            is_adjusted = False

    # Extract adjustment variables if present
    adj_match = re.search(
        r"adjust(?:ed|ing)\s+for\s+([^.;)]+?)(?:\.|;|\)|$)", context
    )
    if adj_match:
        adjusted_for = adj_match.group(1).strip()
        # Clean up: remove leading/trailing punctuation and whitespace
        adjusted_for = re.sub(r"^[,\s]+|[,\s]+$", "", adjusted_for)
        if adjusted_for:
            is_adjusted = True  # If we found adjustment variables, it's adjusted

    return is_adjusted, adjusted_for


def extract_text_from_pdf(file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text content from a PDF file by page.

    Args:
        file_bytes: PDF file contents as bytes.

    Returns:
        List of (page_number, text) tuples. Page numbers are 1-indexed.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        pages.append((page_num, page.get_text()))

    doc.close()
    return pages


def find_effect_measures(text: str, page: int = 1) -> list[dict]:
    """Find effect measures (OR, HR, RR, PR, IRR, β) mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'type', 'value', 'ci_lower', 'ci_upper', 'context', 'page'.
    """
    results = []

    # Normalize text for consistent matching
    normalized = _normalize_text(text)

    # Effect measure patterns organized by type
    # Each list: most specific (with CI) first, standalone last
    # Number pattern: \d+(?:\.\d+)? matches "2" or "2.5" but not "2."
    effect_patterns = {
        "OR": [
            # OR 2.2 (95% confidence interval, 1.4 to 3.4)
            r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?confidence\s+interval[,:\s]+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # OR 2.5 (95% CI: 1.2-3.8)
            r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # aOR 2.5; 95% CI 1.6-3.9
            r"(?:aor|adjusted\s+odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*[;,]?\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)",
            # OR 2.5 (1.2-3.8) - parenthetical CI
            r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)",
            # Standalone OR = 2.5
            r"(?:or|odds\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
            # Standalone aOR = 2.5
            r"(?:aor|adjusted\s+odds\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
        ],
        "HR": [
            # HR 1.45 (95% CI: 1.12-1.89)
            r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # HR 1.45 (1.12-1.89) - parenthetical CI
            r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)",
            # Standalone HR = 1.45
            r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
        ],
        "RR": [
            # RR 0.85 (95% CI: 0.72-0.99)
            r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # RR 0.85 (0.72-0.99) - parenthetical CI
            r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)",
            # Standalone RR = 0.85
            r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
        ],
        "PR": [
            # PR 1.2 (95% CI: 1.1-1.4)
            r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # PR 1.2 (1.1-1.4) - parenthetical CI
            r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)",
            # Standalone PR = 1.2
            r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
        ],
        "IRR": [
            # IRR 1.5 (95% CI: 1.2-1.9)
            r"(?:irr|incidence\s+rate\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)",
            # IRR 1.5 (1.2-1.9) - parenthetical CI
            r"(?:irr|incidence\s+rate\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)",
            # Standalone IRR = 1.5
            r"(?:irr|incidence\s+rate\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)",
        ],
        "β": [
            # β = 0.45 (95% CI: 0.12-0.78)
            r"(?:β|beta)\s*[=:,]\s*(-?\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)\s*\)",
            # Standalone β = 0.45
            r"(?:β|beta)\s*[=:,]\s*(-?\d+(?:\.\d+)?)(?!\s*%)",
        ],
    }

    for measure_type, patterns in effect_patterns.items():
        for pattern in patterns:
            for match in re.finditer(pattern, normalized, re.IGNORECASE):
                groups = match.groups()

                # Parse based on number of captured groups
                if len(groups) == 4:
                    # (value, ci_level?, ci_lower, ci_upper)
                    value = float(groups[0])
                    ci_lower = float(groups[2])
                    ci_upper = float(groups[3])
                elif len(groups) == 3:
                    # (value, ci_lower, ci_upper)
                    value = float(groups[0])
                    ci_lower = float(groups[1])
                    ci_upper = float(groups[2])
                else:
                    # (value) only
                    value = float(groups[0])
                    ci_lower = None
                    ci_upper = None

                # Detect adjustment status
                is_adjusted, adjusted_for = _detect_adjustment_status(
                    normalized, match.start(), match.end()
                )

                # Also check if pattern itself indicates adjusted (aOR, aHR, etc.)
                pattern_lower = pattern.lower()
                if "aor" in pattern_lower or "ahr" in pattern_lower or "arr" in pattern_lower:
                    is_adjusted = True
                elif "adjusted" in pattern_lower:
                    is_adjusted = True

                result = {
                    "type": measure_type,
                    "value": value,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "adjusted": is_adjusted,
                    "adjusted_for": adjusted_for,
                    "context": normalized[max(0, match.start() - 50) : match.end() + 50],
                    "page": page,
                }

                # Avoid duplicates (same type + value + CI on same page)
                if not any(
                    r["type"] == result["type"]
                    and r["value"] == result["value"]
                    and r["ci_lower"] == result["ci_lower"]
                    and r["ci_upper"] == result["ci_upper"]
                    and r["page"] == result["page"]
                    for r in results
                ):
                    results.append(result)

    return results


# Backward compatibility alias
def find_odds_ratios(text: str, page: int = 1) -> list[dict]:
    """Deprecated: Use find_effect_measures() instead."""
    return find_effect_measures(text, page)


def find_confidence_intervals(text: str, page: int = 1) -> list[dict]:
    """Find confidence intervals mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'lower', 'upper', 'level', 'context', 'page'.
    """
    results = []

    # Normalize text for consistent matching
    normalized = _normalize_text(text)

    # Comprehensive CI patterns
    # Supports: dash, comma, and "to" separators; negative numbers; brackets
    # Number pattern: -?\d+(?:\.\d+)? matches "-2", "2", "2.5", "-0.32"
    patterns = [
        # 95% CI: 1.2-3.8 or 95% CI: 1.2, 3.8 (comma or dash separator)
        r"(\d+)\s*%?\s*ci[:\s]*\(?(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)?",
        # 95% CI = 1.2-3.8 or 95% CI = 1.2, 3.8 (equals sign)
        r"(\d+)\s*%?\s*ci\s*=\s*\(?(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)?",
        # 95% confidence interval, 1.4 to 3.4 or 1.4, 3.4
        r"(\d+)\s*%?\s*confidence\s+interval[,:\s]+(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # confidence interval, 1.4 to 3.4 (no percentage)
        r"confidence\s+interval[,:\s]+(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # confidence interval of 1.4 to 3.4
        r"confidence\s+interval\s+of\s+(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # CI: 1.2 to 3.8 or CI = 1.2-3.8 or CI 1.2, 3.4
        r"\bci[:\s=]+(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # (1.2-3.8) or (1.2, 3.4) when preceded by CI
        r"(?:ci|confidence\s+interval)[,:\s]*\((-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)",
        # Bracket notation: [1.2, 3.4] or [1.2-3.4] when preceded by CI
        r"(?:ci|confidence\s+interval)[,:\s]*\[(-?\d+(?:\.\d+)?)\s*(?:[-,;]|to)\s*(-?\d+(?:\.\d+)?)\]",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            groups = match.groups()

            # Parse based on number of captured groups
            if len(groups) >= 3:
                level = int(groups[0]) if groups[0] else 95
                lower = float(groups[1])
                upper = float(groups[2])
            else:
                level = 95
                lower = float(groups[0])
                upper = float(groups[1])

            # Validate CI values
            if abs(lower) > 100 or abs(upper) > 100:
                continue  # Skip year-like ranges
            if lower >= upper:
                continue  # Skip invalid CI (lower should be less than upper)

            result = {
                "level": level,
                "lower": lower,
                "upper": upper,
                "context": normalized[max(0, match.start() - 30) : match.end() + 30],
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["lower"] == result["lower"]
                and r["upper"] == result["upper"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_p_values(text: str, page: int = 1) -> list[dict]:
    """Find p-values mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'operator', 'context', 'page'.
    """
    results = []

    # Patterns: p < 0.05, p = 0.001, P-value: 0.03, p<.001
    patterns = [
        # p < 0.05 or p < .05 - captures operator and digits after decimal
        r"[Pp][-\s]?(?:value)?[:\s]*([<>=≤≥])\s*0?\.?(\d+)",
        # p = 0.001 (full decimal format)
        r"[Pp][-\s]?(?:value)?[:\s]*(\d+\.?\d*(?:e-?\d+)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()

            if len(groups) == 2:
                # Pattern with operator - digits are after the decimal point
                operator = groups[0]
                value_str = groups[1]
                # Always prepend "0." since pattern consumed the "0." prefix
                value = float("0." + value_str)
            else:
                # Pattern without operator - full decimal number
                operator = "="
                value = float(groups[0])

            result = {
                "value": value,
                "operator": operator,
                "context": text[max(0, match.start() - 30) : match.end() + 30],
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["value"] == result["value"]
                and r["operator"] == result["operator"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_sample_sizes(text: str, page: int = 1) -> list[dict]:
    """Find sample sizes mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'page'. Minimum threshold is 50.
    """
    results = []

    # Patterns support comma-separated numbers (e.g., 933,921)
    # Use \d[\d,]* to require at least one digit ([\d,]+ could match just commas)
    patterns = [
        # n = 933,921 or N = 500 or n: 500
        r"[Nn]\s*[=:]\s*(\d[\d,]*)",
        # sample size of 1,234
        r"sample\s+size\s+(?:of\s+)?(\d[\d,]*)",
        # 933,921 ± 88,474 adults/patients/participants/subjects/individuals
        r"(\d[\d,]*)\s*[±]\s*\d[\d,]*\s+(?:adults|patients|participants|subjects|individuals)",
        # 933,921 adults/patients/participants/subjects/individuals
        r"(\d[\d,]*)\s+(?:adults|patients|participants|subjects|individuals)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Strip commas and convert to int
            value = int(match.group(1).replace(",", ""))
            # Filter out small numbers (likely references)
            if value >= 50:
                # Avoid duplicates on same page
                if not any(r["value"] == value and r["page"] == page for r in results):
                    results.append({"value": value, "page": page})

    # Sort by value descending
    return sorted(results, key=lambda x: x["value"], reverse=True)


def find_beta_coefficients(text: str, page: int = 1) -> list[dict]:
    """Find beta coefficients mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'ci_lower', 'ci_upper', 'se', 'context', 'page'.
    """
    results = []
    normalized = _normalize_text(text)

    # Patterns for beta coefficients
    # Ordered: most specific (with CI) first, then SE, then standalone
    patterns = [
        # β = 2.18 (95% CI: 0.30-4.01) or β: 2.18; 95% CI 0.30, 4.01
        (
            r"(?:β|beta)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*[;,]?\s*\(?(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)?",
            "ci",
        ),
        # coefficient = 0.45 (95% CI: 0.12-0.78)
        (
            r"(?:coefficient|coef)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*[;,]?\s*\(?(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)?",
            "ci",
        ),
        # B = 1.23 (SE = 0.45) - unstandardized beta (requires SE context)
        (
            r"\bB\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*\(\s*SE\s*[=:]\s*(\d+(?:\.\d+)?)\s*\)",
            "se",
        ),
        # coefficient B = 1.23 - requires "coefficient" prefix for uppercase B
        (r"coefficient\s+B\s*[=:]\s*(-?\d+(?:\.\d+)?)", "standalone"),
        # Standalone β = 2.18 or beta = -0.52
        (r"(?:β|beta)\s*[=:]\s*(-?\d+(?:\.\d+)?)(?!\s*%)", "standalone"),
        # Standalone coefficient = 0.45
        (r"(?:coefficient|coef)\s*[=:]\s*(-?\d+(?:\.\d+)?)(?!\s*%)", "standalone"),
    ]

    for pattern, pattern_type in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            groups = match.groups()

            if pattern_type == "ci":
                # (value, ci_level?, ci_lower, ci_upper)
                value = float(groups[0])
                ci_lower = float(groups[2])
                ci_upper = float(groups[3])
                se = None
            elif pattern_type == "se":
                # (value, se)
                value = float(groups[0])
                se = float(groups[1])
                ci_lower = None
                ci_upper = None
            else:
                # Standalone (value only)
                value = float(groups[0])
                ci_lower = None
                ci_upper = None
                se = None

            result = {
                "value": value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "se": se,
                "context": normalized[max(0, match.start() - 30) : match.end() + 30],
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["value"] == result["value"]
                and r["ci_lower"] == result["ci_lower"]
                and r["ci_upper"] == result["ci_upper"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_mean_differences(text: str, page: int = 1) -> list[dict]:
    """Find mean differences mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'ci_lower', 'ci_upper', 'context', 'page'.
    """
    results = []
    normalized = _normalize_text(text)

    # Patterns for mean differences
    # Ordered: most specific (with CI) first, then standalone
    patterns = [
        # mean difference: 2.5 (95% CI: 1.2-3.8) or MD = 2.5; 95% CI 1.2, 3.8
        r"(?:mean\s+difference|md)\s*[=:]\s*(-?\d+(?:\.\d+)?)\s*[;,]?\s*\(?(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)\)?",
        # difference of 3.2 points; 95% CI 2.1-4.3 (requires "points")
        r"difference\s+of\s+(-?\d+(?:\.\d+)?)\s*points?\s*[;,]?\s*(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # 0.29 points; 95% CI 0.26-0.31 (requires "points" + CI)
        r"(-?\d+(?:\.\d+)?)\s*points?\s*[;,]\s*(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(-?\d+(?:\.\d+)?)",
        # Standalone mean difference = 2.5 or MD = 2.5
        r"(?:mean\s+difference|md)\s*[=:]\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            groups = match.groups()

            if len(groups) >= 4 and groups[2] is not None:
                # Pattern with CI
                value = float(groups[0])
                ci_lower = float(groups[2])
                ci_upper = float(groups[3])
            else:
                # Standalone pattern
                value = float(groups[0])
                ci_lower = None
                ci_upper = None

            result = {
                "value": value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "context": normalized[max(0, match.start() - 30) : match.end() + 30],
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["value"] == result["value"]
                and r["ci_lower"] == result["ci_lower"]
                and r["ci_upper"] == result["ci_upper"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_standard_deviations(text: str, page: int = 1) -> list[dict]:
    """Find standard deviations and standard errors mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'mean', 'value', 'type' (SD/SE), 'context', 'page'.
    """
    results = []
    normalized = _normalize_text(text)

    # Patterns for SD/SE
    # Each tuple: (pattern, type, has_mean)
    patterns = [
        # mean 3.17 (SD 1.19) or 3.17 (SD = 1.19)
        (r"(?:mean\s*[=:]?\s*)?(-?\d+(?:\.\d+)?)\s*\(\s*SD\s*[=:]?\s*(\d+(?:\.\d+)?)\s*\)", "SD", True),
        # 3.17 ± 1.19 (plus-minus notation)
        (r"(-?\d+(?:\.\d+)?)\s*[±]\s*(\d+(?:\.\d+)?)", "SD", True),
        # SD = 2.5 or SD: 2.5 (standalone)
        (r"\bSD\s*[=:]\s*(\d+(?:\.\d+)?)", "SD", False),
        # SE = 0.45 or SE: 0.45 (standalone)
        (r"\bSE\s*[=:]\s*(\d+(?:\.\d+)?)", "SE", False),
        # standard deviation = 2.5
        (r"standard\s+deviation\s*[=:]\s*(\d+(?:\.\d+)?)", "SD", False),
        # standard error = 0.45
        (r"standard\s+error\s*[=:]\s*(\d+(?:\.\d+)?)", "SE", False),
    ]

    for pattern, sd_type, has_mean in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            groups = match.groups()

            if has_mean and len(groups) >= 2:
                mean_value = float(groups[0])
                sd_value = float(groups[1])
            else:
                mean_value = None
                sd_value = float(groups[0])

            result = {
                "mean": mean_value,
                "value": sd_value,
                "type": sd_type,
                "context": normalized[max(0, match.start() - 30) : match.end() + 30],
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["value"] == result["value"]
                and r["mean"] == result["mean"]
                and r["type"] == result["type"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_weighted_statistics(text: str, page: int = 1) -> list[dict]:
    """Find weighted statistics mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'stat_type', 'value', 'weight_method', 'context', 'page'.
    """
    results = []
    normalized = _normalize_text(text)

    # Patterns for weighted statistics
    # Each tuple: (pattern, stat_type)
    patterns = [
        # weighted prevalence: 25.3% or weighted prevalence = 25.3%
        (r"weighted\s+prevalence[:\s=]+(\d+(?:\.\d+)?)\s*%?", "prevalence"),
        # survey-weighted mean: 3.5 or survey-weighted mean = 3.5
        (r"(?:survey[- ])?weighted\s+mean[:\s=]+(\d+(?:\.\d+)?)", "mean"),
        # IPW estimate: 1.45 or IPTW: 1.45 or IPW = 1.45
        (r"(?:IPW|IPTW)\s*(?:estimate)?[:\s=]+(\d+(?:\.\d+)?)", "IPW"),
        # PS-weighted OR: 2.1 or propensity-weighted OR = 2.1
        (r"(?:PS|propensity)[- ]?weighted\s+(?:OR|HR|RR)[:\s=]+(\d+(?:\.\d+)?)", "PS-weighted"),
        # weighted OR: 2.1 or weighted HR = 1.5
        (r"weighted\s+(?:OR|HR|RR)[:\s=]+(\d+(?:\.\d+)?)", "weighted"),
    ]

    for pattern, stat_type in patterns:
        for match in re.finditer(pattern, normalized, re.IGNORECASE):
            value = float(match.group(1))

            # Detect weight method from context
            context = normalized[max(0, match.start() - 50) : match.end() + 50]
            context_lower = context.lower()

            weight_method = None
            if "ipw" in context_lower or "iptw" in context_lower:
                weight_method = "IPW"
            elif "propensity" in context_lower or "ps-weight" in context_lower:
                weight_method = "Propensity score"
            elif "survey" in context_lower:
                weight_method = "Survey weights"
            elif "sampling" in context_lower:
                weight_method = "Sampling weights"

            result = {
                "stat_type": stat_type,
                "value": value,
                "weight_method": weight_method,
                "context": context,
                "page": page,
            }

            # Avoid duplicates on same page
            if not any(
                r["stat_type"] == result["stat_type"]
                and r["value"] == result["value"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results
