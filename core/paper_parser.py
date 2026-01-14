"""PDF text extraction and statistics parsing.

This module provides functions to extract text from PDFs and identify
epidemiological statistics using regular expressions.
"""

import re

import fitz  # PyMuPDF


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


def find_odds_ratios(text: str, page: int = 1) -> list[dict]:
    """Find odds ratios mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'ci_lower', 'ci_upper', 'context', 'page'.
    """
    results = []

    # Pattern: OR = 2.5 or OR: 2.5 or odds ratio, 2.5
    # With optional CI: OR = 2.5 (95% CI: 1.2-3.8) or OR = 2.5 (1.2, 3.8)
    # Note: Patterns require explicit delimiters to avoid matching page numbers
    patterns = [
        # OR = 2.5 (95% CI: 1.2-3.8) or OR: 2.5 (1.2-3.8)
        r"(?:OR|odds ratio)[,:\s=]+(\d+\.?\d*)\s*\(?\s*(?:95%?\s*CI)?[:\s]*(\d+\.?\d*)\s*[-–,]\s*(\d+\.?\d*)\s*\)?",
        # OR = 2.5 or OR: 2.5 or OR, 2.5 (requires = or : or comma, excludes percentages)
        r"(?:OR|odds ratio)\s*[=:,]\s*(\d+\.?\d*)(?!\s*%)",
        # OR 2.5 (95% CI - requires CI to follow to avoid false positives
        r"(?:OR|odds ratio)\s+(\d+\.?\d*)\s*\(\s*95%?\s*CI",
        # aOR = 2.5 or adjusted odds ratio, 2.5
        r"(?:aOR|adjusted odds ratio)[,:\s=]+(\d+\.?\d*)\s*\(?\s*(?:95%?\s*CI)?[:\s]*(\d+\.?\d*)\s*[-–,]\s*(\d+\.?\d*)\s*\)?",
        # adjusted odds ratio, 2.5 (standalone with comma)
        r"adjusted odds ratio\s*,\s*(\d+\.?\d*)(?!\s*%)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            result = {
                "value": float(groups[0]),
                "ci_lower": float(groups[1]) if len(groups) > 1 and groups[1] else None,
                "ci_upper": float(groups[2]) if len(groups) > 2 and groups[2] else None,
                "context": text[max(0, match.start() - 50) : match.end() + 50],
                "page": page,
            }
            # Avoid duplicates (compare without page for dedup within same page)
            if not any(
                r["value"] == result["value"]
                and r["ci_lower"] == result["ci_lower"]
                and r["ci_upper"] == result["ci_upper"]
                and r["page"] == result["page"]
                for r in results
            ):
                results.append(result)

    return results


def find_confidence_intervals(text: str, page: int = 1) -> list[dict]:
    """Find confidence intervals mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'lower', 'upper', 'level', 'context', 'page'.
    """
    results = []

    # Pattern: 95% CI: 1.2-3.8 or CI (1.2, 3.8) or confidence interval, 1.4 to 3.4
    patterns = [
        # 95% CI: 1.2-3.8 or 95% CI: 1.2 to 3.8
        r"(?:(\d+)%?\s*CI)[:\s]*\(?(\d+\.?\d*)\s*(?:[-–,]|to)\s*(\d+\.?\d*)\)?",
        # 95% confidence interval, 1.4 to 3.4 or confidence interval, 1.4 to 3.4
        r"(?:(\d+)%?\s*)?confidence interval[,:\s]+(\d+\.?\d*)\s*(?:[-–]|to)\s*(\d+\.?\d*)",
        # (1.2, 3.8) or (1.2-3.8) or (1.2 to 3.8)
        r"\((\d+\.?\d*)\s*(?:[-–,]|to)\s*(\d+\.?\d*)\)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 3:
                result = {
                    "level": int(groups[0]) if groups[0] else 95,
                    "lower": float(groups[1]),
                    "upper": float(groups[2]),
                    "context": text[max(0, match.start() - 30) : match.end() + 30],
                    "page": page,
                }
            else:
                result = {
                    "level": 95,
                    "lower": float(groups[0]),
                    "upper": float(groups[1]),
                    "context": text[max(0, match.start() - 30) : match.end() + 30],
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
