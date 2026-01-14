"""PDF text extraction and statistics parsing.

This module provides functions to extract text from PDFs and identify
epidemiological statistics using regular expressions.
"""

import re

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file.

    Args:
        file_bytes: PDF file contents as bytes.

    Returns:
        Extracted text as a single string.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []

    for page in doc:
        text_parts.append(page.get_text())

    doc.close()
    return "\n".join(text_parts)


def find_odds_ratios(text: str) -> list[dict]:
    """Find odds ratios mentioned in text.

    Args:
        text: Text to search.

    Returns:
        List of dicts with 'value', 'ci_lower', 'ci_upper', 'context'.
    """
    results = []

    # Pattern: OR = 2.5 or OR: 2.5 or odds ratio of 2.5
    # With optional CI: OR = 2.5 (95% CI: 1.2-3.8) or OR = 2.5 (1.2, 3.8)
    patterns = [
        # OR = 2.5 (95% CI: 1.2-3.8)
        r"(?:OR|odds ratio)[:\s=]+(\d+\.?\d*)\s*\(?\s*(?:95%?\s*CI)?[:\s]*(\d+\.?\d*)\s*[-–,]\s*(\d+\.?\d*)\s*\)?",
        # OR = 2.5
        r"(?:OR|odds ratio)[:\s=]+(\d+\.?\d*)",
        # aOR = 2.5 (adjusted OR)
        r"(?:aOR|adjusted odds ratio)[:\s=]+(\d+\.?\d*)\s*\(?\s*(?:95%?\s*CI)?[:\s]*(\d+\.?\d*)\s*[-–,]\s*(\d+\.?\d*)\s*\)?",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            result = {
                "value": float(groups[0]),
                "ci_lower": float(groups[1]) if len(groups) > 1 and groups[1] else None,
                "ci_upper": float(groups[2]) if len(groups) > 2 and groups[2] else None,
                "context": text[max(0, match.start() - 50) : match.end() + 50],
            }
            # Avoid duplicates
            if result not in results:
                results.append(result)

    return results


def find_confidence_intervals(text: str) -> list[dict]:
    """Find confidence intervals mentioned in text.

    Args:
        text: Text to search.

    Returns:
        List of dicts with 'lower', 'upper', 'level', 'context'.
    """
    results = []

    # Pattern: 95% CI: 1.2-3.8 or CI (1.2, 3.8) or (95% CI 1.2 to 3.8)
    patterns = [
        r"(?:(\d+)%?\s*CI)[:\s]*\(?(\d+\.?\d*)\s*[-–,to]\s*(\d+\.?\d*)\)?",
        r"\((\d+\.?\d*)\s*[-–,]\s*(\d+\.?\d*)\)",
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
                }
            else:
                result = {
                    "level": 95,
                    "lower": float(groups[0]),
                    "upper": float(groups[1]),
                    "context": text[max(0, match.start() - 30) : match.end() + 30],
                }

            if result not in results:
                results.append(result)

    return results


def find_p_values(text: str) -> list[dict]:
    """Find p-values mentioned in text.

    Args:
        text: Text to search.

    Returns:
        List of dicts with 'value', 'operator', 'context'.
    """
    results = []

    # Patterns: p < 0.05, p = 0.001, P-value: 0.03, p<.001
    patterns = [
        r"[Pp][-\s]?(?:value)?[:\s]*([<>=≤≥])\s*0?\.?(\d+)",
        r"[Pp][-\s]?(?:value)?[:\s]*(\d+\.?\d*(?:e-?\d+)?)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            groups = match.groups()

            if len(groups) == 2:
                # Pattern with operator
                operator = groups[0]
                value_str = groups[1]
                if not value_str.startswith("0"):
                    value_str = "0." + value_str
                value = float(value_str)
            else:
                # Pattern without operator
                operator = "="
                value = float(groups[0])

            result = {
                "value": value,
                "operator": operator,
                "context": text[max(0, match.start() - 30) : match.end() + 30],
            }

            if result not in results:
                results.append(result)

    return results


def find_sample_sizes(text: str) -> list[int]:
    """Find sample sizes mentioned in text.

    Args:
        text: Text to search.

    Returns:
        List of sample size integers.
    """
    results = []

    # Patterns: n = 500, N=1000, sample size of 500, n: 500
    patterns = [
        r"[Nn][:\s=]+(\d{2,})",
        r"sample\s+size[:\s=of]+(\d+)",
        r"(\d+)\s+(?:participants|subjects|patients|individuals)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            value = int(match.group(1))
            if value not in results and value >= 10:  # Filter out small numbers
                results.append(value)

    return sorted(set(results), reverse=True)
