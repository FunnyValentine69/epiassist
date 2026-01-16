#!/usr/bin/env python3
"""Deep diagnostic tool for Paper Analyzer extraction issues.

This script analyzes PDFs to identify why some papers show zero extractions
and suggests pattern fixes.

Usage:
    python scripts/deep_diagnose.py              # Defaults to test_papers/
    python scripts/deep_diagnose.py ./my_papers  # Custom folder
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paper_parser import (
    extract_text_from_pdf,
    _normalize_text,
    find_effect_measures,
    find_confidence_intervals,
    find_p_values,
    find_sample_sizes,
)


# Indicator words to search for (epidemiological keywords)
INDICATOR_PATTERNS = [
    (r"confidence", "confidence"),
    (r"interval", "interval"),
    (r"\bCI\b", "CI"),
    (r"\bOR\b", "OR"),
    (r"\bRR\b", "RR"),
    (r"\bHR\b", "HR"),
    (r"\bPR\b", "PR"),
    (r"\bIRR\b", "IRR"),
    (r"odds\s+ratio", "odds ratio"),
    (r"hazard\s+ratio", "hazard ratio"),
    (r"relative\s+risk", "relative risk"),
    (r"prevalence\s+ratio", "prevalence ratio"),
    (r"p\s*-?\s*value", "p-value"),
    (r"p\s*[=<>]", "p =/</>"),
    (r"\bn\s*=", "n ="),
    (r"\bsample\b", "sample"),
    (r"\bparticipants\b", "participants"),
    (r"\bsubjects\b", "subjects"),
]

# Patterns for uncaptured statistics (potential pattern gaps)
UNCAPTURED_PATTERNS = [
    (r"\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)", "Comma-sep parens (X, Y)"),
    (r"\((-?\d+\.?\d*)\s*;\s*(-?\d+\.?\d*)\)", "Semicolon-sep parens (X; Y)"),
    (r"\[(-?\d+\.?\d*)\s*[,;]\s*(-?\d+\.?\d*)\]", "Bracket notation [X, Y]"),
    (r"ci[:\s]+(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", "CI with comma sep"),
]


def analyze_pdf(pdf_path: Path) -> dict:
    """Analyze a single PDF and return comprehensive diagnostics.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with quality metrics, indicator counts, extractions, and diagnostics.
    """
    result = {
        "filename": pdf_path.name,
        "pages": 0,
        "total_chars": 0,
        "chars_per_page": 0.0,
        "low_content_pages": [],
        "indicator_counts": {},
        "extractions": {
            "effect_measures": 0,
            "confidence_intervals": 0,
            "p_values": 0,
            "sample_sizes": 0,
        },
        "uncaptured_patterns": {},
        "raw_text_preview": "",
        "lines_with_parens": [],
        "table_pages": [],
        "diagnosis": "",
        "cause": "",
    }

    try:
        file_bytes = pdf_path.read_bytes()
        pages = extract_text_from_pdf(file_bytes)
    except Exception as e:
        result["diagnosis"] = "FAILED"
        result["cause"] = f"PDF_ISSUE: {e}"
        return result

    if not pages:
        result["diagnosis"] = "FAILED"
        result["cause"] = "PDF_ISSUE: No pages extracted"
        return result

    # Collect all text and analyze per page
    all_text = ""
    all_normalized = ""
    page_chars = []

    for page_num, page_text in pages:
        normalized = _normalize_text(page_text)
        all_text += page_text + "\n"
        all_normalized += normalized + " "
        char_count = len(page_text)
        page_chars.append(char_count)

        if char_count < 500:
            result["low_content_pages"].append(page_num)

        # Check for tabular data indicators
        lines = page_text.split("\n")
        tab_lines = sum(1 for line in lines if line.count("\t") >= 2)
        space_aligned = sum(1 for line in lines if len(re.findall(r"\s{3,}\d", line)) >= 2)
        if tab_lines >= 3 or space_aligned >= 3:
            result["table_pages"].append(page_num)

    result["pages"] = len(pages)
    result["total_chars"] = sum(page_chars)
    result["chars_per_page"] = result["total_chars"] / len(pages) if pages else 0

    # Count indicator words
    for pattern, label in INDICATOR_PATTERNS:
        count = len(re.findall(pattern, all_normalized, re.IGNORECASE))
        if count > 0:
            result["indicator_counts"][label] = count

    # Run extraction functions
    em_results = []
    ci_results = []
    pval_results = []
    sample_results = []

    for page_num, page_text in pages:
        normalized = _normalize_text(page_text)
        em_results.extend(find_effect_measures(normalized, page_num))
        ci_results.extend(find_confidence_intervals(normalized, page_num))
        pval_results.extend(find_p_values(normalized, page_num))
        sample_results.extend(find_sample_sizes(normalized, page_num))

    result["extractions"]["effect_measures"] = len(em_results)
    result["extractions"]["confidence_intervals"] = len(ci_results)
    result["extractions"]["p_values"] = len(pval_results)
    result["extractions"]["sample_sizes"] = len(sample_results)

    # Find uncaptured patterns
    for pattern, label in UNCAPTURED_PATTERNS:
        matches = re.findall(pattern, all_normalized, re.IGNORECASE)
        if matches:
            # Filter to only show valid-looking numeric pairs
            valid_matches = []
            for m in matches:
                try:
                    v1, v2 = float(m[0]), float(m[1])
                    # Skip if looks like a year range or page numbers
                    if v1 > 100 and v2 > 100:
                        continue
                    valid_matches.append(f"({m[0]}, {m[1]})")
                except (ValueError, IndexError):
                    continue
            if valid_matches:
                result["uncaptured_patterns"][label] = len(valid_matches)

    # Check if any category has zero extractions
    has_zero = any(v == 0 for v in result["extractions"].values())

    if has_zero:
        # Provide raw text preview
        result["raw_text_preview"] = all_text[:2000]

        # Find lines with parenthetical numbers
        lines = all_text.split("\n")
        paren_pattern = r"\([\d.-]+\s*[,;-]\s*[\d.-]+\)"
        for i, line in enumerate(lines[:500], 1):  # Check first 500 lines
            if re.search(paren_pattern, line):
                # Truncate long lines
                display_line = line[:100] + "..." if len(line) > 100 else line
                result["lines_with_parens"].append((i, display_line.strip()))
                if len(result["lines_with_parens"]) >= 10:
                    break

    # Determine diagnosis
    ext = result["extractions"]
    zeros = [k for k, v in ext.items() if v == 0]
    non_zeros = [k for k, v in ext.items() if v > 0]

    if len(zeros) == 0:
        result["diagnosis"] = "GOOD"
        result["cause"] = "All categories extracted"
    elif len(zeros) == 4:
        result["diagnosis"] = "FAILED"
        # Determine cause
        if result["total_chars"] < 1000:
            result["cause"] = "PDF_ISSUE: Very low character count"
        elif sum(result["indicator_counts"].values()) < 5:
            result["cause"] = "NO_STATS: Few indicator words found"
        else:
            result["cause"] = "PATTERN_ISSUE: Indicator words present but no extractions"
    else:
        result["diagnosis"] = "PARTIAL"
        zero_labels = ", ".join(z.replace("_", " ") for z in zeros)
        if result["uncaptured_patterns"]:
            patterns_found = ", ".join(result["uncaptured_patterns"].keys())
            result["cause"] = f"Missing: {zero_labels}. Uncaptured formats: {patterns_found}"
        else:
            result["cause"] = f"Missing: {zero_labels}"

    return result


def print_report(results: list[dict]) -> None:
    """Print the full diagnostic report.

    Args:
        results: List of analysis results from analyze_pdf.
    """
    print("=" * 80)
    print("DEEP DIAGNOSIS REPORT")
    print("=" * 80)
    print()

    good_papers = []
    partial_papers = []
    failed_papers = []
    all_uncaptured = Counter()

    for r in results:
        # Categorize
        if r["diagnosis"] == "GOOD":
            good_papers.append(r["filename"])
        elif r["diagnosis"] == "PARTIAL":
            partial_papers.append((r["filename"], r["cause"]))
        else:
            failed_papers.append((r["filename"], r["cause"]))

        # Collect uncaptured patterns
        for pattern, count in r.get("uncaptured_patterns", {}).items():
            all_uncaptured[pattern] += count

        # Print individual report
        print(f"FILE: {r['filename']}")
        print("-" * 80)

        # Quality metrics
        quality = "GOOD" if r["chars_per_page"] >= 1000 else "LOW"
        print(f"Quality: {r['pages']} pages, {r['total_chars']:,} chars "
              f"({r['chars_per_page']:.0f}/page avg) - {quality}")

        if r["low_content_pages"]:
            print(f"Low content pages (<500 chars): {r['low_content_pages']}")
        else:
            print("Low content pages: None")

        # Indicator words
        print()
        print("Indicator Words Found:")
        if r["indicator_counts"]:
            items = [f"{k}: {v}" for k, v in sorted(r["indicator_counts"].items(), key=lambda x: -x[1])]
            # Print in rows of 5
            for i in range(0, len(items), 5):
                print(f"  {', '.join(items[i:i+5])}")
        else:
            print("  None found")

        # Extraction results
        print()
        print("Extraction Results:")
        ext = r["extractions"]
        for category, count in ext.items():
            label = category.replace("_", " ").title()
            status = "\u2713" if count > 0 else "\u2717 (ISSUE)"
            print(f"  {label}: {count} {status}")

        # Uncaptured patterns
        if r["uncaptured_patterns"]:
            print()
            print("Uncaptured Patterns Found:")
            for pattern, count in r["uncaptured_patterns"].items():
                print(f"  {pattern}: {count} occurrences")

        # Raw text preview (only for papers with zeros)
        if r["raw_text_preview"]:
            print()
            print("Raw Text Preview (first 2000 chars):")
            preview = r["raw_text_preview"][:500].replace("\n", " ")[:200]
            print(f"  \"{preview}...\"")

        # Lines with parenthetical numbers
        if r["lines_with_parens"]:
            print()
            print("Lines with Parenthetical Numbers:")
            for line_num, line_text in r["lines_with_parens"][:5]:
                print(f"  Line {line_num}: {line_text}")
            if len(r["lines_with_parens"]) > 5:
                print(f"  ... and {len(r['lines_with_parens']) - 5} more")

        # Table detection
        if r["table_pages"]:
            print()
            print(f"Table Detection: Possible tabular data on pages {r['table_pages']}")

        # Diagnosis
        print()
        print(f"Diagnosis: {r['diagnosis']} - {r['cause']}")
        print()
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(results)
    print(f"GOOD ({len(good_papers)}/{total}):    {', '.join(good_papers) if good_papers else 'None'}")
    print(f"PARTIAL ({len(partial_papers)}/{total}): ", end="")
    if partial_papers:
        print(", ".join(f"{p[0]}" for p in partial_papers))
    else:
        print("None")
    print(f"FAILED ({len(failed_papers)}/{total}):  ", end="")
    if failed_papers:
        print(", ".join(f"{p[0]}" for p in failed_papers))
    else:
        print("None")

    # Pattern suggestions
    if all_uncaptured:
        print()
        print("=" * 80)
        print("SUGGESTED PATTERN ADDITIONS")
        print("=" * 80)
        print()

        if all_uncaptured.get("Comma-sep parens (X, Y)", 0) > 0 or all_uncaptured.get("CI with comma sep", 0) > 0:
            print("1. Add comma separator to CI patterns:")
            print('   r"(\\d+)\\s*%?\\s*ci[:\\s]*(-?\\d+(?:\\.\\d+)?)\\s*,\\s*(-?\\d+(?:\\.\\d+)?)"')
            print()

        if all_uncaptured.get("Bracket notation [X, Y]", 0) > 0:
            print("2. Add bracket notation for CIs:")
            print('   r"\\[(-?\\d+(?:\\.\\d+)?)\\s*[,;-]\\s*(-?\\d+(?:\\.\\d+)?)\\]"')
            print()

        if all_uncaptured.get("Semicolon-sep parens (X; Y)", 0) > 0:
            print("3. Add semicolon separator:")
            print('   r"(\\d+)\\s*%?\\s*ci[:\\s]*(-?\\d+(?:\\.\\d+)?)\\s*;\\s*(-?\\d+(?:\\.\\d+)?)"')
            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Deep diagnostic tool for Paper Analyzer extraction issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/deep_diagnose.py
  python scripts/deep_diagnose.py ./my_papers
        """,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="test_papers",
        help="Folder containing PDF files (default: test_papers/)",
    )
    args = parser.parse_args()

    # Resolve paths
    folder = Path(args.folder)
    if not folder.is_absolute():
        folder = Path.cwd() / folder

    if not folder.exists():
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)

    # Find PDFs
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{folder}'.")
        sys.exit(0)

    print(f"Analyzing {len(pdfs)} PDFs...")
    print()

    # Analyze each PDF
    results = []
    for pdf_path in sorted(pdfs):
        result = analyze_pdf(pdf_path)
        results.append(result)

    # Print report
    print_report(results)


if __name__ == "__main__":
    main()
