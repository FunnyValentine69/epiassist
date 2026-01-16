#!/usr/bin/env python3
"""Diagnostic tool for Paper Analyzer extraction patterns.

This script processes PDFs and generates reports showing:
1. What statistics were extracted by each pattern
2. Per-pattern match counts for debugging

Usage:
    python scripts/diagnose_extraction.py              # Defaults to test_papers/
    python scripts/diagnose_extraction.py ./my_papers  # Custom folder
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.paper_parser import (
    extract_text_from_pdf,
    _normalize_text,
    find_effect_measures,
    find_confidence_intervals,
    find_p_values,
    find_sample_sizes,
    find_beta_coefficients,
    find_mean_differences,
    find_standard_deviations,
)


# =============================================================================
# PATTERNS FROM paper_parser.py (for per-pattern tracking)
# NOTE: Per-pattern counts may exceed totals due to deduplication in the
# extraction functions. e.g., em_p1+em_p2=8 but em_total=5 because multiple
# patterns can match the same text, but only one result is kept.
# =============================================================================

# Effect measure patterns (OR, HR, RR, PR, IRR, β) from paper_parser.py
EFFECT_PATTERNS = [
    # OR patterns
    (r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?confidence\s+interval[,:\s]+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "OR + full CI"),
    (r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "OR + CI abbrev"),
    (r"(?:aor|adjusted\s+odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*[;,]?\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", "aOR + CI"),
    (r"(?:or|odds\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)", "OR + paren CI"),
    (r"(?:or|odds\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone OR"),
    (r"(?:aor|adjusted\s+odds\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone aOR"),
    # HR patterns
    (r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "HR + CI"),
    (r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)", "HR + paren CI"),
    (r"(?:a?hr|(?:adjusted\s+)?hazard\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone HR"),
    # RR patterns
    (r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "RR + CI"),
    (r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)", "RR + paren CI"),
    (r"(?:a?rr|(?:adjusted\s+)?(?:relative\s+risk|risk\s+ratio))\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone RR"),
    # PR patterns
    (r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "PR + CI"),
    (r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)", "PR + paren CI"),
    (r"(?:a?pr|(?:adjusted\s+)?prevalence\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone PR"),
    # IRR patterns
    (r"(?:irr|incidence\s+rate\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*\)", "IRR + CI"),
    (r"(?:irr|incidence\s+rate\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)", "IRR + paren CI"),
    (r"(?:irr|incidence\s+rate\s+ratio)\s*[=:,]\s*(\d+(?:\.\d+)?)(?!\s*%)", "Standalone IRR"),
    # β patterns
    (r"(?:β|beta)\s*[=:,]\s*(-?\d+(?:\.\d+)?)\s*\(\s*(?:(\d+)%?\s*)?ci[:\s]*(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)\s*\)", "Beta + CI"),
    (r"(?:β|beta)\s*[=:,]\s*(-?\d+(?:\.\d+)?)(?!\s*%)", "Standalone Beta"),
]

# CI patterns (from paper_parser.py lines 146-159)
CI_PATTERNS = [
    (r"(\d+)\s*%?\s*ci[:\s]*\(?(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)?", "% CI: X-Y"),
    (r"(\d+)\s*%?\s*confidence\s+interval[,:\s]+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", "% confidence interval"),
    (r"confidence\s+interval[,:\s]+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", "CI without %"),
    (r"confidence\s+interval\s+of\s+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", "CI of X to Y"),
    (r"\bci[:\s=]+(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", "Bare CI notation"),
    (r"(?:ci|confidence\s+interval)[,:\s]*\((\d+(?:\.\d+)?)\s*(?:[-,]|to)\s*(\d+(?:\.\d+)?)\)", "CI parenthetical"),
]

# P-value patterns (from paper_parser.py lines 214-219)
PVAL_PATTERNS = [
    (r"[Pp][-\s]?(?:value)?[:\s]*([<>=≤≥])\s*0?\.?(\d+)", "p with operator"),
    (r"[Pp][-\s]?(?:value)?[:\s]*(\d+\.?\d*(?:e-?\d+)?)", "p full decimal"),
]

# Sample size patterns (from paper_parser.py lines 269-278)
SAMPLE_PATTERNS = [
    (r"[Nn]\s*[=:]\s*(\d[\d,]*)", "n = X"),
    (r"sample\s+size\s+(?:of\s+)?(\d[\d,]*)", "sample size of X"),
    (r"(\d[\d,]*)\s*[±]\s*\d[\d,]*\s+(?:adults|patients|participants|subjects|individuals)", "X ± Y participants"),
    (r"(\d[\d,]*)\s+(?:adults|patients|participants|subjects|individuals)", "X participants"),
]



def count_pattern_matches(text: str, patterns: list[tuple[str, str]]) -> list[int]:
    """Count matches for each pattern in the list.

    Args:
        text: Normalized text to search.
        patterns: List of (regex_pattern, description) tuples.

    Returns:
        List of match counts, one per pattern.
    """
    counts = []
    for pattern, _ in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        counts.append(len(matches))
    return counts


def process_pdf(pdf_path: Path, output_dir: Path) -> dict:
    """Process a single PDF and return extraction results.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to save extracted text.

    Returns:
        Dict with counts for each pattern type, or None if processing failed.
    """
    try:
        file_bytes = pdf_path.read_bytes()
        pages = extract_text_from_pdf(file_bytes)
    except Exception as e:
        print(f"  Warning: Could not process {pdf_path.name}: {e}")
        return None

    # Combine all pages for pattern matching
    all_text = ""
    text_output = []

    for page_num, page_text in pages:
        normalized = _normalize_text(page_text)
        all_text += normalized + " "
        text_output.append(f"===== PAGE {page_num} =====")
        text_output.append(normalized)
        text_output.append("")

    # Save extracted text
    txt_path = output_dir / f"{pdf_path.stem}.txt"
    txt_path.write_text("\n".join(text_output))

    # Run extraction functions for totals
    em_results = []
    ci_results = []
    pval_results = []
    sample_results = []
    beta_results = []
    md_results = []
    sd_results = []

    for page_num, page_text in pages:
        normalized = _normalize_text(page_text)
        em_results.extend(find_effect_measures(normalized, page_num))
        ci_results.extend(find_confidence_intervals(normalized, page_num))
        pval_results.extend(find_p_values(normalized, page_num))
        sample_results.extend(find_sample_sizes(normalized, page_num))
        beta_results.extend(find_beta_coefficients(normalized, page_num))
        md_results.extend(find_mean_differences(normalized, page_num))
        sd_results.extend(find_standard_deviations(normalized, page_num))

    # Count per-pattern matches (on combined text)
    em_pattern_counts = count_pattern_matches(all_text, EFFECT_PATTERNS)
    ci_pattern_counts = count_pattern_matches(all_text, CI_PATTERNS)
    pval_pattern_counts = count_pattern_matches(all_text, PVAL_PATTERNS)
    sample_pattern_counts = count_pattern_matches(all_text, SAMPLE_PATTERNS)

    return {
        "filename": pdf_path.name,
        "pages": len(pages),
        "em_total": len(em_results),
        "em_patterns": em_pattern_counts,
        "ci_total": len(ci_results),
        "ci_patterns": ci_pattern_counts,
        "pval_total": len(pval_results),
        "pval_patterns": pval_pattern_counts,
        "sample_total": len(sample_results),
        "sample_patterns": sample_pattern_counts,
        "beta_total": len(beta_results),
        "md_total": len(md_results),
        "sd_total": len(sd_results),
    }


def write_csv_report(results: list[dict], output_path: Path) -> None:
    """Write extraction results to CSV.

    Args:
        results: List of result dicts from process_pdf.
        output_path: Path to output CSV file.
    """
    # Build fieldnames dynamically based on number of patterns
    # em_p1-p6: OR, em_p7-p9: HR, em_p10-p12: RR, em_p13-p15: PR,
    # em_p16-p18: IRR, em_p19-p20: Beta
    em_fields = [f"em_p{i}" for i in range(1, len(EFFECT_PATTERNS) + 1)]
    ci_fields = [f"ci_p{i}" for i in range(1, len(CI_PATTERNS) + 1)]
    pval_fields = [f"pval_p{i}" for i in range(1, len(PVAL_PATTERNS) + 1)]
    sample_fields = [f"sample_p{i}" for i in range(1, len(SAMPLE_PATTERNS) + 1)]

    fieldnames = (
        ["filename", "pages", "em_total"] + em_fields +
        ["ci_total"] + ci_fields +
        ["pval_total"] + pval_fields +
        ["sample_total"] + sample_fields
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "filename": r["filename"],
                "pages": r["pages"],
                "em_total": r["em_total"],
                "ci_total": r["ci_total"],
                "pval_total": r["pval_total"],
                "sample_total": r["sample_total"],
            }
            # Add per-pattern counts
            for i, count in enumerate(r["em_patterns"], 1):
                row[f"em_p{i}"] = count
            for i, count in enumerate(r["ci_patterns"], 1):
                row[f"ci_p{i}"] = count
            for i, count in enumerate(r["pval_patterns"], 1):
                row[f"pval_p{i}"] = count
            for i, count in enumerate(r["sample_patterns"], 1):
                row[f"sample_p{i}"] = count

            writer.writerow(row)


def write_summary_report(results: list[dict], output_path: Path) -> None:
    """Write extraction summary report.

    Args:
        results: List of result dicts from process_pdf.
        output_path: Path to output text file.
    """
    lines = [
        "=" * 70,
        "EXTRACTION SUMMARY REPORT",
        "Effect measures: OR, HR, RR, PR, IRR | Also: Beta, MD, SD/SE",
        "=" * 70,
        "",
    ]

    # Totals
    total_em = sum(r["em_total"] for r in results)
    total_ci = sum(r["ci_total"] for r in results)
    total_pval = sum(r["pval_total"] for r in results)
    total_sample = sum(r["sample_total"] for r in results)
    total_beta = sum(r.get("beta_total", 0) for r in results)
    total_md = sum(r.get("md_total", 0) for r in results)
    total_sd = sum(r.get("sd_total", 0) for r in results)

    lines.append("TOTALS ACROSS ALL FILES:")
    lines.append(f"  Effect Measures: {total_em}")
    lines.append(f"  Beta Coefficients: {total_beta}")
    lines.append(f"  Confidence Intervals: {total_ci}")
    lines.append(f"  P-values: {total_pval}")
    lines.append(f"  Mean Differences: {total_md}")
    lines.append(f"  SD/SE: {total_sd}")
    lines.append(f"  Sample Sizes: {total_sample}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("PER-FILE BREAKDOWN")
    lines.append("=" * 70)
    lines.append("")

    for r in results:
        lines.append(f"File: {r['filename']}")
        lines.append(f"  Pages: {r['pages']}")
        lines.append(f"  Effect Measures: {r['em_total']}")
        lines.append(f"  Beta Coefficients: {r.get('beta_total', 0)}")
        lines.append(f"  Confidence Intervals: {r['ci_total']}")
        lines.append(f"  P-values: {r['pval_total']}")
        lines.append(f"  Mean Differences: {r.get('md_total', 0)}")
        lines.append(f"  SD/SE: {r.get('sd_total', 0)}")
        lines.append(f"  Sample Sizes: {r['sample_total']}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("PATTERN LEGEND (for CSV columns)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Effect Measure Patterns (em_p1-em_p20):")
    for i, (_, desc) in enumerate(EFFECT_PATTERNS, 1):
        lines.append(f"  em_p{i}: {desc}")
    lines.append("")

    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose Paper Analyzer extraction patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/diagnose_extraction.py
  python scripts/diagnose_extraction.py ./my_papers
  python scripts/diagnose_extraction.py /path/to/pdfs
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
        print("Create the folder and add PDF files, then run again.")
        sys.exit(1)

    # Find PDFs
    pdfs = list(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{folder}'.")
        print("Add PDF files and run again.")
        sys.exit(0)

    # Create output directory
    extracted_dir = folder / "extracted_text"
    extracted_dir.mkdir(exist_ok=True)

    # Process PDFs
    print(f"Processing {len(pdfs)} PDFs...")
    results = []
    totals = {"em": 0, "ci": 0, "pval": 0, "sample": 0, "beta": 0, "md": 0, "sd": 0}

    for pdf_path in sorted(pdfs):
        result = process_pdf(pdf_path, extracted_dir)
        if result:
            results.append(result)
            totals["em"] += result["em_total"]
            totals["ci"] += result["ci_total"]
            totals["pval"] += result["pval_total"]
            totals["sample"] += result["sample_total"]
            totals["beta"] += result.get("beta_total", 0)
            totals["md"] += result.get("md_total", 0)
            totals["sd"] += result.get("sd_total", 0)

            print(f"  {pdf_path.name}: {result['pages']} pages, "
                  f"{result['em_total']} EM, {result.get('beta_total', 0)} β, "
                  f"{result['ci_total']} CI, {result['pval_total']} p, "
                  f"{result.get('md_total', 0)} MD, {result.get('sd_total', 0)} SD, "
                  f"{result['sample_total']} n")

    if not results:
        print("No PDFs were successfully processed.")
        sys.exit(1)

    # Write reports
    csv_path = folder / "extraction_report.csv"
    summary_path = folder / "extraction_summary.txt"

    write_csv_report(results, csv_path)
    write_summary_report(results, summary_path)

    # Print summary
    print("")
    print("Summary:")
    print(f"  Effect Measures: {totals['em']}, Beta: {totals['beta']}, CIs: {totals['ci']}")
    print(f"  P-values: {totals['pval']}, Mean Diff: {totals['md']}, SD/SE: {totals['sd']}")
    print(f"  Sample Sizes: {totals['sample']}")
    print("")
    print("Reports saved:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
