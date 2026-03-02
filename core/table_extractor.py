"""StatSift table extraction integration for EpiAssist.

Converts StatSift's structured table data (ParsedValue objects) into
EpiAssist's flat-dict result format used by the Paper Analyzer page.

StatSift is an optional dependency — when unavailable, all functions
gracefully return empty results or False.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Effect measure type patterns matched against column header text
_EFFECT_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OR", re.compile(r"\b[ac]?OR\b", re.I)),
    ("HR", re.compile(r"\b[ac]?HR\b", re.I)),
    ("RR", re.compile(r"\b[ac]?RR\b", re.I)),
    ("PR", re.compile(r"\bPR\b", re.I)),
    ("IRR", re.compile(r"\bIRR\b", re.I)),
]


def is_statsift_available() -> bool:
    """Check if statsift is installed and importable."""
    try:
        import statsift

        return hasattr(statsift, "__version__")
    except Exception:
        return False


def _infer_effect_type(header: str) -> str:
    """Infer effect measure type (OR, HR, RR, etc.) from column header text."""
    for label, pattern in _EFFECT_TYPE_PATTERNS:
        if pattern.search(header):
            return label
    return "OR"


def _infer_adjusted(header: str) -> bool | None:
    """Infer adjusted/crude status from column header text."""
    h = header.lower()
    if "crude" in h or "unadjusted" in h:
        return False
    if any(kw in h for kw in ("adjusted", "aor", "ahr", "arr")):
        return True
    return None


def convert_tables_to_results(
    tables: list,
) -> dict[str, list[dict]]:
    """Convert StatSift TableResult objects into EpiAssist's result format.

    Parameters
    ----------
    tables : list[TableResult]
        Parsed tables from statsift's ExtractionResult.

    Returns
    -------
    dict with keys: effect_measures, confidence_intervals, p_values,
    sample_sizes, beta_coefficients, mean_differences, standard_deviations,
    weighted_statistics. Each value is a list of flat dicts matching the
    format used by Paper Analyzer.
    """
    results: dict[str, list[dict]] = {
        "effect_measures": [],
        "confidence_intervals": [],
        "p_values": [],
        "sample_sizes": [],
        "beta_coefficients": [],
        "mean_differences": [],
        "standard_deviations": [],
        "weighted_statistics": [],
    }

    for table in tables:
        if not table.parsed_data:
            continue

        for row in table.parsed_data:
            # Find the label column value for context
            label = ""
            for col_name, pv in row.items():
                if pv.value_type.value == "label":
                    label = pv.raw
                    break

            for col_name, pv in row.items():
                vtype = pv.value_type.value
                vals = pv.values
                page = table.page_number

                context = (
                    f"Table {table.table_index + 1}: {label} — {col_name}"
                    if label
                    else f"Table {table.table_index + 1}: {col_name}"
                )

                if vtype in ("effect_ci", "effect"):
                    results["effect_measures"].append(
                        {
                            "type": _infer_effect_type(col_name),
                            "value": vals.get("effect"),
                            "ci_lower": vals.get("ci_lo"),
                            "ci_upper": vals.get("ci_hi"),
                            "adjusted": _infer_adjusted(col_name),
                            "adjusted_for": None,
                            "context": context,
                            "page": page,
                            "source": "statsift",
                        }
                    )

                elif vtype == "pvalue":
                    p_val = vals.get("pvalue")
                    if p_val is not None:
                        results["p_values"].append(
                            {
                                "value": p_val,
                                "operator": pv.qualifier or "=",
                                "context": context,
                                "page": page,
                                "source": "statsift",
                            }
                        )

                elif vtype == "ci":
                    results["confidence_intervals"].append(
                        {
                            "level": 95,
                            "lower": vals.get("ci_lo"),
                            "upper": vals.get("ci_hi"),
                            "context": context,
                            "page": page,
                            "source": "statsift",
                        }
                    )

                elif vtype == "mean_sd":
                    results["standard_deviations"].append(
                        {
                            "mean": vals.get("mean"),
                            "value": vals.get("sd"),
                            "type": "SD",
                            "context": context,
                            "page": page,
                            "source": "statsift",
                        }
                    )

                elif vtype == "count" and vals.get("count") is not None:
                    count_val = vals["count"]
                    if count_val >= 10:
                        results["sample_sizes"].append(
                            {
                                "value": int(count_val),
                                "page": page,
                                "source": "statsift",
                            }
                        )

    return results


def extract_from_pdf(pdf_bytes: bytes) -> dict[str, list[dict]] | None:
    """Run statsift extraction on PDF bytes and return EpiAssist-format results.

    Returns None if statsift is not available or extraction fails.
    """
    if not is_statsift_available():
        return None

    try:
        import tempfile
        from pathlib import Path

        from statsift.config import ExtractConfig, resolve_device
        from statsift.content import extract_content
        from statsift.converter import convert_pdf
        from statsift.repair import repair_table, repair_tables
        from statsift.schema import build_result
        from statsift.tables import extract_tables

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        config = ExtractConfig()
        resolved_device = resolve_device(config.device)
        conv_result = convert_pdf(tmp_path, config)
        raw_tables = extract_tables(conv_result, pages=config.pages)
        raw_tables = repair_tables(raw_tables)
        raw_tables = [repair_table(t)[0] for t in raw_tables]

        paper_content = extract_content(conv_result)
        num_pages = (
            len(conv_result.document.pages) if hasattr(conv_result.document, "pages") else 0
        )

        result = build_result(
            raw_tables=raw_tables,
            content=paper_content,
            filename="uploaded.pdf",
            num_pages=num_pages,
            processing_time=0.0,
            device_used=resolved_device,
            table_mode=config.table_mode.value,
            ocr_enabled=config.ocr_enabled,
        )

        tmp_path.unlink(missing_ok=True)
        return convert_tables_to_results(result.tables)

    except Exception:
        logger.exception("StatSift extraction failed")
        return None
