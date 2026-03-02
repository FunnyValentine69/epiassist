"""Tests for StatSift table extraction integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from core.table_extractor import (
    _infer_adjusted,
    _infer_effect_type,
    convert_tables_to_results,
)


def _make_parsed_value(
    raw: str,
    value_type: str,
    values: dict | None = None,
    qualifier: str | None = None,
    confidence: float = 0.0,
) -> MagicMock:
    """Create a mock ParsedValue-like object."""
    pv = MagicMock()
    pv.raw = raw
    pv.value_type = MagicMock()
    pv.value_type.value = value_type
    pv.values = values or {}
    pv.qualifier = qualifier
    pv.confidence = confidence
    return pv


def _make_table(
    table_index: int = 0,
    page_number: int = 1,
    headers: list[str] | None = None,
    parsed_data: list[dict] | None = None,
) -> MagicMock:
    """Create a mock TableResult-like object."""
    t = MagicMock()
    t.table_index = table_index
    t.page_number = page_number
    t.headers = headers or []
    t.parsed_data = parsed_data or []
    return t


class TestInferEffectType:
    def test_or(self) -> None:
        assert _infer_effect_type("OR (95% CI)") == "OR"

    def test_adjusted_or(self) -> None:
        assert _infer_effect_type("aOR (95% CI)") == "OR"

    def test_hr(self) -> None:
        assert _infer_effect_type("HR") == "HR"

    def test_rr(self) -> None:
        assert _infer_effect_type("Adjusted RR") == "RR"

    def test_irr(self) -> None:
        assert _infer_effect_type("IRR (95% CI)") == "IRR"

    def test_default_or(self) -> None:
        assert _infer_effect_type("Effect (95% CI)") == "OR"


class TestInferAdjusted:
    def test_adjusted(self) -> None:
        assert _infer_adjusted("Adjusted OR") is True

    def test_aor(self) -> None:
        assert _infer_adjusted("aOR (95% CI)") is True

    def test_crude(self) -> None:
        assert _infer_adjusted("Crude OR") is False

    def test_unadjusted(self) -> None:
        assert _infer_adjusted("Unadjusted HR") is False

    def test_unknown(self) -> None:
        assert _infer_adjusted("OR (95% CI)") is None


class TestConvertTablesToResults:
    def test_effect_ci(self) -> None:
        table = _make_table(
            parsed_data=[
                {
                    "Group": _make_parsed_value("Male", "label"),
                    "OR (95% CI)": _make_parsed_value(
                        "1.5 (1.1-2.0)",
                        "effect_ci",
                        {"effect": 1.5, "ci_lo": 1.1, "ci_hi": 2.0},
                        confidence=1.0,
                    ),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["effect_measures"]) == 1
        em = results["effect_measures"][0]
        assert em["type"] == "OR"
        assert em["value"] == 1.5
        assert em["ci_lower"] == 1.1
        assert em["ci_upper"] == 2.0
        assert em["source"] == "statsift"

    def test_pvalue_with_qualifier(self) -> None:
        table = _make_table(
            parsed_data=[
                {
                    "Group": _make_parsed_value("A", "label"),
                    "P-value": _make_parsed_value(
                        "<0.001", "pvalue", {"pvalue": 0.001}, qualifier="<"
                    ),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["p_values"]) == 1
        pv = results["p_values"][0]
        assert pv["value"] == 0.001
        assert pv["operator"] == "<"

    def test_mean_sd(self) -> None:
        table = _make_table(
            parsed_data=[
                {
                    "Var": _make_parsed_value("Age", "label"),
                    "Mean (SD)": _make_parsed_value(
                        "45.2 (12.3)", "mean_sd", {"mean": 45.2, "sd": 12.3}
                    ),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["standard_deviations"]) == 1
        sd = results["standard_deviations"][0]
        assert sd["mean"] == 45.2
        assert sd["value"] == 12.3
        assert sd["type"] == "SD"

    def test_bare_ci(self) -> None:
        table = _make_table(
            parsed_data=[
                {
                    "Outcome": _make_parsed_value("Death", "label"),
                    "95% CI": _make_parsed_value(
                        "1.1-2.0", "ci", {"ci_lo": 1.1, "ci_hi": 2.0}
                    ),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["confidence_intervals"]) == 1
        ci = results["confidence_intervals"][0]
        assert ci["lower"] == 1.1
        assert ci["upper"] == 2.0

    def test_count_as_sample_size(self) -> None:
        table = _make_table(
            parsed_data=[
                {
                    "Group": _make_parsed_value("Total", "label"),
                    "N": _make_parsed_value("250", "count", {"count": 250.0}),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["sample_sizes"]) == 1
        assert results["sample_sizes"][0]["value"] == 250

    def test_small_count_excluded(self) -> None:
        """Counts below 10 are not sample sizes."""
        table = _make_table(
            parsed_data=[
                {
                    "X": _make_parsed_value("Cat", "label"),
                    "N": _make_parsed_value("3", "count", {"count": 3.0}),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["sample_sizes"]) == 0

    def test_empty_tables(self) -> None:
        results = convert_tables_to_results([])
        for cat in results.values():
            assert cat == []

    def test_no_parsed_data(self) -> None:
        table = _make_table(parsed_data=[])
        results = convert_tables_to_results([table])
        for cat in results.values():
            assert cat == []

    def test_context_includes_label(self) -> None:
        table = _make_table(
            table_index=2,
            parsed_data=[
                {
                    "Group": _make_parsed_value("Female", "label"),
                    "OR (95% CI)": _make_parsed_value(
                        "0.8 (0.5-1.2)",
                        "effect_ci",
                        {"effect": 0.8, "ci_lo": 0.5, "ci_hi": 1.2},
                    ),
                }
            ],
        )
        results = convert_tables_to_results([table])
        em = results["effect_measures"][0]
        assert "Table 3" in em["context"]
        assert "Female" in em["context"]

    def test_ns_pvalue_excluded(self) -> None:
        """NS p-values (pvalue=None) should not be included."""
        table = _make_table(
            parsed_data=[
                {
                    "X": _make_parsed_value("A", "label"),
                    "P": _make_parsed_value("NS", "pvalue", {"pvalue": None}),
                }
            ],
        )
        results = convert_tables_to_results([table])
        assert len(results["p_values"]) == 0
