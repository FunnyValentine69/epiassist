"""Tests for core/paper_parser.py — regex extraction + PDF handling.

Most tests use plain text strings (no PDF needed). PDF extraction is
tested separately using in-memory PDFs built with PyMuPDF.
"""

import fitz  # PyMuPDF

from core.paper_parser import (
    _normalize_text,
    extract_text_from_pdf,
    find_beta_coefficients,
    find_confidence_intervals,
    find_effect_measures,
    find_mean_differences,
    find_p_values,
    find_sample_sizes,
    find_standard_deviations,
    find_weighted_statistics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_bytes(text: str) -> bytes:
    """Create an in-memory single-page PDF containing the given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# === Normalize Text =========================================================


class TestNormalizeText:
    """Tests for _normalize_text."""

    def test_en_dash(self):
        """En-dash should be replaced with hyphen."""
        assert "-" in _normalize_text("1.2\u20132.3")

    def test_em_dash(self):
        """Em-dash should be replaced with hyphen."""
        assert "-" in _normalize_text("1.2\u20142.3")

    def test_unicode_minus(self):
        """Unicode minus (U+2212) should be replaced with hyphen."""
        assert "-" in _normalize_text("1.2\u22122.3")

    def test_soft_hyphen_removed(self):
        """Soft hyphen (U+00AD) should be removed."""
        assert "\u00ad" not in _normalize_text("con\u00adfidence")

    def test_line_break_join(self):
        """Hyphenated line breaks should be joined."""
        result = _normalize_text("confi-\ndence")
        assert "confidence" in result


# === Extract Text from PDF ==================================================


class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf."""

    def test_single_page(self):
        """Should extract text from a single-page PDF."""
        pdf = _make_pdf_bytes("Hello world")
        pages = extract_text_from_pdf(pdf)
        assert len(pages) == 1
        assert "Hello world" in pages[0][1]

    def test_pages_are_1_indexed(self):
        """Page numbers should start at 1."""
        pdf = _make_pdf_bytes("Test page")
        pages = extract_text_from_pdf(pdf)
        assert pages[0][0] == 1

    def test_empty_pdf(self):
        """An empty PDF (no text) should return a page with empty/whitespace text."""
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()
        pages = extract_text_from_pdf(pdf_bytes)
        assert len(pages) == 1
        assert pages[0][1].strip() == ""


# === Find Effect Measures ===================================================


class TestFindEffectMeasures:
    """Tests for find_effect_measures."""

    def test_or_with_ci(self):
        """Should extract OR with 95% CI."""
        text = "OR = 2.5 (95% CI: 1.2-3.8)"
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 2.5 for r in results)

    def test_or_ci_values(self):
        """Should capture CI lower and upper bounds."""
        text = "OR = 2.5 (95% CI: 1.2-3.8)"
        results = find_effect_measures(text)
        or_result = next(r for r in results if r["type"] == "OR")
        assert or_result["ci_lower"] == 1.2
        assert or_result["ci_upper"] == 3.8

    def test_hr_extraction(self):
        """Should extract hazard ratios."""
        text = "HR 1.45 (95% CI: 1.12-1.89)"
        results = find_effect_measures(text)
        assert any(r["type"] == "HR" and r["value"] == 1.45 for r in results)

    def test_rr_extraction(self):
        """Should extract risk ratios."""
        text = "RR = 0.85 (95% CI: 0.72-0.99)"
        results = find_effect_measures(text)
        assert any(r["type"] == "RR" and r["value"] == 0.85 for r in results)

    def test_pr_extraction(self):
        """Should extract prevalence ratios."""
        text = "prevalence ratio = 1.2 (95% CI: 1.1-1.4)"
        results = find_effect_measures(text)
        assert any(r["type"] == "PR" and r["value"] == 1.2 for r in results)

    def test_irr_extraction(self):
        """Should extract incidence rate ratios."""
        text = "IRR 1.5 (95% CI: 1.2-1.9)"
        results = find_effect_measures(text)
        assert any(r["type"] == "IRR" and r["value"] == 1.5 for r in results)

    def test_beta_extraction(self):
        """Should extract beta coefficients via effect measures."""
        text = "beta = 0.45 (95% CI: 0.12-0.78)"
        results = find_effect_measures(text)
        assert any(r["type"] == "\u03b2" and r["value"] == 0.45 for r in results)

    def test_adjusted_detection(self):
        """Should detect adjusted measures (aOR with semicolon CI)."""
        text = "aOR 2.1; 95% CI 1.3-3.2"
        results = find_effect_measures(text)
        adjusted = [r for r in results if r["type"] == "OR"]
        assert len(adjusted) >= 1
        assert any(r.get("adjusted") is True for r in adjusted)

    def test_to_separator(self):
        """Should handle 'to' as CI separator."""
        text = "OR 2.5 (95% CI: 1.2 to 3.8)"
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["ci_lower"] == 1.2 for r in results)

    def test_spelled_out_odds_ratio(self):
        """Should match 'odds ratio' spelled out."""
        text = "odds ratio = 3.1 (95% CI: 1.5-6.4)"
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 3.1 for r in results)

    def test_no_duplicates(self):
        """Same measure with same CI mentioned twice should not duplicate."""
        text = "OR = 2.5 (95% CI: 1.2-3.8). The OR: 2.5 (95% CI: 1.2-3.8)."
        results = find_effect_measures(text)
        # Dedup is by (type, value, ci_lower, ci_upper, page) — should be 1 with CI
        or_with_ci = [
            r for r in results
            if r["type"] == "OR" and r["value"] == 2.5 and r["ci_lower"] == 1.2
        ]
        assert len(or_with_ci) == 1

    def test_standalone_or(self):
        """Should extract standalone OR without CI."""
        text = "OR = 2.5 was reported."
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 2.5 for r in results)

    def test_confidence_interval_spelled_out(self):
        """Should handle '95% confidence interval' spelled out."""
        text = "OR 2.2 (95% confidence interval, 1.4 to 3.4)"
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 2.2 for r in results)

    def test_aor_pattern(self):
        """Should extract aOR (adjusted odds ratio)."""
        text = "aOR 2.5; 95% CI 1.6-3.9"
        results = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 2.5 for r in results)

    def test_parenthetical_ci(self):
        """Should extract OR with parenthetical CI (no CI label)."""
        text = "OR 2.5 (1.2-3.8)"
        results = find_effect_measures(text)
        assert any(
            r["type"] == "OR" and r["ci_lower"] == 1.2 and r["ci_upper"] == 3.8
            for r in results
        )


# === Find Confidence Intervals ==============================================


class TestFindConfidenceIntervals:
    """Tests for find_confidence_intervals."""

    def test_standard_ci(self):
        """Should extract 95% CI: 1.2-3.8."""
        text = "95% CI: 1.2-3.8"
        results = find_confidence_intervals(text)
        assert len(results) >= 1
        assert results[0]["lower"] == 1.2
        assert results[0]["upper"] == 3.8

    def test_to_separator(self):
        """Should handle 'to' as separator."""
        text = "95% CI: 1.2 to 3.8"
        results = find_confidence_intervals(text)
        assert len(results) >= 1
        assert results[0]["lower"] == 1.2

    def test_negative_values(self):
        """Should handle negative CI bounds."""
        text = "95% CI: -0.32-0.45"
        results = find_confidence_intervals(text)
        assert any(r["lower"] == -0.32 for r in results)

    def test_bracket_notation(self):
        """Should extract CI in bracket notation: [1.2, 3.4]."""
        text = "CI [1.2, 3.4]"
        results = find_confidence_intervals(text)
        assert any(r["lower"] == 1.2 and r["upper"] == 3.4 for r in results)

    def test_year_filtering(self):
        """Should skip year-like ranges (>100)."""
        text = "95% CI: 1990-2020"
        results = find_confidence_intervals(text)
        assert not any(r["lower"] == 1990.0 for r in results)

    def test_ci_level_detected(self):
        """Should detect the CI level (e.g. 95)."""
        text = "95% CI: 1.2-3.8"
        results = find_confidence_intervals(text)
        assert results[0]["level"] == 95

    def test_confidence_interval_spelled_out(self):
        """Should handle 'confidence interval' spelled out."""
        text = "95% confidence interval, 1.4 to 3.4"
        results = find_confidence_intervals(text)
        assert len(results) >= 1
        assert results[0]["level"] == 95
        assert results[0]["lower"] == 1.4
        assert results[0]["upper"] == 3.4


# === Find P-Values ==========================================================


class TestFindPValues:
    """Tests for find_p_values."""

    def test_p_less_than(self):
        """Should extract p < 0.05."""
        text = "p < 0.05"
        results = find_p_values(text)
        assert any(r["value"] == 0.05 and r["operator"] == "<" for r in results)

    def test_p_equals(self):
        """Should extract p = 0.001."""
        text = "p = 0.001"
        results = find_p_values(text)
        assert any(r["value"] == 0.001 for r in results)

    def test_p_value_with_colon(self):
        """Should extract P-value: 0.03."""
        text = "P-value: 0.03"
        results = find_p_values(text)
        assert any(abs(r["value"] - 0.03) < 0.001 for r in results)

    def test_p_without_leading_zero(self):
        """Should extract p<.001."""
        text = "p<.001"
        results = find_p_values(text)
        assert any(r["value"] == 0.001 for r in results)

    def test_uppercase_p(self):
        """Should handle uppercase P."""
        text = "P = 0.042"
        results = find_p_values(text)
        assert any(abs(r["value"] - 0.042) < 0.001 for r in results)


# === Find Sample Sizes ======================================================


class TestFindSampleSizes:
    """Tests for find_sample_sizes."""

    def test_n_equals(self):
        """Should extract n = 500."""
        text = "n = 500"
        results = find_sample_sizes(text)
        assert any(r["value"] == 500 for r in results)

    def test_comma_stripping(self):
        """Should handle comma-separated numbers like 933,921."""
        text = "N = 933,921"
        results = find_sample_sizes(text)
        assert any(r["value"] == 933921 for r in results)

    def test_participants_keyword(self):
        """Should extract '1500 participants'."""
        text = "A total of 1500 participants were enrolled."
        results = find_sample_sizes(text)
        assert any(r["value"] == 1500 for r in results)

    def test_below_threshold_filtered(self):
        """Values below 50 should be filtered out."""
        text = "n = 10"
        results = find_sample_sizes(text)
        assert not any(r["value"] == 10 for r in results)

    def test_sorted_descending(self):
        """Results should be sorted by value descending."""
        text = "n = 100. n = 500. n = 200."
        results = find_sample_sizes(text)
        values = [r["value"] for r in results]
        assert values == sorted(values, reverse=True)


# === Find Beta Coefficients =================================================


class TestFindBetaCoefficients:
    """Tests for find_beta_coefficients."""

    def test_beta_with_ci(self):
        """Should extract beta with CI."""
        text = "\u03b2 = 2.18 (95% CI: 0.30-4.01)"
        results = find_beta_coefficients(text)
        assert any(r["value"] == 2.18 and r["ci_lower"] == 0.30 for r in results)

    def test_beta_with_se(self):
        """Should extract B with SE."""
        text = "B = 1.23 (SE = 0.45)"
        results = find_beta_coefficients(text)
        assert any(r["value"] == 1.23 and r["se"] == 0.45 for r in results)

    def test_standalone_beta(self):
        """Should extract standalone beta."""
        text = "beta = -0.52"
        results = find_beta_coefficients(text)
        assert any(r["value"] == -0.52 for r in results)

    def test_coefficient_synonym(self):
        """Should extract 'coefficient = 0.45'."""
        text = "coefficient = 0.45 (95% CI: 0.12-0.78)"
        results = find_beta_coefficients(text)
        assert any(r["value"] == 0.45 for r in results)


# === Find Mean Differences ==================================================


class TestFindMeanDifferences:
    """Tests for find_mean_differences."""

    def test_md_with_ci(self):
        """Should extract MD with CI."""
        text = "mean difference = 2.5 (95% CI: 1.2-3.8)"
        results = find_mean_differences(text)
        assert any(r["value"] == 2.5 and r["ci_lower"] == 1.2 for r in results)

    def test_standalone_md(self):
        """Should extract standalone MD."""
        text = "MD = 3.2"
        results = find_mean_differences(text)
        assert any(r["value"] == 3.2 for r in results)

    def test_negative_md(self):
        """Should handle negative mean differences."""
        text = "mean difference = -1.5"
        results = find_mean_differences(text)
        assert any(r["value"] == -1.5 for r in results)


# === Find Standard Deviations ===============================================


class TestFindStandardDeviations:
    """Tests for find_standard_deviations."""

    def test_mean_sd_parenthetical(self):
        """Should extract mean (SD) format: 3.17 (SD 1.19)."""
        text = "3.17 (SD 1.19)"
        results = find_standard_deviations(text)
        assert any(r["mean"] == 3.17 and r["value"] == 1.19 for r in results)

    def test_plus_minus_notation(self):
        """Should extract plus-minus notation: 3.17 \u00b1 1.19."""
        text = "3.17 \u00b1 1.19"
        results = find_standard_deviations(text)
        assert any(r["mean"] == 3.17 and r["value"] == 1.19 for r in results)

    def test_standalone_sd(self):
        """Should extract standalone SD = 2.5."""
        text = "SD = 2.5"
        results = find_standard_deviations(text)
        assert any(r["value"] == 2.5 and r["type"] == "SD" for r in results)

    def test_se_detected(self):
        """Should extract SE values."""
        text = "SE = 0.45"
        results = find_standard_deviations(text)
        assert any(r["value"] == 0.45 and r["type"] == "SE" for r in results)


# === Find Weighted Statistics ===============================================


class TestFindWeightedStatistics:
    """Tests for find_weighted_statistics."""

    def test_weighted_prevalence(self):
        """Should extract weighted prevalence."""
        text = "weighted prevalence: 25.3%"
        results = find_weighted_statistics(text)
        assert any(r["stat_type"] == "prevalence" and r["value"] == 25.3 for r in results)

    def test_ipw_estimate(self):
        """Should extract IPW estimate."""
        text = "IPW estimate: 1.45"
        results = find_weighted_statistics(text)
        assert any(r["stat_type"] == "IPW" for r in results)

    def test_ps_weighted(self):
        """Should extract PS-weighted OR."""
        text = "PS-weighted OR: 2.1"
        results = find_weighted_statistics(text)
        assert any(r["stat_type"] == "PS-weighted" for r in results)

    def test_survey_weighted(self):
        """Should extract survey-weighted mean."""
        text = "survey-weighted mean: 3.5"
        results = find_weighted_statistics(text)
        assert any(r["stat_type"] == "mean" for r in results)
