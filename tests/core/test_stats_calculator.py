"""Tests for core/stats_calculator.py — OR, RR, RD, Chi-square, CI, NNT.

The existing test_mh_calculator.py covers calculate_mantel_haenszel.
This file covers all other public functions in stats_calculator.
"""

from core.stats_calculator import (
    calculate_chi_square,
    calculate_confidence_interval,
    calculate_nnt,
    calculate_odds_ratio,
    calculate_risk_difference,
    calculate_risk_ratio,
)

# ---------------------------------------------------------------------------
# Shared test data: hand-calculable 2x2 table
#
#                 Outcome+  Outcome-
#   Exposed          20        80     (n1=100)
#   Unexposed        10        90     (n0=100)
#
# OR  = (20*90)/(80*10) = 2.25
# RR  = (20/100)/(10/100) = 2.0
# RD  = 0.20 - 0.10 = 0.10
# ---------------------------------------------------------------------------
A, B, C, D = 20, 80, 10, 90


# === Odds Ratio ============================================================


class TestCalculateOddsRatio:
    """Tests for calculate_odds_ratio."""

    def test_known_or_value(self):
        """OR for reference table should be 2.25."""
        result = calculate_odds_ratio(A, B, C, D)
        assert result["value"] == 2.25

    def test_ci_brackets_estimate(self):
        """CI should bracket the point estimate."""
        result = calculate_odds_ratio(A, B, C, D)
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_strong_or_ci_excludes_one(self):
        """A large-effect table (OR=6.0) should have CI entirely above 1."""
        result = calculate_odds_ratio(40, 60, 10, 90)
        assert result["ci_lower"] > 1.0

    def test_zero_cell_haldane_correction(self):
        """A zero cell should apply Haldane correction (+0.5 to all cells)."""
        result = calculate_odds_ratio(0, 50, 10, 40)
        # Haldane: (0.5*40.5)/(50.5*10.5) ≈ 0.038
        assert result["value"] == 0.038
        assert result["ci_lower"] > 0

    def test_null_association(self):
        """Equal cell counts should give OR close to 1."""
        result = calculate_odds_ratio(50, 50, 50, 50)
        assert result["value"] == 1.0

    def test_result_has_p_value(self):
        """Result should include a p-value from chi-square."""
        result = calculate_odds_ratio(A, B, C, D)
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_result_has_se(self):
        """Result should include standard error of log(OR)."""
        result = calculate_odds_ratio(A, B, C, D)
        assert "se" in result
        assert result["se"] > 0


# === Risk Ratio =============================================================


class TestCalculateRiskRatio:
    """Tests for calculate_risk_ratio."""

    def test_known_rr_value(self):
        """RR for reference table should be 2.0."""
        result = calculate_risk_ratio(A, B, C, D)
        assert result["value"] == 2.0

    def test_ci_brackets_estimate(self):
        """CI should bracket the point estimate."""
        result = calculate_risk_ratio(A, B, C, D)
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_zero_denominator_exposed(self):
        """Zero exposed total should return None."""
        result = calculate_risk_ratio(0, 0, 10, 90)
        assert result["value"] is None

    def test_zero_denominator_unexposed(self):
        """Zero unexposed total should return None."""
        result = calculate_risk_ratio(20, 80, 0, 0)
        assert result["value"] is None

    def test_zero_unexposed_risk(self):
        """Zero risk in unexposed group should return None (division by zero)."""
        result = calculate_risk_ratio(20, 80, 0, 100)
        assert result["value"] is None

    def test_protective_rr(self):
        """When exposure is protective, RR should be < 1."""
        result = calculate_risk_ratio(5, 95, 20, 80)
        assert result["value"] < 1.0


# === Risk Difference ========================================================


class TestCalculateRiskDifference:
    """Tests for calculate_risk_difference."""

    def test_known_rd_value(self):
        """RD for reference table should be 0.10."""
        result = calculate_risk_difference(A, B, C, D)
        assert result["value"] == 0.1

    def test_ci_brackets_estimate(self):
        """CI should bracket the point estimate."""
        result = calculate_risk_difference(A, B, C, D)
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_negative_rd(self):
        """When exposure is protective, RD should be negative."""
        result = calculate_risk_difference(5, 95, 20, 80)
        assert result["value"] < 0

    def test_zero_denominator(self):
        """Zero total in a group should return None."""
        result = calculate_risk_difference(0, 0, 10, 90)
        assert result["value"] is None

    def test_zero_rd(self):
        """Equal risks should give RD = 0."""
        result = calculate_risk_difference(10, 90, 10, 90)
        assert result["value"] == 0.0


# === Chi-Square =============================================================


class TestCalculateChiSquare:
    """Tests for calculate_chi_square."""

    def test_significant_table(self):
        """A strong-effect table should be statistically significant."""
        result = calculate_chi_square(40, 60, 10, 90)
        assert result["p_value"] < 0.05

    def test_null_table(self):
        """Equal cells should not be significant."""
        result = calculate_chi_square(50, 50, 50, 50)
        assert result["p_value"] > 0.05
        assert abs(result["value"]) < 0.01

    def test_df_is_one(self):
        """Degrees of freedom for a 2x2 table should be 1."""
        result = calculate_chi_square(A, B, C, D)
        assert result["df"] == 1

    def test_expected_values_returned(self):
        """Expected values array should be returned."""
        result = calculate_chi_square(A, B, C, D)
        assert result["expected"] is not None
        assert result["expected"].shape == (2, 2)

    def test_interpretation_present(self):
        """Result should include an interpretation string."""
        result = calculate_chi_square(A, B, C, D)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


# === Confidence Interval ====================================================


class TestCalculateConfidenceInterval:
    """Tests for calculate_confidence_interval."""

    def test_95ci_width(self):
        """95% CI with SE=1 should span roughly ±1.96."""
        lower, upper = calculate_confidence_interval(0, 1.0, 0.95)
        assert abs(lower - (-1.96)) < 0.01
        assert abs(upper - 1.96) < 0.01

    def test_se_effect(self):
        """Larger SE should produce wider CI."""
        narrow = calculate_confidence_interval(0, 0.5, 0.95)
        wide = calculate_confidence_interval(0, 2.0, 0.95)
        narrow_width = narrow[1] - narrow[0]
        wide_width = wide[1] - wide[0]
        assert wide_width > narrow_width

    def test_custom_level(self):
        """99% CI should be wider than 95% CI."""
        ci_95 = calculate_confidence_interval(5.0, 1.0, 0.95)
        ci_99 = calculate_confidence_interval(5.0, 1.0, 0.99)
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 > width_95


# === NNT ====================================================================


class TestCalculateNnt:
    """Tests for calculate_nnt."""

    def test_nnt_from_positive_rd(self):
        """Positive RD → Number Needed to Harm."""
        result = calculate_nnt(0.1)
        assert result["value"] == 10.0
        assert "Harm" in result["interpretation"]

    def test_nnt_from_negative_rd(self):
        """Negative RD → Number Needed to Treat."""
        result = calculate_nnt(-0.05)
        assert result["value"] == 20.0
        assert "Treat" in result["interpretation"]

    def test_zero_rd(self):
        """Zero RD should return None (undefined NNT)."""
        result = calculate_nnt(0.0)
        assert result["value"] is None
