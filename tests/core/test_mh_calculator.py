"""Tests for Mantel-Haenszel stratified analysis."""

import pytest

from core.stats_calculator import calculate_mantel_haenszel
from utils.interpretations import interpret_mantel_haenszel


# --- Textbook example: Simpson's paradox ---
# Crude: OR ~ 1.0, but within strata the association is strong
STRATUM_YOUNG = {"a": 30, "b": 70, "c": 10, "d": 90}
STRATUM_OLD = {"a": 60, "b": 40, "c": 30, "d": 70}

# Heterogeneous strata (OR differs substantially — effect modification)
STRATUM_HIGH = {"a": 60, "b": 40, "c": 10, "d": 90}  # OR = 13.5
STRATUM_LOW = {"a": 15, "b": 85, "c": 40, "d": 60}  # OR = 0.265


class TestCalculateMantelHaenszel:
    """Tests for calculate_mantel_haenszel."""

    def test_basic_two_strata(self):
        """Pooled OR should be computed from two valid strata."""
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        assert result["n_strata"] == 2
        assert result["or_value"] > 0
        assert result["or_ci_lower"] < result["or_value"] < result["or_ci_upper"]

    def test_textbook_confounding(self):
        """Adjusted OR should differ from crude OR when confounding exists."""
        # Crude table (summed):
        # a=90, b=110, c=40, d=160 → crude OR = (90*160)/(110*40) = 3.27
        crude_or = (90 * 160) / (110 * 40)
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        # Adjusted OR should differ from crude
        assert abs(result["or_value"] - crude_or) > 0.01

    def test_homogeneous_strata(self):
        """Breslow-Day p > 0.05 for strata with similar ORs."""
        # Two strata with similar ORs
        s1 = {"a": 40, "b": 60, "c": 20, "d": 80}  # OR = 2.67
        s2 = {"a": 35, "b": 65, "c": 15, "d": 85}  # OR = 2.85
        result = calculate_mantel_haenszel([s1, s2])
        assert result["homogeneity_p_value"] > 0.05

    def test_heterogeneous_strata(self):
        """Breslow-Day p < 0.05 for strata with very different ORs."""
        result = calculate_mantel_haenszel([STRATUM_HIGH, STRATUM_LOW])
        assert result["homogeneity_p_value"] < 0.05

    def test_single_valid_stratum(self):
        """Single stratum should work but homogeneity fields should be None."""
        # One valid + one invalid (zero margin)
        invalid = {"a": 0, "b": 0, "c": 5, "d": 10}
        result = calculate_mantel_haenszel([STRATUM_YOUNG, invalid])
        assert result["n_strata"] == 1
        assert result["homogeneity_statistic"] is None
        assert result["homogeneity_p_value"] is None
        assert result["or_value"] > 0

    def test_all_strata_invalid_raises(self):
        """All strata with zero margins should raise ValueError."""
        invalid1 = {"a": 0, "b": 0, "c": 5, "d": 10}
        invalid2 = {"a": 3, "b": 7, "c": 0, "d": 0}
        with pytest.raises(ValueError, match="No valid strata"):
            calculate_mantel_haenszel([invalid1, invalid2])

    def test_zero_margin_stratum_skipped(self):
        """Strata with zero row margins should be skipped, not crash."""
        invalid = {"a": 0, "b": 0, "c": 5, "d": 10}
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD, invalid])
        assert result["n_strata"] == 2

    def test_rr_pooled_present(self):
        """Pooled RR should be present and reasonable."""
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        assert result["rr_value"] is not None
        assert result["rr_value"] > 0

    def test_rr_ci_present(self):
        """RR confidence interval should be computed."""
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        assert result["rr_ci_lower"] is not None
        assert result["rr_ci_upper"] is not None
        assert result["rr_ci_lower"] < result["rr_value"] < result["rr_ci_upper"]

    def test_n_strata_matches_valid(self):
        """n_strata should count only valid strata."""
        invalid = {"a": 0, "b": 0, "c": 5, "d": 10}
        s3 = {"a": 20, "b": 30, "c": 10, "d": 40}
        result = calculate_mantel_haenszel([STRATUM_YOUNG, invalid, s3])
        assert result["n_strata"] == 2

    def test_interpretation_present(self):
        """Result should include an interpretation string."""
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 20

    def test_mh_test_fields(self):
        """MH test statistic and p-value should be present."""
        result = calculate_mantel_haenszel([STRATUM_YOUNG, STRATUM_OLD])
        assert result["mh_test_statistic"] is not None
        assert 0 <= result["mh_p_value"] <= 1


class TestInterpretMantelHaenszel:
    """Tests for interpret_mantel_haenszel."""

    def test_or_above_one_significant(self):
        """OR > 1 with CI excluding 1 should say 'higher' and 'significant'."""
        text = interpret_mantel_haenszel(2.5, 1.3, 4.8, 0.45, 3, "age")
        assert "higher" in text
        assert "significant" in text.lower()
        assert "excludes" in text

    def test_or_below_one_significant(self):
        """OR < 1 with CI excluding 1 should say 'lower' and 'significant'."""
        text = interpret_mantel_haenszel(0.4, 0.2, 0.8, 0.50, 2, "sex")
        assert "lower" in text
        assert "significant" in text.lower()

    def test_ci_crosses_one_not_significant(self):
        """CI crossing 1.0 should say 'NOT statistically significant'."""
        text = interpret_mantel_haenszel(1.2, 0.8, 1.8, 0.60, 2, "race")
        assert "NOT" in text

    def test_homogeneity_warning(self):
        """Breslow-Day p < 0.05 should warn about effect modification."""
        text = interpret_mantel_haenszel(2.0, 1.1, 3.6, 0.01, 3, "education")
        assert "effect modification" in text.lower() or "varies" in text.lower()

    def test_single_stratum_no_homogeneity(self):
        """Single stratum (homogeneity_p=None) should not mention Breslow-Day."""
        text = interpret_mantel_haenszel(1.8, 1.1, 2.9, None, 1, "age")
        assert "Breslow-Day" not in text
