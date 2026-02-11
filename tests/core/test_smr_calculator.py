"""Tests for SMR/SIR calculator."""

import pytest

from core.smr_calculator import (
    calculate_smr,
    calculate_expected_events,
    calculate_smr_stratified,
)
from utils.interpretations import interpret_smr


# ── Test data ──────────────────────────────────────────────────────────────
BASIC_STRATA = [
    {"stratum_name": "20-39", "person_time": 5000, "reference_rate": 0.001, "observed": 8},
    {"stratum_name": "40-59", "person_time": 8000, "reference_rate": 0.004, "observed": 40},
    {"stratum_name": "60-79", "person_time": 3000, "reference_rate": 0.012, "observed": 42},
]


class TestCalculateSmr:
    """Tests for calculate_smr."""

    def test_smr_greater_than_one(self):
        """SMR > 1 when observed exceeds expected."""
        result = calculate_smr(observed=45, expected=30.0)
        assert result["value"] == 1.5
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_smr_less_than_one(self):
        """SMR < 1 when observed is below expected."""
        result = calculate_smr(observed=15, expected=30.0)
        assert result["value"] == 0.5
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_smr_equals_one(self):
        """SMR = 1.0 when observed equals expected."""
        result = calculate_smr(observed=30, expected=30.0)
        assert result["value"] == 1.0

    def test_zero_observed(self):
        """SMR = 0 with valid upper CI when no events observed."""
        result = calculate_smr(observed=0, expected=10.0)
        assert result["value"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] > 0.0

    def test_small_observed(self):
        """CI is valid for small observed counts (< 5)."""
        result = calculate_smr(observed=3, expected=5.0)
        assert result["ci_lower"] >= 0
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_large_observed_narrow_ci(self):
        """Large counts produce narrower CI."""
        small = calculate_smr(observed=10, expected=10.0)
        large = calculate_smr(observed=1000, expected=1000.0)
        small_width = small["ci_upper"] - small["ci_lower"]
        large_width = large["ci_upper"] - large["ci_lower"]
        assert large_width < small_width

    def test_zero_expected_raises(self):
        """Expected = 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Expected count must be positive"):
            calculate_smr(observed=5, expected=0.0)

    def test_negative_observed_raises(self):
        """Negative observed should raise ValueError."""
        with pytest.raises(ValueError, match="Observed count must be non-negative"):
            calculate_smr(observed=-1, expected=10.0)

    def test_negative_expected_raises(self):
        """Negative expected should raise ValueError."""
        with pytest.raises(ValueError, match="Expected count must be positive"):
            calculate_smr(observed=5, expected=-10.0)

    def test_invalid_ci_level_raises(self):
        """ci_level outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            calculate_smr(observed=10, expected=10.0, ci_level=0.0)
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            calculate_smr(observed=10, expected=10.0, ci_level=1.0)
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            calculate_smr(observed=10, expected=10.0, ci_level=1.5)

    def test_interpretation_present(self):
        """Result includes a non-empty interpretation string."""
        result = calculate_smr(observed=45, expected=30.0)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestCalculateExpectedEvents:
    """Tests for calculate_expected_events."""

    def test_basic_expected(self):
        """Expected = sum(rate * person_time) for each stratum."""
        result = calculate_expected_events(BASIC_STRATA)
        # 5000*0.001 + 8000*0.004 + 3000*0.012 = 5 + 32 + 36 = 73
        assert result["expected"] == 73.0
        assert result["total_observed"] == 90
        assert len(result["strata_details"]) == 3

    def test_empty_strata_raises(self):
        """Empty strata list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_expected_events([])

    def test_negative_person_time_raises(self):
        """Negative person-time should raise ValueError."""
        strata = [{"stratum_name": "A", "person_time": -100, "reference_rate": 0.01, "observed": 5}]
        with pytest.raises(ValueError, match="Person-time must be non-negative"):
            calculate_expected_events(strata)

    def test_negative_reference_rate_raises(self):
        """Negative reference rate should raise ValueError."""
        strata = [{"stratum_name": "A", "person_time": 100, "reference_rate": -0.01, "observed": 5}]
        with pytest.raises(ValueError, match="Reference rate must be non-negative"):
            calculate_expected_events(strata)

    def test_negative_observed_in_stratum_raises(self):
        """Negative observed count in a stratum should raise ValueError."""
        strata = [{"stratum_name": "A", "person_time": 100, "reference_rate": 0.01, "observed": -5}]
        with pytest.raises(ValueError, match="Observed count must be non-negative"):
            calculate_expected_events(strata)


class TestCalculateSmrStratified:
    """Tests for calculate_smr_stratified."""

    def test_full_pipeline(self):
        """Stratified SMR computes valid result with strata details."""
        result = calculate_smr_stratified(BASIC_STRATA)
        assert result["value"] > 0
        assert result["ci_lower"] < result["value"] < result["ci_upper"]
        assert "strata_details" in result
        assert "total_person_time" in result

    def test_strata_details_length(self):
        """Strata details length matches input."""
        result = calculate_smr_stratified(BASIC_STRATA)
        assert len(result["strata_details"]) == len(BASIC_STRATA)

    def test_zero_total_expected_raises(self):
        """All reference rates zero → total expected=0 → ValueError."""
        strata = [
            {"stratum_name": "A", "person_time": 1000, "reference_rate": 0.0, "observed": 5},
            {"stratum_name": "B", "person_time": 2000, "reference_rate": 0.0, "observed": 3},
        ]
        with pytest.raises(ValueError, match="Total expected events must be positive"):
            calculate_smr_stratified(strata)


class TestInterpretSmr:
    """Tests for interpret_smr."""

    def test_significant_higher(self):
        """SMR > 1 with CI excluding 1.0 → 'higher' and 'significant'."""
        text = interpret_smr(1.5, 1.1, 2.0)
        assert "higher" in text
        assert "statistically significant" in text
        assert "NOT" not in text

    def test_significant_lower(self):
        """SMR < 1 with CI excluding 1.0 → 'lower' and 'significant'."""
        text = interpret_smr(0.5, 0.3, 0.9)
        assert "lower" in text
        assert "statistically significant" in text
        assert "NOT" not in text

    def test_not_significant(self):
        """CI crossing 1.0 → 'NOT statistically significant'."""
        text = interpret_smr(1.2, 0.8, 1.8)
        assert "NOT statistically significant" in text
