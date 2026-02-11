"""Tests for direct standardization calculator."""

import pytest

from core.direct_standardization import (
    calculate_stratum_rates,
    calculate_direct_standardized_rate,
)
from utils.interpretations import interpret_direct_standardized_rate


# ── Test data ──────────────────────────────────────────────────────────────
# 3 age groups with hand-calculable values:
# Rates: 10/1000=0.01, 30/2000=0.015, 50/5000=0.01
# Weighted events: 0.01*3000=30, 0.015*5000=75, 0.01*2000=20 → total=125
# Total weight: 3000+5000+2000=10000
# Adjusted rate: (125/10000)*100000 = 1250.0
# Crude: (90/8000)*100000 = 1125.0
BASIC_STRATA = [
    {"stratum_name": "Young", "events": 10, "population": 1000, "standard_weight": 3000},
    {"stratum_name": "Middle", "events": 30, "population": 2000, "standard_weight": 5000},
    {"stratum_name": "Old", "events": 50, "population": 5000, "standard_weight": 2000},
]


class TestCalculateStratumRates:
    """Tests for calculate_stratum_rates."""

    def test_basic_rate(self):
        """Rate = events / population for each stratum."""
        result = calculate_stratum_rates(BASIC_STRATA)
        assert result[0]["rate"] == pytest.approx(0.01)
        assert result[1]["rate"] == pytest.approx(0.015)
        assert result[2]["rate"] == pytest.approx(0.01)

    def test_weighted_events(self):
        """Weighted events = rate * standard_weight."""
        result = calculate_stratum_rates(BASIC_STRATA)
        assert result[0]["weighted_events"] == pytest.approx(30.0)
        assert result[1]["weighted_events"] == pytest.approx(75.0)
        assert result[2]["weighted_events"] == pytest.approx(20.0)

    def test_empty_strata_raises(self):
        """Empty strata list should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_stratum_rates([])

    def test_negative_events_raises(self):
        """Negative events should raise ValueError."""
        strata = [{"stratum_name": "A", "events": -1, "population": 100, "standard_weight": 50}]
        with pytest.raises(ValueError, match="Events must be non-negative"):
            calculate_stratum_rates(strata)

    def test_zero_population_raises(self):
        """Zero population should raise ValueError."""
        strata = [{"stratum_name": "A", "events": 5, "population": 0, "standard_weight": 50}]
        with pytest.raises(ValueError, match="Population must be positive"):
            calculate_stratum_rates(strata)

    def test_negative_weight_raises(self):
        """Negative standard weight should raise ValueError."""
        strata = [{"stratum_name": "A", "events": 5, "population": 100, "standard_weight": -10}]
        with pytest.raises(ValueError, match="Standard weight must be non-negative"):
            calculate_stratum_rates(strata)


class TestCalculateDirectStandardizedRate:
    """Tests for calculate_direct_standardized_rate."""

    def test_basic_adjusted_rate(self):
        """Adjusted rate matches hand calculation: (125/10000)*100000 = 1250."""
        result = calculate_direct_standardized_rate(BASIC_STRATA)
        assert result["value"] == pytest.approx(1250.0, rel=1e-3)

    def test_crude_vs_adjusted_differ(self):
        """Crude and adjusted rates differ when age distributions differ."""
        result = calculate_direct_standardized_rate(BASIC_STRATA)
        assert result["crude_rate"] != result["value"]

    def test_ci_contains_point_estimate(self):
        """CI should bracket the point estimate."""
        result = calculate_direct_standardized_rate(BASIC_STRATA)
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_multiplier_scaling(self):
        """Per 1,000 rate should be 100x smaller than per 100,000."""
        r_100k = calculate_direct_standardized_rate(BASIC_STRATA, multiplier=100_000)
        r_1k = calculate_direct_standardized_rate(BASIC_STRATA, multiplier=1_000)
        assert r_1k["value"] == pytest.approx(r_100k["value"] / 100, rel=1e-3)

    def test_zero_events(self):
        """Zero events everywhere → rate = 0, CI = [0, 0]."""
        strata = [
            {"stratum_name": "A", "events": 0, "population": 1000, "standard_weight": 5000},
            {"stratum_name": "B", "events": 0, "population": 2000, "standard_weight": 5000},
        ]
        result = calculate_direct_standardized_rate(strata)
        assert result["value"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] == 0.0

    def test_single_stratum(self):
        """Single stratum produces valid result."""
        strata = [{"stratum_name": "All", "events": 20, "population": 1000, "standard_weight": 10000}]
        result = calculate_direct_standardized_rate(strata)
        # Rate = (20/1000) * 100000 = 2000
        assert result["value"] == pytest.approx(2000.0, rel=1e-3)
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_strata_details_length(self):
        """Strata details length matches input."""
        result = calculate_direct_standardized_rate(BASIC_STRATA)
        assert len(result["strata_details"]) == len(BASIC_STRATA)

    def test_invalid_ci_level_raises(self):
        """ci_level outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            calculate_direct_standardized_rate(BASIC_STRATA, ci_level=0.0)
        with pytest.raises(ValueError, match="ci_level must be between 0 and 1"):
            calculate_direct_standardized_rate(BASIC_STRATA, ci_level=1.0)

    def test_zero_total_weight_raises(self):
        """All weights zero → ValueError."""
        strata = [
            {"stratum_name": "A", "events": 5, "population": 100, "standard_weight": 0},
            {"stratum_name": "B", "events": 3, "population": 200, "standard_weight": 0},
        ]
        with pytest.raises(ValueError, match="Total standard population weight must be positive"):
            calculate_direct_standardized_rate(strata)

    def test_single_event_boundary(self):
        """Single event across strata produces valid finite CI."""
        strata = [
            {"stratum_name": "A", "events": 1, "population": 10000, "standard_weight": 5000},
            {"stratum_name": "B", "events": 0, "population": 20000, "standard_weight": 5000},
        ]
        result = calculate_direct_standardized_rate(strata)
        assert result["value"] > 0
        assert 0 <= result["ci_lower"] < result["value"] < result["ci_upper"]
        assert result["ci_upper"] < float("inf")

    def test_negative_multiplier_raises(self):
        """Negative or zero multiplier should raise ValueError."""
        with pytest.raises(ValueError, match="Rate multiplier must be a positive"):
            calculate_direct_standardized_rate(BASIC_STRATA, multiplier=0)
        with pytest.raises(ValueError, match="Rate multiplier must be a positive"):
            calculate_direct_standardized_rate(BASIC_STRATA, multiplier=-100)

    def test_all_expected_keys(self):
        """Result dict contains all expected keys."""
        result = calculate_direct_standardized_rate(BASIC_STRATA)
        expected_keys = {
            "value", "ci_lower", "ci_upper", "interpretation",
            "strata_details", "total_standard_pop", "total_events",
            "total_population", "crude_rate", "multiplier",
        }
        assert set(result.keys()) == expected_keys


class TestInterpretDirectStandardizedRate:
    """Tests for interpret_direct_standardized_rate."""

    def test_rate_higher_than_crude(self):
        """When adjusted > crude, interpretation says 'higher'."""
        text = interpret_direct_standardized_rate(150.0, 120.0, 180.0, 100.0, 100_000)
        assert "higher" in text

    def test_rate_lower_than_crude(self):
        """When adjusted < crude, interpretation says 'lower'."""
        text = interpret_direct_standardized_rate(80.0, 60.0, 100.0, 120.0, 100_000)
        assert "lower" in text

    def test_rate_similar_to_crude(self):
        """When adjusted ~ crude, interpretation says 'similar'."""
        text = interpret_direct_standardized_rate(100.0, 80.0, 120.0, 101.0, 100_000)
        assert "similar" in text

    def test_zero_crude_rate(self):
        """Zero crude rate handled gracefully (no division by zero)."""
        text = interpret_direct_standardized_rate(0.0, 0.0, 0.0, 0.0, 100_000)
        assert isinstance(text, str)
        assert len(text) > 0
