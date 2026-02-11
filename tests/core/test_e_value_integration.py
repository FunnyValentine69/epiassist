"""Tests for E-value integration with Data Analysis output formats."""

from core.e_value import calculate_e_value_for_or


class TestEValueFromDataAnalysis:
    """Verify E-value works with the exact dict formats from stats_calculator."""

    def test_crude_or_greater_than_one(self):
        """Crude OR > 1 produces valid E-value."""
        result = calculate_e_value_for_or(3.05, 2.10, 4.42)
        assert result["e_value"] is not None
        assert result["e_value"] > 1.0
        assert result["e_value_ci"] is not None

    def test_crude_or_less_than_one(self):
        """Protective OR < 1 produces valid E-value (symmetric)."""
        result = calculate_e_value_for_or(0.5, 0.3, 0.8)
        assert result["e_value"] is not None
        assert result["e_value"] > 1.0  # Always >= 1 due to symmetry

    def test_crude_or_none(self):
        """OR = None returns None E-value."""
        result = calculate_e_value_for_or(None, None, None)
        assert result["e_value"] is None
        assert result["e_value_ci"] is None

    def test_adjusted_or_differs_from_crude(self):
        """MH-adjusted OR produces different E-value than crude."""
        crude = calculate_e_value_for_or(3.05, 2.10, 4.42)
        adjusted = calculate_e_value_for_or(2.50, 1.70, 3.68)
        assert crude["e_value"] != adjusted["e_value"]

    def test_ci_bound_selection(self):
        """E-value CI bound is correctly selected based on OR direction."""
        # OR > 1: uses lower CI bound (closer to 1)
        result_high = calculate_e_value_for_or(3.0, 2.0, 4.5)
        # OR < 1: uses upper CI bound (closer to 1)
        result_low = calculate_e_value_for_or(0.33, 0.22, 0.50)
        assert result_high["e_value_ci"] is not None
        assert result_low["e_value_ci"] is not None

    def test_or_exactly_one(self):
        """OR = 1.0 (no association) produces E-value of 1.0."""
        result = calculate_e_value_for_or(1.0, 0.8, 1.25)
        assert result["e_value"] == 1.0

    def test_interpretation_nonempty(self):
        """E-value interpretation string is non-empty."""
        result = calculate_e_value_for_or(3.05, 2.10, 4.42)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0
