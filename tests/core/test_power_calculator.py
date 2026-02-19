"""Tests for core/power_calculator.py — sample size, power, power curves, OR sizing."""

import pandas as pd

from core.power_calculator import (
    calculate_power,
    calculate_sample_size,
    calculate_sample_size_for_or,
    classify_effect_size,
    effect_size_from_proportions,
    generate_power_curve,
)


# === Sample Size ============================================================


class TestCalculateSampleSize:
    """Tests for calculate_sample_size."""

    def test_medium_effect_default_params(self):
        """Medium effect (h=0.5) at 80% power should need ~63/group."""
        n = calculate_sample_size(0.5)
        assert 60 <= n <= 70

    def test_larger_effect_needs_fewer(self):
        """Larger effect size should require smaller sample."""
        n_small = calculate_sample_size(0.3)
        n_large = calculate_sample_size(0.8)
        assert n_large < n_small

    def test_lower_alpha_needs_more(self):
        """More stringent alpha should require larger sample."""
        n_005 = calculate_sample_size(0.5, alpha=0.05)
        n_001 = calculate_sample_size(0.5, alpha=0.01)
        assert n_001 > n_005

    def test_higher_power_needs_more(self):
        """Higher power should require larger sample."""
        n_80 = calculate_sample_size(0.5, power=0.80)
        n_90 = calculate_sample_size(0.5, power=0.90)
        assert n_90 > n_80

    def test_returns_integer(self):
        """Sample size should be a whole number (ceiling)."""
        n = calculate_sample_size(0.5)
        assert isinstance(n, int)

    def test_small_effect_large_n(self):
        """Small effect (h=0.2) should need many more subjects."""
        n = calculate_sample_size(0.2)
        assert n > 300


# === Power ==================================================================


class TestCalculatePower:
    """Tests for calculate_power."""

    def test_known_power_roundtrip(self):
        """Power at the sample size from calculate_sample_size should be ~0.80."""
        n = calculate_sample_size(0.5)
        power = calculate_power(n, 0.5)
        assert abs(power - 0.80) < 0.02

    def test_large_n_high_power(self):
        """Large n with medium effect should give power near 1."""
        power = calculate_power(1000, 0.5)
        assert power > 0.99

    def test_small_n_low_power(self):
        """Very small n should give low power."""
        power = calculate_power(5, 0.3)
        assert power < 0.20

    def test_zero_n_returns_zero(self):
        """n=0 should return 0 power."""
        assert calculate_power(0, 0.5) == 0.0

    def test_zero_effect_returns_zero(self):
        """Zero effect size should return 0 power."""
        assert calculate_power(100, 0) == 0.0

    def test_power_bounded(self):
        """Power should always be in [0, 1]."""
        power = calculate_power(500, 0.8)
        assert 0 <= power <= 1


# === Power Curve ============================================================


class TestGeneratePowerCurve:
    """Tests for generate_power_curve."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        df = generate_power_curve(0.5)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        """DataFrame should have 'n' and 'power' columns."""
        df = generate_power_curve(0.5)
        assert "n" in df.columns
        assert "power" in df.columns

    def test_monotonically_increasing(self):
        """Power should increase (or stay equal) as n increases."""
        df = generate_power_curve(0.5)
        power_vals = df["power"].tolist()
        for i in range(1, len(power_vals)):
            assert power_vals[i] >= power_vals[i - 1] - 1e-9

    def test_custom_range(self):
        """Custom n_range should be respected."""
        df = generate_power_curve(0.5, n_range=(50, 200))
        assert df["n"].min() >= 50
        assert df["n"].max() <= 200


# === Sample Size for OR =====================================================


class TestCalculateSampleSizeForOr:
    """Tests for calculate_sample_size_for_or."""

    def test_basic_calculation(self):
        """Should return positive integer sample sizes."""
        result = calculate_sample_size_for_or(0.1, 2.0)
        assert result["n_exposed"] > 0
        assert result["n_unexposed"] > 0
        assert result["n_total"] == result["n_exposed"] + result["n_unexposed"]

    def test_larger_or_needs_fewer(self):
        """Larger OR (bigger effect) should need fewer subjects."""
        small_or = calculate_sample_size_for_or(0.1, 1.5)
        large_or = calculate_sample_size_for_or(0.1, 3.0)
        assert large_or["n_total"] < small_or["n_total"]

    def test_unequal_ratio(self):
        """ratio=2 should give n_unexposed ≈ 2 * n_exposed."""
        result = calculate_sample_size_for_or(0.1, 2.0, ratio=2.0)
        assert abs(result["n_unexposed"] / result["n_exposed"] - 2.0) < 0.1

    def test_or_one_returns_none(self):
        """OR=1 (no effect) should return None for sample sizes."""
        result = calculate_sample_size_for_or(0.1, 1.0)
        assert result["n_exposed"] is None

    def test_interpretation_present(self):
        """Result should include an interpretation string."""
        result = calculate_sample_size_for_or(0.1, 2.0)
        assert "interpretation" in result
        assert "OR" in result["interpretation"]


# === Effect Size Helpers ====================================================


class TestEffectSizeHelpers:
    """Tests for effect_size_from_proportions and classify_effect_size."""

    def test_equal_proportions_zero(self):
        """Equal proportions should give effect size of 0."""
        h = effect_size_from_proportions(0.3, 0.3)
        assert abs(h) < 1e-10

    def test_symmetry(self):
        """Effect size should be the same regardless of group order."""
        h1 = effect_size_from_proportions(0.3, 0.5)
        h2 = effect_size_from_proportions(0.5, 0.3)
        assert abs(h1 - h2) < 1e-10

    def test_classify_negligible(self):
        """h < 0.2 should be classified as negligible."""
        assert classify_effect_size(0.1) == "negligible"

    def test_classify_small(self):
        """0.2 <= h < 0.5 should be classified as small."""
        assert classify_effect_size(0.3) == "small"

    def test_classify_medium(self):
        """0.5 <= h < 0.8 should be classified as medium."""
        assert classify_effect_size(0.6) == "medium"

    def test_classify_large(self):
        """h >= 0.8 should be classified as large."""
        assert classify_effect_size(1.0) == "large"
