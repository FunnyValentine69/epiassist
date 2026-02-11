"""Tests for core.meta_analysis module."""

import math

import pytest

from core.meta_analysis import (
    _calculate_se_from_ci,
    _prepare_studies,
    fixed_effect_meta,
    heterogeneity_stats,
    random_effects_meta,
    run_meta_analysis,
    validate_studies,
)


# --- Sample data ---

OR_STUDIES = [
    {"name": "Smith 2020", "effect": 1.50, "ci_lower": 1.10, "ci_upper": 2.10},
    {"name": "Jones 2021", "effect": 2.00, "ci_lower": 1.30, "ci_upper": 3.10},
    {"name": "Lee 2022", "effect": 1.80, "ci_lower": 1.20, "ci_upper": 2.70},
]

MD_STUDIES = [
    {"name": "Trial A", "effect": 2.5, "ci_lower": 1.0, "ci_upper": 4.0},
    {"name": "Trial B", "effect": 3.0, "ci_lower": 1.5, "ci_upper": 4.5},
    {"name": "Trial C", "effect": 1.8, "ci_lower": 0.5, "ci_upper": 3.1},
]

IDENTICAL_STUDIES = [
    {"name": "Study A", "effect": 2.0, "ci_lower": 1.5, "ci_upper": 2.7},
    {"name": "Study B", "effect": 2.0, "ci_lower": 1.5, "ci_upper": 2.7},
    {"name": "Study C", "effect": 2.0, "ci_lower": 1.5, "ci_upper": 2.7},
]


# --- TestValidateStudies ---


class TestValidateStudies:
    def test_valid_studies(self):
        errors = validate_studies(OR_STUDIES)
        assert errors == []

    def test_too_few_studies(self):
        errors = validate_studies([OR_STUDIES[0]])
        assert len(errors) == 1
        assert "At least 2" in errors[0]

    def test_empty_list(self):
        errors = validate_studies([])
        assert len(errors) == 1

    def test_missing_effect(self):
        bad = [
            {"name": "A", "effect": None, "ci_lower": 1.0, "ci_upper": 2.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("Effect estimate" in e for e in errors)

    def test_missing_ci(self):
        bad = [
            {"name": "A", "effect": 1.5, "ci_lower": None, "ci_upper": 2.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("CI bounds" in e for e in errors)

    def test_ci_lower_greater_than_upper(self):
        bad = [
            {"name": "A", "effect": 1.5, "ci_lower": 3.0, "ci_upper": 1.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("lower must be less" in e for e in errors)

    def test_effect_outside_ci(self):
        bad = [
            {"name": "A", "effect": 5.0, "ci_lower": 1.0, "ci_upper": 2.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("within CI" in e for e in errors)

    def test_non_numeric_effect(self):
        bad = [
            {"name": "A", "effect": "abc", "ci_lower": 1.0, "ci_upper": 2.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("must be a number" in e for e in errors)

    def test_non_numeric_ci(self):
        bad = [
            {"name": "A", "effect": 1.5, "ci_lower": "low", "ci_upper": 2.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        errors = validate_studies(bad)
        assert any("must be numbers" in e for e in errors)


# --- TestCalculateSeFromCi ---


class TestCalculateSeFromCi:
    def test_natural_scale(self):
        se = _calculate_se_from_ci(1.0, 3.0, is_log_scale=False)
        expected = (3.0 - 1.0) / (2 * 1.96)
        assert abs(se - expected) < 1e-10

    def test_log_scale(self):
        se = _calculate_se_from_ci(1.5, 3.0, is_log_scale=True)
        expected = (math.log(3.0) - math.log(1.5)) / (2 * 1.96)
        assert abs(se - expected) < 1e-10

    def test_narrow_ci_gives_small_se(self):
        se_narrow = _calculate_se_from_ci(1.9, 2.1, is_log_scale=False)
        se_wide = _calculate_se_from_ci(1.0, 3.0, is_log_scale=False)
        assert se_narrow < se_wide

    def test_equal_bounds_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            _calculate_se_from_ci(2.0, 2.0, is_log_scale=False)

    def test_log_scale_non_positive_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _calculate_se_from_ci(0.0, 2.0, is_log_scale=True)

    def test_log_scale_negative_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            _calculate_se_from_ci(-1.0, 2.0, is_log_scale=True)


# --- TestPrepareStudies ---


class TestPrepareStudies:
    def test_ratio_measure_log_transform(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        for p, s in zip(prepared, OR_STUDIES):
            assert abs(p["theta"] - math.log(s["effect"])) < 1e-10
            assert p["effect"] == s["effect"]
            assert p["se"] > 0
            assert p["variance"] > 0
            assert p["weight"] > 0

    def test_difference_measure_no_transform(self):
        prepared = _prepare_studies(MD_STUDIES, "MD")
        for p, s in zip(prepared, MD_STUDIES):
            assert p["theta"] == s["effect"]

    def test_preserves_name(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        assert prepared[0]["name"] == "Smith 2020"


# --- TestFixedEffectMeta ---


class TestFixedEffectMeta:
    def test_weights_sum_to_100(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = fixed_effect_meta(prepared, "OR")
        assert abs(sum(result["weights"]) - 100.0) < 0.01

    def test_identical_studies_yield_same_estimate(self):
        prepared = _prepare_studies(IDENTICAL_STUDIES, "OR")
        result = fixed_effect_meta(prepared, "OR")
        assert abs(result["value"] - 2.0) < 0.01

    def test_pooled_within_study_range(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = fixed_effect_meta(prepared, "OR")
        effects = [s["effect"] for s in OR_STUDIES]
        assert min(effects) <= result["value"] <= max(effects)

    def test_ci_contains_pooled(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = fixed_effect_meta(prepared, "OR")
        assert result["ci_lower"] < result["value"] < result["ci_upper"]

    def test_has_interpretation(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = fixed_effect_meta(prepared, "OR")
        assert "Pooled" in result["interpretation"]

    def test_difference_measure(self):
        prepared = _prepare_studies(MD_STUDIES, "MD")
        result = fixed_effect_meta(prepared, "MD")
        assert result["ci_lower"] < result["value"] < result["ci_upper"]


# --- TestRandomEffectsMeta ---


class TestRandomEffectsMeta:
    def test_collapses_to_fixed_when_no_heterogeneity(self):
        """Identical studies → tau²=0 → random should equal fixed."""
        prepared = _prepare_studies(IDENTICAL_STUDIES, "OR")
        fixed = fixed_effect_meta(prepared, "OR")
        random = random_effects_meta(prepared, 0.0, "OR")
        assert abs(fixed["value"] - random["value"]) < 0.01

    def test_prediction_interval_wider_than_ci(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        result = random_effects_meta(prepared, het["tau_squared"], "OR")
        pi = result["prediction_interval"]
        # PI should be at least as wide as CI
        assert pi[0] <= result["ci_lower"] or abs(pi[0] - result["ci_lower"]) < 0.01
        assert pi[1] >= result["ci_upper"] or abs(pi[1] - result["ci_upper"]) < 0.01

    def test_weights_sum_to_100(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        result = random_effects_meta(prepared, het["tau_squared"], "OR")
        assert abs(sum(result["weights"]) - 100.0) < 0.01

    def test_has_tau_squared(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = random_effects_meta(prepared, 0.05, "OR")
        assert result["tau_squared"] == 0.05

    def test_prediction_interval_fallback_for_two_studies(self):
        """With k=2, prediction interval should equal CI and include a note."""
        prepared = _prepare_studies(OR_STUDIES[:2], "OR")
        result = random_effects_meta(prepared, 0.0, "OR")
        pi = result["prediction_interval"]
        assert abs(pi[0] - result["ci_lower"]) < 0.001
        assert abs(pi[1] - result["ci_upper"]) < 0.001
        assert result["prediction_interval_note"] is not None

    def test_prediction_interval_no_note_for_three_studies(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        result = random_effects_meta(prepared, 0.0, "OR")
        assert result["prediction_interval_note"] is None


# --- TestHeterogeneityStats ---


class TestHeterogeneityStats:
    def test_i_squared_in_range(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        assert 0 <= het["i_squared"] <= 100

    def test_identical_studies_zero_heterogeneity(self):
        prepared = _prepare_studies(IDENTICAL_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        assert het["i_squared"] == 0.0
        assert het["tau_squared"] == 0.0

    def test_q_p_value_in_range(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        assert 0 <= het["q_p_value"] <= 1

    def test_has_interpretation(self):
        prepared = _prepare_studies(OR_STUDIES, "OR")
        het = heterogeneity_stats(prepared)
        assert "I²" in het["interpretation"]


# --- TestRunMetaAnalysis ---


class TestRunMetaAnalysis:
    def test_full_pipeline_both_models(self):
        result = run_meta_analysis(OR_STUDIES, "OR", "both")
        assert "errors" not in result
        assert result["fixed"] is not None
        assert result["random"] is not None
        assert result["heterogeneity"] is not None
        assert result["is_ratio"] is True
        assert result["null_value"] == 1.0

    def test_fixed_only(self):
        result = run_meta_analysis(OR_STUDIES, "OR", "fixed")
        assert result["fixed"] is not None
        assert result["random"] is None

    def test_random_only(self):
        result = run_meta_analysis(OR_STUDIES, "OR", "random")
        assert result["fixed"] is None
        assert result["random"] is not None

    def test_difference_measure(self):
        result = run_meta_analysis(MD_STUDIES, "MD", "both")
        assert "errors" not in result
        assert result["is_ratio"] is False
        assert result["null_value"] == 0.0

    def test_validation_errors_returned(self):
        result = run_meta_analysis([OR_STUDIES[0]], "OR")
        assert "errors" in result

    def test_negative_ratio_rejected(self):
        bad = [
            {"name": "A", "effect": -1.0, "ci_lower": -2.0, "ci_upper": -0.5},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        result = run_meta_analysis(bad, "OR")
        assert "errors" in result

    def test_two_studies(self):
        result = run_meta_analysis(OR_STUDIES[:2], "OR", "both")
        assert "errors" not in result
        assert result["fixed"] is not None

    def test_hr_measure(self):
        hr_studies = [
            {"name": "A", "effect": 1.30, "ci_lower": 1.05, "ci_upper": 1.60},
            {"name": "B", "effect": 1.50, "ci_lower": 1.10, "ci_upper": 2.05},
        ]
        result = run_meta_analysis(hr_studies, "HR", "both")
        assert "errors" not in result
        assert result["is_ratio"] is True

    def test_beta_measure(self):
        beta_studies = [
            {"name": "A", "effect": 0.35, "ci_lower": 0.10, "ci_upper": 0.60},
            {"name": "B", "effect": 0.50, "ci_lower": 0.20, "ci_upper": 0.80},
        ]
        result = run_meta_analysis(beta_studies, "beta", "both")
        assert "errors" not in result
        assert result["is_ratio"] is False

    def test_invalid_model_rejected(self):
        result = run_meta_analysis(OR_STUDIES, "OR", "invalid")
        assert "errors" in result
        assert "Invalid model" in result["errors"][0]

    def test_invalid_measure_type_rejected(self):
        result = run_meta_analysis(OR_STUDIES, "xyz")
        assert "errors" in result
        assert "Unknown measure type" in result["errors"][0]

    def test_zero_ratio_values_rejected(self):
        bad = [
            {"name": "A", "effect": 1.5, "ci_lower": 0.0, "ci_upper": 3.0},
            {"name": "B", "effect": 1.5, "ci_lower": 1.0, "ci_upper": 2.0},
        ]
        result = run_meta_analysis(bad, "OR")
        assert "errors" in result
        assert "positive" in result["errors"][0]

    def test_multiple_ratio_errors_accumulated(self):
        bad = [
            {"name": "A", "effect": 0.0, "ci_lower": 0.0, "ci_upper": 1.0},
            {"name": "B", "effect": -1.0, "ci_lower": -2.0, "ci_upper": -0.5},
        ]
        result = run_meta_analysis(bad, "OR")
        assert "errors" in result
        assert len(result["errors"]) == 2

    def test_two_studies_prediction_interval_note(self):
        result = run_meta_analysis(OR_STUDIES[:2], "OR", "both")
        assert "errors" not in result
        assert result["random"]["prediction_interval_note"] is not None
