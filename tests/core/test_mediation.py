"""Tests for core/mediation.py — Baron-Kenny mediation analysis."""

import numpy as np
import pandas as pd
import pytest

from core.mediation import (
    _prepare_mediation_data,
    bootstrap_mediation_ci,
    calculate_mediation_effects,
    fit_mediation_models,
    run_mediation_analysis,
)
from utils.interpretations import interpret_mediation


# ---------------------------------------------------------------------------
# Test data factory
# ---------------------------------------------------------------------------


def _make_mediation_df(n: int = 500) -> pd.DataFrame:
    """Generate a DataFrame with known mediation structure.

    DGP: exposure → mediator (a=0.5) and exposure → outcome (direct c'=0.3)
         mediator → outcome (b=0.4), age confounds all three.
    """
    rng = np.random.default_rng(42)
    age = rng.normal(50, 10, n)
    exposure_latent = rng.normal(0, 1, n) + 0.3 * (age - 50)
    exposure = (exposure_latent > 0).astype(int)
    mediator = 0.5 * exposure + 0.2 * age + rng.normal(0, 1, n)
    outcome_cont = 0.3 * exposure + 0.4 * mediator + 0.1 * age + rng.normal(0, 1, n)
    # Binary outcome via logistic transform
    logit_y = -2 + 0.5 * exposure + 0.3 * mediator + 0.02 * age
    prob_y = 1 / (1 + np.exp(-logit_y))
    outcome_bin = (rng.uniform(0, 1, n) < prob_y).astype(int)
    outcome_bin_str = np.where(outcome_bin == 1, "Yes", "No")

    return pd.DataFrame({
        "exposure": np.where(exposure == 1, "Exposed", "Unexposed"),
        "mediator": mediator,
        "outcome": outcome_cont,
        "outcome_binary": outcome_bin_str,
        "age": age,
        "sex": rng.choice(["M", "F"], n),
        "weight_col": rng.uniform(0.5, 5.0, n),
    })


DF = _make_mediation_df()


# -- TestPrepareMediationData ------------------------------------------------


class TestPrepareMediationData:
    """Tests for _prepare_mediation_data."""

    def test_nan_rows_dropped_across_all_vars(self):
        """NaN in any relevant column drops the row for all models."""
        df = DF.copy()
        df.loc[0, "mediator"] = np.nan
        df.loc[1, "outcome"] = np.nan
        df.loc[2, "age"] = np.nan
        outcome, exposure, mediator, conf_X, _, n_obs, n_dropped = (
            _prepare_mediation_data(
                df, "outcome", "exposure", "mediator", ["age"],
                exposure_positive="Exposed",
            )
        )
        assert n_dropped == 3
        assert n_obs == len(df) - 3
        # All outputs have same length
        assert len(outcome) == n_obs
        assert len(exposure) == n_obs
        assert len(mediator) == n_obs
        assert len(conf_X) == n_obs

    def test_exposure_binarized(self):
        """Categorical exposure is binarized to {0, 1}."""
        _, exposure, _, _, _, _, _ = _prepare_mediation_data(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        assert set(exposure.unique()) == {0, 1}

    def test_outcome_binarized_when_requested(self):
        """Outcome is binarized when binarize_outcome=True."""
        outcome, _, _, _, _, _, _ = _prepare_mediation_data(
            DF, "outcome_binary", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", binarize_outcome=True,
            outcome_positive="Yes",
        )
        assert set(outcome.unique()) == {0, 1}

    def test_categorical_confounders_get_dummies(self):
        """Categorical confounders are dummy-encoded with drop_first."""
        _, _, _, conf_X, _, _, _ = _prepare_mediation_data(
            DF, "outcome", "exposure", "mediator", ["age", "sex"],
            exposure_positive="Exposed",
        )
        # sex has 2 levels → 1 dummy; age is continuous → 1 column
        # conf_X should have exactly 2 columns
        assert conf_X.shape[1] == 2

    def test_weight_col_returned_aligned(self):
        """Weight column is returned aligned with y."""
        _, _, _, _, weights, n_obs, _ = _prepare_mediation_data(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", weight_col="weight_col",
        )
        assert weights is not None
        assert len(weights) == n_obs
        assert (weights > 0).all()

    def test_no_weight_col_returns_none(self):
        """No weight column returns None."""
        _, _, _, _, weights, _, _ = _prepare_mediation_data(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        assert weights is None

    def test_too_few_observations_raises(self):
        """Fewer than 10 observations raises ValueError."""
        df_small = DF.head(5).copy()
        # Make most rows NaN to trigger the check
        df_small.loc[:3, "mediator"] = np.nan
        with pytest.raises(ValueError, match="At least 10"):
            _prepare_mediation_data(
                df_small, "outcome", "exposure", "mediator", ["age"],
                exposure_positive="Exposed",
            )


# -- TestFitMediationModels --------------------------------------------------


class TestFitMediationModels:
    """Tests for fit_mediation_models."""

    def test_all_three_models_present(self):
        """Result contains total, a_path, and direct model keys."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        assert "total" in result["models"]
        assert "a_path" in result["models"]
        assert "direct" in result["models"]

    def test_coefficients_are_finite(self):
        """All coefficients from converged models are finite."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        models = result["models"]
        assert np.isfinite(models["total"]["coef_c"])
        assert np.isfinite(models["a_path"]["coef_a"])
        assert np.isfinite(models["direct"]["coef_c_prime"])
        assert np.isfinite(models["direct"]["coef_b"])

    def test_convergence_flags(self):
        """All models report convergence."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        for key in ["total", "a_path", "direct"]:
            assert result["models"][key]["converged"] is True

    def test_confounders_included(self):
        """Models work with multiple confounders."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", ["age", "sex"],
            exposure_positive="Exposed",
        )
        assert result["models"]["total"]["converged"] is True

    def test_no_confounders(self):
        """Models work with no confounders."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", [],
            exposure_positive="Exposed",
        )
        assert result["models"]["total"]["converged"] is True

    def test_n_observations_tracked(self):
        """n_observations and n_dropped are reported."""
        result = fit_mediation_models(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed",
        )
        assert result["n_observations"] > 0
        assert result["n_dropped"] >= 0
        assert result["n_observations"] + result["n_dropped"] == len(DF)

    def test_binary_outcome(self):
        """Models fit with binary outcome (logistic)."""
        result = fit_mediation_models(
            DF, "outcome_binary", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", outcome_type="binary",
            binarize_outcome=True, outcome_positive="Yes",
        )
        assert result["models"]["total"]["converged"] is True


# -- TestCalculateMediationEffects -------------------------------------------


class TestCalculateMediationEffects:
    """Tests for calculate_mediation_effects."""

    def test_product_method_continuous(self):
        """Indirect = a*b for continuous outcomes."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="continuous",
        )
        assert effects["method"] == "product"
        assert abs(effects["indirect"] - 0.2) < 1e-10

    def test_difference_method_binary(self):
        """Indirect = c - c' for binary outcomes."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="binary",
        )
        assert effects["method"] == "difference"
        assert abs(effects["indirect"] - 0.2) < 1e-10

    def test_sobel_se_formula(self):
        """Sobel SE = sqrt(a^2 * se_b^2 + b^2 * se_a^2)."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="continuous",
        )
        expected_se = np.sqrt(0.5**2 * 0.1**2 + 0.4**2 * 0.1**2)
        assert abs(effects["sobel_se"] - expected_se) < 1e-10

    def test_sobel_z_and_p(self):
        """Sobel z and p are computed for continuous outcomes."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="continuous",
        )
        assert effects["sobel_z"] is not None
        assert effects["sobel_p"] is not None
        assert 0 <= effects["sobel_p"] <= 1

    def test_no_sobel_for_binary(self):
        """Sobel test is None for binary outcomes."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="binary",
        )
        assert effects["sobel_se"] is None
        assert effects["sobel_z"] is None
        assert effects["sobel_p"] is None

    def test_proportion_mediated_computed(self):
        """Proportion mediated is indirect / total when same sign."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="continuous",
        )
        assert effects["proportion_mediated"] is not None
        assert 0 <= effects["proportion_mediated"] <= 1
        expected = 0.2 / 0.5
        assert abs(effects["proportion_mediated"] - expected) < 1e-10

    def test_proportion_mediated_none_when_signs_differ(self):
        """Proportion mediated is None when indirect and total have opposite signs."""
        effects = calculate_mediation_effects(
            a=0.5, b=-0.4, c=0.1, c_prime=0.3,
            se_a=0.1, se_b=0.1, outcome_type="continuous",
        )
        # indirect = 0.5 * -0.4 = -0.2, total = 0.1 → different signs
        assert effects["proportion_mediated"] is None

    def test_direct_equals_c_prime(self):
        """Direct effect equals c'."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1,
        )
        assert effects["direct"] == 0.3

    def test_total_equals_c(self):
        """Total effect equals c."""
        effects = calculate_mediation_effects(
            a=0.5, b=0.4, c=0.5, c_prime=0.3,
            se_a=0.1, se_b=0.1,
        )
        assert effects["total"] == 0.5


# -- TestBootstrapMediationCI -----------------------------------------------


class TestBootstrapMediationCI:
    """Tests for bootstrap_mediation_ci."""

    def test_cis_returned(self):
        """Bootstrap returns indirect, direct, and total CIs."""
        ci = bootstrap_mediation_ci(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", n_boot=50,
        )
        assert "indirect_ci" in ci
        assert "direct_ci" in ci
        assert "total_ci" in ci

    def test_cis_are_finite(self):
        """CI bounds are finite."""
        ci = bootstrap_mediation_ci(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", n_boot=50,
        )
        for key in ["indirect_ci", "direct_ci", "total_ci"]:
            lower, upper = ci[key]
            assert np.isfinite(lower)
            assert np.isfinite(upper)

    def test_lower_less_than_upper(self):
        """Lower CI bound < upper CI bound."""
        ci = bootstrap_mediation_ci(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", n_boot=50,
        )
        for key in ["indirect_ci", "direct_ci", "total_ci"]:
            lower, upper = ci[key]
            assert lower < upper

    def test_no_confounders(self):
        """Bootstrap works with no confounders."""
        ci = bootstrap_mediation_ci(
            DF, "outcome", "exposure", "mediator", [],
            exposure_positive="Exposed", n_boot=50,
        )
        assert np.isfinite(ci["indirect_ci"][0])


# -- TestRunMediationAnalysis ------------------------------------------------


class TestRunMediationAnalysis:
    """Tests for the full run_mediation_analysis pipeline."""

    def test_full_pipeline_continuous(self):
        """Full pipeline with continuous outcome produces all expected keys."""
        result = run_mediation_analysis(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", n_boot=50,
        )
        assert "models" in result
        assert "effects" in result
        assert "ci" in result
        assert "n_observations" in result
        assert "n_boot" in result
        assert "interpretation" in result
        assert result["effects"]["method"] == "product"

    def test_full_pipeline_binary(self):
        """Full pipeline with binary outcome uses difference method."""
        result = run_mediation_analysis(
            DF, "outcome_binary", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", outcome_type="binary",
            binarize_outcome=True, outcome_positive="Yes", n_boot=50,
        )
        assert result["effects"]["method"] == "difference"
        assert result["effects"]["sobel_p"] is None

    def test_survey_weights(self):
        """Full pipeline works with survey weights."""
        result = run_mediation_analysis(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", weight_col="weight_col", n_boot=50,
        )
        assert result["n_observations"] > 0
        assert "survey-weighted" in result["interpretation"].lower()

    def test_no_confounders(self):
        """Full pipeline works with no confounders."""
        result = run_mediation_analysis(
            DF, "outcome", "exposure", "mediator", [],
            exposure_positive="Exposed", n_boot=50,
        )
        assert result["models"]["total"]["converged"] is True

    def test_interpretation_nonempty(self):
        """Interpretation is a non-empty string."""
        result = run_mediation_analysis(
            DF, "outcome", "exposure", "mediator", ["age"],
            exposure_positive="Exposed", n_boot=50,
        )
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 50


# -- TestInterpretMediation --------------------------------------------------


def _make_interpretation(**kwargs) -> str:
    """Helper with sensible defaults for interpret_mediation."""
    defaults = {
        "mediator_name": "mediator",
        "exposure_name": "exposure",
        "outcome_name": "outcome",
        "indirect": 0.2,
        "direct": 0.3,
        "total": 0.5,
        "indirect_ci": (0.05, 0.35),
        "direct_ci": (0.10, 0.50),
        "sobel_p": 0.01,
        "proportion_mediated": 0.4,
        "method": "product",
        "n_obs": 500,
        "confounder_names": ["age"],
        "weighted": False,
    }
    defaults.update(kwargs)
    return interpret_mediation(**defaults)


class TestInterpretMediation:
    """Tests for interpret_mediation."""

    def test_mentions_mediator(self):
        """Interpretation mentions the mediator name."""
        text = _make_interpretation(mediator_name="smoking")
        assert "smoking" in text

    def test_mentions_exposure_and_outcome(self):
        """Interpretation mentions exposure and outcome names."""
        text = _make_interpretation(
            exposure_name="treatment", outcome_name="mortality",
        )
        assert "treatment" in text
        assert "mortality" in text

    def test_significant_indirect_says_mediation(self):
        """Significant indirect effect mentions mediation."""
        text = _make_interpretation(
            indirect_ci=(0.05, 0.35),  # excludes 0
        )
        assert "mediation" in text.lower()

    def test_nonsignificant_indirect_says_no_mediation(self):
        """Non-significant indirect CI spanning 0 says no evidence."""
        text = _make_interpretation(
            indirect_ci=(-0.1, 0.3),  # spans 0
        )
        assert "no evidence" in text.lower()

    def test_mentions_proportion_mediated(self):
        """When proportion is available and significant, it's mentioned."""
        text = _make_interpretation(
            proportion_mediated=0.4, indirect_ci=(0.05, 0.35),
        )
        assert "40.0%" in text

    def test_confounder_names_in_text(self):
        """Confounder names appear in interpretation."""
        text = _make_interpretation(confounder_names=["age", "sex"])
        assert "age" in text
        assert "sex" in text

    def test_survey_weighted_prefix(self):
        """Weighted analysis mentions survey-weighted."""
        text = _make_interpretation(weighted=True)
        assert "Survey-weighted" in text

    def test_difference_method_noted(self):
        """Binary outcome method is noted in interpretation."""
        text = _make_interpretation(
            method="difference", sobel_p=None,
        )
        assert "difference method" in text.lower()

    def test_partial_mediation_classification(self):
        """Both significant → partial mediation."""
        text = _make_interpretation(
            indirect_ci=(0.05, 0.35),  # excludes 0 → indirect sig
            direct_ci=(0.10, 0.50),    # excludes 0 → direct sig
        )
        assert "partial" in text.lower()

    def test_full_mediation_classification(self):
        """Significant indirect + non-significant direct → full mediation."""
        text = _make_interpretation(
            indirect_ci=(0.05, 0.35),  # excludes 0 → indirect sig
            direct_ci=(-0.05, 0.50),   # spans 0 → direct not sig
        )
        assert "full mediation" in text.lower()
