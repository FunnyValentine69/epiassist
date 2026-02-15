"""Tests for propensity score analysis module."""

import numpy as np
import pandas as pd
import pytest

from core.propensity_score import (
    _prepare_ps_data,
    assess_common_support,
    balance_diagnostics,
    calculate_iptw_weights,
    calculate_smd,
    estimate_propensity_scores,
    estimate_treatment_effect,
    run_propensity_score_analysis,
)
from utils.interpretations import interpret_propensity_score


# ── Test data fixture ─────────────────────────────────────────────────


def _make_ps_df(n: int = 500) -> pd.DataFrame:
    """Create synthetic data with known confounding structure.

    Treatment assignment and outcome both depend on age and sex,
    creating confounding that PS analysis should address.
    """
    rng = np.random.default_rng(42)

    age = rng.normal(50, 10, size=n)
    sex = rng.choice(["M", "F"], size=n)
    sex_num = (np.array(sex) == "M").astype(int)

    # Treatment depends on confounders
    logit_t = -1 + 0.03 * age + 0.5 * sex_num
    prob_t = 1 / (1 + np.exp(-logit_t))
    treatment = rng.binomial(1, prob_t)

    # Binary outcome depends on treatment + confounders
    logit_y = -2 + 0.7 * treatment + 0.02 * age + 0.3 * sex_num
    prob_y = 1 / (1 + np.exp(-logit_y))
    outcome = rng.binomial(1, prob_y)

    # Continuous outcome
    bp = 120 + 5 * treatment + 0.3 * age + rng.normal(0, 10, n)

    return pd.DataFrame({
        "treatment": treatment,
        "age": age.round(1),
        "sex": sex,
        "outcome": outcome,
        "blood_pressure": bp.round(1),
    })


DF = _make_ps_df()


# ── TestPreparePSData ─────────────────────────────────────────────────


class TestPreparePSData:
    """Tests for _prepare_ps_data."""

    def test_returns_binary_treatment(self):
        """Treatment is binarized as 0/1."""
        treatment, X, names, w = _prepare_ps_data(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert set(treatment.unique()) <= {0, 1}

    def test_nan_rows_dropped(self):
        """Rows with NaN in relevant columns are dropped."""
        df = DF.copy()
        df.loc[0, "age"] = np.nan
        df.loc[1, "sex"] = np.nan
        treatment, X, names, w = _prepare_ps_data(
            df, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert len(treatment) == len(DF) - 2

    def test_categorical_confounders_encoded(self):
        """Categorical confounders get dummy encoding."""
        treatment, X, names, w = _prepare_ps_data(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        # sex has 2 levels -> 1 dummy (drop_first=True)
        assert any("sex_" in n for n in names)

    def test_constant_added(self):
        """Design matrix includes constant column."""
        treatment, X, names, w = _prepare_ps_data(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert "const" in X.columns

    def test_weight_col_aligned(self):
        """Survey weights returned aligned with other outputs."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(99).uniform(0.5, 5.0, size=len(df))
        treatment, X, names, w = _prepare_ps_data(
            df, "treatment", ["age", "sex"], treatment_positive=1, weight_col="weight"
        )
        assert w is not None
        assert len(w) == len(treatment)

    def test_negative_weights_raise(self):
        """Negative weights raise ValueError."""
        df = DF.copy()
        df["weight"] = -1.0
        with pytest.raises(ValueError, match="positive"):
            _prepare_ps_data(
                df, "treatment", ["age", "sex"], treatment_positive=1, weight_col="weight"
            )

    def test_zero_weights_raise(self):
        """Zero weights raise ValueError."""
        df = DF.copy()
        df["weight"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            _prepare_ps_data(
                df, "treatment", ["age", "sex"], treatment_positive=1, weight_col="weight"
            )

    def test_treatment_positive_none_raises(self):
        """treatment_positive=None raises ValueError."""
        with pytest.raises(ValueError, match="treatment_positive must be specified"):
            _prepare_ps_data(
                DF, "treatment", ["age", "sex"], treatment_positive=None
            )


# ── TestEstimatePropensityScores ──────────────────────────────────────


class TestEstimatePropensityScores:
    """Tests for estimate_propensity_scores."""

    def test_scores_between_0_and_1(self):
        """All propensity scores are in (0, 1)."""
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert (result["ps_scores"] > 0).all()
        assert (result["ps_scores"] < 1).all()

    def test_returns_expected_keys(self):
        """Result dict contains all expected keys."""
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        expected_keys = {
            "ps_scores", "treatment_binary", "n_observations", "n_dropped",
            "converged", "model_coefficients", "survey_weights",
        }
        assert expected_keys == set(result.keys())

    def test_no_confounders_raises(self):
        """Raises ValueError when confounder_cols is empty."""
        with pytest.raises(ValueError, match="At least one confounder"):
            estimate_propensity_scores(
                DF, "treatment", [], treatment_positive=1
            )

    def test_single_treatment_level_raises(self):
        """Raises ValueError when treatment has only one level."""
        df = DF.copy()
        df["treatment"] = 1  # all treated
        with pytest.raises(ValueError, match="only one level"):
            estimate_propensity_scores(
                df, "treatment", ["age", "sex"], treatment_positive=1
            )

    def test_model_converges(self):
        """Model converges on well-behaved data."""
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert result["converged"] is True

    def test_n_observations_matches(self):
        """n_observations matches length of PS scores."""
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert result["n_observations"] == len(result["ps_scores"])

    def test_too_few_observations_raises(self):
        """Raises ValueError when too few observations for parameters."""
        # 5 rows but many confounders -> underdetermined model
        small_df = DF.head(5).copy()
        # Add many numeric confounders to exceed observation count
        rng = np.random.default_rng(0)
        for i in range(10):
            small_df[f"conf_{i}"] = rng.normal(size=5)
        many_confs = [f"conf_{i}" for i in range(10)]
        with pytest.raises(ValueError, match="Not enough observations"):
            estimate_propensity_scores(
                small_df, "treatment", many_confs, treatment_positive=1
            )

    def test_n_dropped_reflects_missing_data(self):
        """n_dropped counts rows removed due to NaN."""
        df = DF.copy()
        df.loc[0, "age"] = np.nan
        df.loc[1, "age"] = np.nan
        result = estimate_propensity_scores(
            df, "treatment", ["age", "sex"], treatment_positive=1
        )
        assert result["n_dropped"] == 2


# ── TestAssessCommonSupport ───────────────────────────────────────────


class TestAssessCommonSupport:
    """Tests for assess_common_support."""

    def test_good_overlap(self):
        """Returns sufficient=True when distributions overlap well."""
        ps_result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        ps = ps_result["ps_scores"]
        t = ps_result["treatment_binary"]
        cs = assess_common_support(ps[t == 1].values, ps[t == 0].values)
        assert cs["sufficient"] is True

    def test_poor_overlap_warning(self):
        """Returns sufficient=False and warning when overlap < 90%."""
        # Distributions that overlap partially but less than 90%
        treated = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])
        control = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45])
        cs = assess_common_support(treated, control)
        assert cs["sufficient"] is False
        assert cs["warning"] is not None
        assert cs["overlap_pct"] > 0  # Has some overlap, not zero

    def test_no_overlap(self):
        """Returns 0% overlap when distributions don't overlap at all."""
        treated = np.array([0.8, 0.9, 0.95])
        control = np.array([0.05, 0.1, 0.15])
        cs = assess_common_support(treated, control)
        assert cs["overlap_pct"] == 0.0

    def test_overlap_pct_in_range(self):
        """overlap_pct is between 0 and 100."""
        ps_result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        ps = ps_result["ps_scores"]
        t = ps_result["treatment_binary"]
        cs = assess_common_support(ps[t == 1].values, ps[t == 0].values)
        assert 0 <= cs["overlap_pct"] <= 100


# ── TestCalculateIPTWWeights ──────────────────────────────────────────


class TestCalculateIPTWWeights:
    """Tests for calculate_iptw_weights."""

    def _get_ps_and_treatment(self):
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        return result["ps_scores"], result["treatment_binary"]

    def test_ate_weights_positive(self):
        """All ATE weights are positive."""
        ps, t = self._get_ps_and_treatment()
        result = calculate_iptw_weights(ps, t, estimand="ATE")
        assert (result["weights"] > 0).all()

    def test_att_weights_treated_equal_one(self):
        """ATT unstabilized weights for treated group are 1.0."""
        ps, t = self._get_ps_and_treatment()
        result = calculate_iptw_weights(ps, t, estimand="ATT", stabilized=False)
        treated_weights = result["weights"][t == 1]
        np.testing.assert_allclose(treated_weights, 1.0)

    def test_stabilized_reduces_variance(self):
        """Stabilized weights have lower variance than unstabilized."""
        ps, t = self._get_ps_and_treatment()
        stab = calculate_iptw_weights(ps, t, estimand="ATE", stabilized=True)
        unstab = calculate_iptw_weights(ps, t, estimand="ATE", stabilized=False)
        assert stab["weight_summary"]["sd"] <= unstab["weight_summary"]["sd"]

    def test_trimming_counts(self):
        """Trimming reports positive number of trimmed observations."""
        ps, t = self._get_ps_and_treatment()
        result = calculate_iptw_weights(ps, t, trim_quantile=0.05)
        # With 5% trimming on 500 observations, some should be trimmed
        assert result["trimmed_n"] > 0

    def test_effective_n_less_than_n(self):
        """Effective sample size <= actual sample size."""
        ps, t = self._get_ps_and_treatment()
        result = calculate_iptw_weights(ps, t)
        assert result["weight_summary"]["effective_n"] <= len(ps)

    def test_invalid_estimand_raises(self):
        """Invalid estimand raises ValueError."""
        ps, t = self._get_ps_and_treatment()
        with pytest.raises(ValueError, match="estimand must be"):
            calculate_iptw_weights(ps, t, estimand="INVALID")


# ── TestCalculateSMD ──────────────────────────────────────────────────


class TestCalculateSMD:
    """Tests for calculate_smd."""

    def test_identical_groups_zero_smd(self):
        """SMD is 0 when groups are identical."""
        vals = pd.Series([1, 2, 3, 4, 5])
        assert calculate_smd(vals, vals) == 0.0

    def test_smd_positive(self):
        """SMD is always >= 0 (absolute value)."""
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        assert calculate_smd(a, b) >= 0
        assert calculate_smd(b, a) >= 0

    def test_smd_symmetric(self):
        """SMD(a, b) == SMD(b, a)."""
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6])
        assert calculate_smd(a, b) == calculate_smd(b, a)

    def test_weighted_smd_differs_from_unweighted(self):
        """Weighted SMD differs from unweighted when weights vary."""
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([3.0, 4.0, 5.0, 6.0])
        w_a = pd.Series([10.0, 1.0, 1.0, 1.0])  # Heavy weight on low value
        w_b = pd.Series([1.0, 1.0, 1.0, 10.0])  # Heavy weight on high value
        smd_unw = calculate_smd(a, b)
        smd_w = calculate_smd(a, b, w_a, w_b)
        assert smd_unw != smd_w

    def test_constant_series_zero_smd(self):
        """SMD is 0 when both groups are constant (pooled SD = 0)."""
        a = pd.Series([5, 5, 5])
        b = pd.Series([5, 5, 5])
        assert calculate_smd(a, b) == 0.0

    def test_unequal_group_sizes(self):
        """SMD handles unequal group sizes."""
        a = pd.Series([1, 2, 3])
        b = pd.Series([4, 5, 6, 7, 8])
        smd = calculate_smd(a, b)
        assert np.isfinite(smd)
        assert smd > 0


# ── TestBalanceDiagnostics ────────────────────────────────────────────


class TestBalanceDiagnostics:
    """Tests for balance_diagnostics."""

    def _get_aligned_data(self):
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        iptw = calculate_iptw_weights(result["ps_scores"], result["treatment_binary"])
        aligned_df = DF.loc[result["treatment_binary"].index]
        return aligned_df, result["treatment_binary"], iptw["weights"]

    def test_returns_before_and_after(self):
        """Returns SMD_before and SMD_after for each confounder."""
        aligned_df, t, w = self._get_aligned_data()
        bal = balance_diagnostics(aligned_df, ["age", "sex"], t, w)
        for d in bal["diagnostics"]:
            assert "smd_before" in d
            assert "smd_after" in d
            assert d["smd_after"] is not None

    def test_iptw_improves_balance(self):
        """Average SMD is lower after IPTW weighting."""
        aligned_df, t, w = self._get_aligned_data()
        bal = balance_diagnostics(aligned_df, ["age", "sex"], t, w)
        avg_before = np.mean([d["smd_before"] for d in bal["diagnostics"]])
        avg_after = np.mean([d["smd_after"] for d in bal["diagnostics"]])
        assert avg_after <= avg_before

    def test_threshold_is_point_one(self):
        """Balance threshold is 0.1."""
        aligned_df, t, w = self._get_aligned_data()
        bal = balance_diagnostics(aligned_df, ["age", "sex"], t, w)
        assert bal["threshold"] == 0.1

    def test_before_only_when_no_weights(self):
        """When iptw_weights is None, smd_after is None."""
        aligned_df, t, _ = self._get_aligned_data()
        bal = balance_diagnostics(aligned_df, ["age", "sex"], t, iptw_weights=None)
        for d in bal["diagnostics"]:
            assert d["smd_after"] is None

    def test_summary_counts(self):
        """Summary contains expected count keys."""
        aligned_df, t, w = self._get_aligned_data()
        bal = balance_diagnostics(aligned_df, ["age", "sex"], t, w)
        assert "n_balanced_before" in bal["summary"]
        assert "n_balanced_after" in bal["summary"]
        assert "n_total" in bal["summary"]
        assert bal["summary"]["n_total"] > 0


# ── TestEstimateTreatmentEffect ───────────────────────────────────────


class TestEstimateTreatmentEffect:
    """Tests for estimate_treatment_effect."""

    def _get_components(self):
        result = estimate_propensity_scores(
            DF, "treatment", ["age", "sex"], treatment_positive=1
        )
        iptw = calculate_iptw_weights(result["ps_scores"], result["treatment_binary"])
        aligned_df = DF.loc[result["treatment_binary"].index]
        return aligned_df, result["treatment_binary"], iptw["weights"]

    def test_binary_outcome_returns_or(self):
        """Binary outcome produces OR > 0."""
        df, t, w = self._get_components()
        te = estimate_treatment_effect(
            df, "outcome", t, w,
            outcome_type="binary", outcome_positive=1, n_bootstrap=50
        )
        assert te["value"] > 0

    def test_continuous_outcome_returns_mean_diff(self):
        """Continuous outcome produces a finite mean difference."""
        df, t, w = self._get_components()
        te = estimate_treatment_effect(
            df, "blood_pressure", t, w,
            outcome_type="continuous", n_bootstrap=50
        )
        assert np.isfinite(te["value"])

    def test_ci_brackets_estimate(self):
        """CI lower <= value <= CI upper."""
        df, t, w = self._get_components()
        te = estimate_treatment_effect(
            df, "outcome", t, w,
            outcome_type="binary", outcome_positive=1, n_bootstrap=100
        )
        assert np.isfinite(te["ci_lower"]) and np.isfinite(te["ci_upper"])
        assert te["ci_lower"] <= te["value"] <= te["ci_upper"]

    def test_effective_n_positive(self):
        """Effective N is positive."""
        df, t, w = self._get_components()
        te = estimate_treatment_effect(
            df, "outcome", t, w,
            outcome_type="binary", outcome_positive=1, n_bootstrap=50
        )
        assert te["effective_n"] > 0

    def test_n_observations_correct(self):
        """n_observations matches input size."""
        df, t, w = self._get_components()
        te = estimate_treatment_effect(
            df, "outcome", t, w,
            outcome_type="binary", outcome_positive=1, n_bootstrap=50
        )
        assert te["n_observations"] == len(t)

    def test_invalid_outcome_type_raises(self):
        """Invalid outcome_type raises ValueError."""
        df, t, w = self._get_components()
        with pytest.raises(ValueError, match="outcome_type must be"):
            estimate_treatment_effect(
                df, "outcome", t, w,
                outcome_type="Binary", n_bootstrap=10
            )


# ── TestRunPropensityScoreAnalysis ────────────────────────────────────


class TestRunPropensityScoreAnalysis:
    """Tests for the full pipeline run_propensity_score_analysis."""

    def test_full_pipeline_binary(self):
        """Full pipeline runs to completion with binary outcome."""
        result = run_propensity_score_analysis(
            DF, "outcome", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_positive=1,
            outcome_type="binary", n_bootstrap=50,
        )
        assert "ps_model" in result
        assert "common_support" in result
        assert "iptw" in result
        assert "balance" in result
        assert "treatment_effect" in result
        assert "interpretation" in result

    def test_full_pipeline_continuous(self):
        """Full pipeline runs to completion with continuous outcome."""
        result = run_propensity_score_analysis(
            DF, "blood_pressure", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_type="continuous", n_bootstrap=50,
        )
        assert result["treatment_effect"]["outcome_type"] == "continuous"
        assert np.isfinite(result["treatment_effect"]["value"])

    def test_ate_vs_att_different(self):
        """ATE and ATT estimates can differ."""
        ate = run_propensity_score_analysis(
            DF, "outcome", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_positive=1,
            estimand="ATE", n_bootstrap=50,
        )
        att = run_propensity_score_analysis(
            DF, "outcome", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_positive=1,
            estimand="ATT", n_bootstrap=50,
        )
        # They might be close but the estimands should be labeled differently
        assert ate["treatment_effect"]["estimand"] == "ATE"
        assert att["treatment_effect"]["estimand"] == "ATT"

    def test_survey_weighted(self):
        """Pipeline works with survey weights."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(99).uniform(0.5, 5.0, size=len(df))
        result = run_propensity_score_analysis(
            df, "outcome", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_positive=1,
            weight_col="weight", n_bootstrap=50,
        )
        assert "survey-weighted" in result["interpretation"].lower()

    def test_interpretation_present(self):
        """Top-level interpretation is a non-empty string."""
        result = run_propensity_score_analysis(
            DF, "outcome", "treatment", ["age", "sex"],
            treatment_positive=1, outcome_positive=1, n_bootstrap=50,
        )
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


# ── TestInterpretPropensityScore ──────────────────────────────────────


class TestInterpretPropensityScore:
    """Tests for interpret_propensity_score."""

    def _make_interpretation(self, **kwargs):
        defaults = {
            "estimand": "ATE",
            "effect_value": 1.5,
            "ci_lower": 1.1,
            "ci_upper": 2.0,
            "outcome_type": "binary",
            "treatment_name": "treatment",
            "confounder_names": ["age", "sex"],
            "n_obs": 500,
            "effective_n": 400,
            "all_balanced": True,
            "n_balanced": 2,
            "n_total_covariates": 2,
        }
        defaults.update(kwargs)
        return interpret_propensity_score(**defaults)

    def test_mentions_estimand(self):
        """Interpretation mentions ATE or ATT."""
        interp = self._make_interpretation(estimand="ATE")
        assert "ATE" in interp
        interp_att = self._make_interpretation(estimand="ATT")
        assert "ATT" in interp_att

    def test_mentions_confounders(self):
        """Interpretation lists confounder names."""
        interp = self._make_interpretation(confounder_names=["age", "smoking"])
        assert "age" in interp
        assert "smoking" in interp

    def test_mentions_significance(self):
        """Interpretation states statistical significance."""
        interp = self._make_interpretation(ci_lower=1.1, ci_upper=2.0)
        assert "statistically significant" in interp

    def test_not_significant(self):
        """Interpretation notes non-significance when CI crosses null."""
        interp = self._make_interpretation(
            effect_value=1.1, ci_lower=0.8, ci_upper=1.5
        )
        assert "NOT statistically significant" in interp

    def test_mentions_balance(self):
        """Interpretation mentions covariate balance."""
        interp = self._make_interpretation(all_balanced=True)
        assert "balance" in interp.lower()

    def test_poor_balance_warning(self):
        """Interpretation warns when not all covariates balanced."""
        interp = self._make_interpretation(
            all_balanced=False, n_balanced=1, n_total_covariates=3
        )
        assert "imbalance" in interp.lower() or "1 of 3" in interp

    def test_low_effective_n_warning(self):
        """Interpretation warns when effective N is much less than N."""
        interp = self._make_interpretation(n_obs=500, effective_n=100)
        assert "substantially smaller" in interp.lower()

    def test_survey_weighted_prefix(self):
        """Survey-weighted interpretation includes prefix."""
        interp = self._make_interpretation(weighted=True)
        assert "survey-weighted" in interp.lower()

    def test_nan_inputs_graceful(self):
        """NaN effect/CI values produce a bootstrap failure message, not 'nan' text."""
        interp = self._make_interpretation(
            effect_value=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
        )
        assert "nan" not in interp.lower()
        assert "bootstrap" in interp.lower() or "reliable" in interp.lower()
