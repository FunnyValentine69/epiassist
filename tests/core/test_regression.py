"""Tests for regression analysis module."""

import numpy as np
import pandas as pd
import pytest

from core.regression import (
    _is_categorical,
    _prepare_regression_data,
    run_linear_regression,
    run_logistic_regression,
    run_poisson_regression,
)
from utils.interpretations import (
    interpret_linear_regression,
    interpret_logistic_regression,
    interpret_poisson_regression,
)


# ── Test data fixture ─────────────────────────────────────────────────────


def _make_regression_df(n: int = 200) -> pd.DataFrame:
    """Create a synthetic dataset for regression testing."""
    rng = np.random.default_rng(42)

    exposure = rng.choice(["Yes", "No"], size=n)
    age = rng.normal(50, 10, size=n).round(1)
    education = rng.choice(["High School", "College", "Graduate"], size=n)

    # Binary outcome (influenced by exposure and age)
    exposure_num = (exposure == "Yes").astype(int)
    logit = -2 + 0.8 * exposure_num + 0.02 * age
    prob = 1 / (1 + np.exp(-logit))
    disease = rng.binomial(1, prob)

    # Continuous outcome
    blood_pressure = 120 + 5 * exposure_num + 0.3 * age + rng.normal(0, 10, size=n)

    # Count outcome
    rate = np.exp(0.5 + 0.3 * exposure_num + 0.01 * age)
    visits = rng.poisson(rate)

    return pd.DataFrame({
        "exposure": exposure,
        "age": age,
        "education": education,
        "disease": disease.astype(str).tolist(),  # categorical binary
        "blood_pressure": blood_pressure.round(1),
        "visits": visits,
    })


DF = _make_regression_df()


# ── TestPrepareRegressionData ─────────────────────────────────────────────


class TestPrepareRegressionData:
    """Tests for _prepare_regression_data."""

    def test_nan_rows_dropped(self):
        """Rows with NaN in relevant columns are dropped."""
        df = DF.copy()
        df.loc[0, "age"] = np.nan
        df.loc[1, "exposure"] = np.nan
        y, X, _, _ = _prepare_regression_data(
            df, "blood_pressure", "exposure", ["age"],
            exposure_positive="Yes",
        )
        assert len(y) == len(DF) - 2

    def test_categorical_confounders_get_dummies(self):
        """Categorical confounders are encoded as dummy variables."""
        y, X, feature_names, _ = _prepare_regression_data(
            DF, "blood_pressure", "exposure", ["education"],
            exposure_positive="Yes",
        )
        # education has 3 levels → 2 dummies after drop_first
        edu_cols = [f for f in feature_names if f.startswith("education_")]
        assert len(edu_cols) == 2

    def test_exposure_binarized(self):
        """Exposure is binarized when exposure_positive is given."""
        y, X, feature_names, _ = _prepare_regression_data(
            DF, "blood_pressure", "exposure", [],
            exposure_positive="Yes",
        )
        assert "exposure" in feature_names
        # Values should be 0 or 1
        exp_values = X["exposure"].unique()
        assert set(exp_values) == {0, 1}

    def test_continuous_exposure_passed_through(self):
        """Numeric exposure with >10 unique values is not binarized."""
        y, X, feature_names, _ = _prepare_regression_data(
            DF, "blood_pressure", "age", [],
        )
        assert "age" in feature_names
        assert X["age"].nunique() > 2

    def test_constant_column_added(self):
        """X matrix contains a constant column."""
        y, X, _, _ = _prepare_regression_data(
            DF, "blood_pressure", "exposure", [],
            exposure_positive="Yes",
        )
        assert "const" in X.columns

    def test_weight_col_nan_drops_row(self):
        """NaN in weight column causes that row to be dropped."""
        df = DF.copy()
        df["weight"] = 1.0
        df.loc[0, "weight"] = np.nan
        df.loc[1, "weight"] = np.nan
        y, X, _, weights = _prepare_regression_data(
            df, "blood_pressure", "exposure", [],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert len(y) == len(DF) - 2
        assert weights is not None
        assert len(weights) == len(y)

    def test_weight_col_returned_aligned(self):
        """Returned weights have same length as y."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        y, X, _, weights = _prepare_regression_data(
            df, "blood_pressure", "exposure", [],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert weights is not None
        assert len(weights) == len(y)

    def test_no_weight_col_returns_none(self):
        """When weight_col is None, fourth return element is None."""
        _, _, _, weights = _prepare_regression_data(
            DF, "blood_pressure", "exposure", [],
            exposure_positive="Yes",
        )
        assert weights is None


# ── TestRunLogisticRegression ──────────────────────────────────────────────


class TestRunLogisticRegression:
    """Tests for run_logistic_regression."""

    def test_basic_run(self):
        """Basic run returns valid dict with all expected keys."""
        result = run_logistic_regression(
            DF, "disease", "exposure", ["age"],
            outcome_positive="1",
            exposure_positive="Yes",
        )
        assert result["model_type"] == "logistic"
        assert "exposure_effect" in result
        assert "coefficients" in result
        assert "model_fit" in result
        assert "interpretation" in result
        assert result["n_observations"] > 0

    def test_or_positive_and_ci_brackets(self):
        """OR is positive and CI brackets the estimate."""
        result = run_logistic_regression(
            DF, "disease", "exposure", ["age"],
            outcome_positive="1",
            exposure_positive="Yes",
        )
        eff = result["exposure_effect"]
        assert eff["effect"] > 0
        assert eff["ci_lower"] <= eff["effect"] <= eff["ci_upper"]

    def test_p_value_in_range(self):
        """P-value is in [0, 1]."""
        result = run_logistic_regression(
            DF, "disease", "exposure", [],
            outcome_positive="1",
            exposure_positive="Yes",
        )
        assert 0 <= result["exposure_effect"]["p_value"] <= 1

    def test_empty_confounders(self):
        """Works with no confounders (unadjusted model)."""
        result = run_logistic_regression(
            DF, "disease", "exposure", [],
            outcome_positive="1",
            exposure_positive="Yes",
        )
        assert result["model_type"] == "logistic"
        assert len(result["coefficients"]) == 1  # exposure only

    def test_single_level_outcome_raises(self):
        """Raises ValueError if outcome has only one level after binarization."""
        df = DF.copy()
        df["constant_outcome"] = "Yes"
        with pytest.raises(ValueError, match="only one level"):
            run_logistic_regression(
                df, "constant_outcome", "exposure", [],
                outcome_positive="No",
                exposure_positive="Yes",
            )

    def test_weighted_regression(self):
        """Weighted logistic regression returns weighted=True."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_logistic_regression(
            df, "disease", "exposure", ["age"],
            outcome_positive="1",
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert result["weighted"] is True
        assert result["exposure_effect"]["effect"] > 0

    def test_weighted_interpretation_mentions_survey(self):
        """Weighted logistic regression interpretation says 'survey-weighted'."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_logistic_regression(
            df, "disease", "exposure", ["age"],
            outcome_positive="1",
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert "survey-weighted" in result["interpretation"].lower()


# ── TestRunLinearRegression ────────────────────────────────────────────────


class TestRunLinearRegression:
    """Tests for run_linear_regression."""

    def test_basic_run(self):
        """Basic run returns valid dict with all expected keys."""
        result = run_linear_regression(
            DF, "blood_pressure", "exposure", ["age"],
            exposure_positive="Yes",
        )
        assert result["model_type"] == "linear"
        assert "r_squared" in result["model_fit"]
        assert result["n_observations"] > 0

    def test_r_squared_in_range(self):
        """R-squared is in [0, 1]."""
        result = run_linear_regression(
            DF, "blood_pressure", "exposure", ["age"],
            exposure_positive="Yes",
        )
        assert 0 <= result["model_fit"]["r_squared"] <= 1

    def test_non_numeric_outcome_raises(self):
        """Raises ValueError for non-numeric outcome."""
        with pytest.raises(ValueError, match="must be numeric"):
            run_linear_regression(
                DF, "education", "exposure", [],
                exposure_positive="Yes",
            )

    def test_continuous_exposure(self):
        """Works with continuous exposure (age predicting blood_pressure)."""
        result = run_linear_regression(
            DF, "blood_pressure", "age", [],
        )
        assert result["model_type"] == "linear"
        assert result["exposure_effect"]["variable"] == "age"

    def test_weighted_regression(self):
        """Weighted linear regression returns weighted=True."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_linear_regression(
            df, "blood_pressure", "exposure", ["age"],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert result["weighted"] is True

    def test_weighted_interpretation_mentions_survey(self):
        """Weighted linear regression interpretation says 'survey-weighted'."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_linear_regression(
            df, "blood_pressure", "exposure", ["age"],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert "survey-weighted" in result["interpretation"].lower()


# ── TestRunPoissonRegression ───────────────────────────────────────────────


class TestRunPoissonRegression:
    """Tests for run_poisson_regression."""

    def test_basic_run(self):
        """Basic run returns valid dict with all expected keys."""
        result = run_poisson_regression(
            DF, "visits", "exposure", ["age"],
            exposure_positive="Yes",
        )
        assert result["model_type"] == "poisson"
        assert result["n_observations"] > 0

    def test_irr_positive_and_ci_brackets(self):
        """IRR is positive and CI brackets the estimate."""
        result = run_poisson_regression(
            DF, "visits", "exposure", ["age"],
            exposure_positive="Yes",
        )
        eff = result["exposure_effect"]
        assert eff["effect"] > 0
        assert eff["ci_lower"] <= eff["effect"] <= eff["ci_upper"]

    def test_negative_outcome_raises(self):
        """Raises ValueError for negative outcome values."""
        df = DF.copy()
        df.loc[0, "visits"] = -1
        with pytest.raises(ValueError, match="negative values"):
            run_poisson_regression(
                df, "visits", "exposure", [],
                exposure_positive="Yes",
            )

    def test_empty_confounders(self):
        """Works with no confounders (unadjusted model)."""
        result = run_poisson_regression(
            DF, "visits", "exposure", [],
            exposure_positive="Yes",
        )
        assert len(result["coefficients"]) == 1

    def test_weighted_regression(self):
        """Weighted Poisson regression returns weighted=True."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_poisson_regression(
            df, "visits", "exposure", ["age"],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert result["weighted"] is True

    def test_weighted_interpretation_mentions_survey(self):
        """Weighted Poisson regression interpretation says 'survey-weighted'."""
        df = DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = run_poisson_regression(
            df, "visits", "exposure", ["age"],
            exposure_positive="Yes",
            weight_col="weight",
        )
        assert "survey-weighted" in result["interpretation"].lower()


# ── TestRegressionInterpretations ──────────────────────────────────────────


class TestRegressionInterpretations:
    """Tests for regression interpretation functions."""

    def test_logistic_interpretation(self):
        """Logistic interpretation mentions adjusted, confounders, and significance."""
        text = interpret_logistic_regression(
            "smoking", 2.5, 1.5, 4.2, 0.001, ["age", "sex"], 500
        )
        assert "adjusted" in text.lower()
        assert "age" in text
        assert "sex" in text
        assert "statistically significant" in text

    def test_linear_interpretation(self):
        """Linear interpretation mentions beta, R-squared, and confounders."""
        text = interpret_linear_regression(
            "smoking", 5.3, 2.1, 8.5, 0.003, ["age"], 300, 0.35
        )
        assert "adjusted" in text.lower()
        assert "age" in text
        assert "R²" in text
        assert "statistically significant" in text

    def test_poisson_interpretation(self):
        """Poisson interpretation mentions IRR, confounders, and significance."""
        text = interpret_poisson_regression(
            "smoking", 1.8, 1.2, 2.7, 0.01, ["age", "bmi"], 400
        )
        assert "adjusted" in text.lower()
        assert "age" in text
        assert "bmi" in text
        assert "statistically significant" in text

    def test_not_significant_interpretation(self):
        """Non-significant result says NOT statistically significant."""
        text = interpret_logistic_regression(
            "smoking", 1.1, 0.8, 1.5, 0.45, [], 100
        )
        assert "NOT statistically significant" in text
