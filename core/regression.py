"""Regression analysis for epidemiological data.

Provides logistic, linear, and Poisson regression using statsmodels GLM
for adjusted effect estimates (OR, β, IRR) with confidence intervals.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial, Gaussian, Poisson

from utils.interpretations import (
    interpret_linear_regression,
    interpret_logistic_regression,
    interpret_poisson_regression,
)


def _is_categorical(series: pd.Series) -> bool:
    """Check if a Series should be treated as categorical.

    Uses the same heuristic as data_analyzer.summarize_columns:
    NOT numeric dtype OR nunique <= 10.

    Args:
        series: A pandas Series.

    Returns:
        True if categorical, False if numeric.
    """
    return not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10


def _prepare_regression_data(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    confounder_cols: list[str],
    exposure_positive: object | None = None,
    binarize_outcome: bool = False,
    outcome_positive: object | None = None,
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Prepare data for GLM regression.

    Drops NaN rows, binarizes exposure/outcome if needed, encodes
    categorical confounders with dummies, and adds a constant.

    Args:
        df: Input DataFrame.
        outcome_col: Name of the outcome column.
        exposure_col: Name of the exposure column.
        confounder_cols: List of confounder column names.
        exposure_positive: If given, binarize exposure (1 = this value, 0 = other).
        binarize_outcome: If True, binarize outcome using outcome_positive.
        outcome_positive: Value to treat as positive outcome (1).

    Returns:
        Tuple of (y, X, feature_names) where feature_names lists
        predictors excluding the constant.
    """
    relevant_cols = [outcome_col, exposure_col] + list(confounder_cols)
    subset = df[relevant_cols].dropna().copy()

    # Binarize outcome if requested
    if binarize_outcome:
        subset[outcome_col] = (subset[outcome_col] == outcome_positive).astype(int)

    y = subset[outcome_col]

    # Binarize exposure if positive value given and exposure is categorical
    if exposure_positive is not None and _is_categorical(subset[exposure_col]):
        subset[exposure_col] = (subset[exposure_col] == exposure_positive).astype(int)

    # Build predictor DataFrame: exposure first, then confounders
    predictor_parts = [subset[[exposure_col]]]

    for col in confounder_cols:
        if _is_categorical(subset[col]):
            dummies = pd.get_dummies(subset[col], prefix=col, drop_first=True)
            # Ensure integer dtype for dummy columns
            dummies = dummies.astype(int)
            predictor_parts.append(dummies)
        else:
            predictor_parts.append(subset[[col]])

    X = pd.concat(predictor_parts, axis=1)
    feature_names = list(X.columns)

    # Add constant
    X = sm.add_constant(X)

    return y, X, feature_names


def _extract_glm_results(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_names: list[str],
    model_type: str,
    exposure_col: str,
) -> dict:
    """Extract standardized results from a fitted GLM.

    Args:
        result: Fitted GLM results object.
        feature_names: List of predictor names (excluding constant).
        model_type: One of "logistic", "linear", "poisson".
        exposure_col: Name of the exposure variable.

    Returns:
        Dict with exposure_effect, coefficients, model_fit, and converged.
    """
    params = result.params
    conf_int = result.conf_int()
    pvalues = result.pvalues
    bse = result.bse

    exponentiate = model_type in ("logistic", "poisson")

    coefficients = []
    exposure_effect = None

    for name in feature_names:
        coef = float(params[name])
        se = float(bse[name])
        ci_low = float(conf_int.loc[name, 0])
        ci_high = float(conf_int.loc[name, 1])
        p_val = float(pvalues[name])

        if exponentiate:
            effect = float(np.exp(coef))
            ci_lower = float(np.exp(ci_low))
            ci_upper = float(np.exp(ci_high))
        else:
            effect = coef
            ci_lower = ci_low
            ci_upper = ci_high

        entry = {
            "variable": name,
            "coef": round(coef, 4),
            "effect": round(effect, 4),
            "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4),
            "p_value": round(p_val, 6),
            "se": round(se, 4),
        }
        coefficients.append(entry)

        # Identify exposure variable
        if name == exposure_col and exposure_effect is None:
            exposure_effect = entry

    # Model fit statistics
    model_fit: dict = {
        "aic": round(float(result.aic), 2),
        "bic": round(float(result.bic_deviance), 2),
        "deviance": round(float(result.deviance), 2),
        "null_deviance": round(float(result.null_deviance), 2),
    }

    if model_type == "linear":
        # R-squared: 1 - deviance/null_deviance
        if result.null_deviance > 0:
            r_sq = 1 - result.deviance / result.null_deviance
            model_fit["r_squared"] = round(float(r_sq), 4)
        else:
            model_fit["r_squared"] = 0.0
    else:
        # Pseudo R-squared (McFadden)
        if result.null_deviance > 0:
            pseudo_r2 = 1 - result.deviance / result.null_deviance
            model_fit["pseudo_r_squared"] = round(float(pseudo_r2), 4)
        else:
            model_fit["pseudo_r_squared"] = 0.0

    if exposure_effect is None:
        available = [c["variable"] for c in coefficients]
        raise ValueError(
            f"Could not identify exposure variable '{exposure_col}' in model results. "
            f"Available variables: {available}"
        )

    return {
        "exposure_effect": exposure_effect,
        "coefficients": coefficients,
        "model_fit": model_fit,
        "converged": bool(result.converged),
    }


def run_logistic_regression(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    confounder_cols: list[str],
    outcome_positive: object,
    exposure_positive: object | None = None,
) -> dict:
    """Run logistic regression (adjusted odds ratios).

    Args:
        df: Input DataFrame.
        outcome_col: Name of the outcome column.
        exposure_col: Name of the exposure column.
        confounder_cols: List of confounder column names.
        outcome_positive: Value indicating positive outcome.
        exposure_positive: Value indicating positive exposure (for binarization).

    Returns:
        Dict with model_type, n_observations, n_dropped, converged,
        exposure_effect, coefficients, model_fit, and interpretation.

    Raises:
        ValueError: If outcome cannot be binarized or has only one level.
    """
    n_total = len(df)

    y, X, feature_names = _prepare_regression_data(
        df, outcome_col, exposure_col, confounder_cols,
        exposure_positive=exposure_positive,
        binarize_outcome=True,
        outcome_positive=outcome_positive,
    )

    n_obs = len(y)
    n_dropped = n_total - n_obs

    # Validate binary outcome
    unique_values = y.unique()
    if len(unique_values) < 2:
        raise ValueError(
            f"Outcome '{outcome_col}' has only one level after binarization. "
            f"Both 0 and 1 must be present."
        )

    # Check enough observations vs parameters
    n_params = X.shape[1]
    if n_obs <= n_params:
        raise ValueError(
            f"Not enough observations ({n_obs}) for the number of parameters ({n_params}). "
            f"Reduce confounders or increase sample size."
        )

    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=Warning)
        try:
            model = sm.GLM(y, X, family=Binomial())
            fit_result = model.fit()
        except Warning as w:
            raise ValueError(f"Model fitting warning: {w}")
        except Exception as e:
            raise ValueError(f"Model fitting failed: {e}")

    extracted = _extract_glm_results(fit_result, feature_names, "logistic", exposure_col)

    # Build interpretation
    confounder_names = confounder_cols if confounder_cols else []
    exp_eff = extracted["exposure_effect"]
    interpretation = interpret_logistic_regression(
        exposure_name=exposure_col,
        or_value=exp_eff["effect"],
        ci_lower=exp_eff["ci_lower"],
        ci_upper=exp_eff["ci_upper"],
        p_value=exp_eff["p_value"],
        confounder_names=confounder_names,
        n_obs=n_obs,
    )

    return {
        "model_type": "logistic",
        "n_observations": n_obs,
        "n_dropped": n_dropped,
        **extracted,
        "interpretation": interpretation,
    }


def run_linear_regression(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    confounder_cols: list[str],
    exposure_positive: object | None = None,
) -> dict:
    """Run linear regression (adjusted beta coefficients).

    Args:
        df: Input DataFrame.
        outcome_col: Name of the outcome column.
        exposure_col: Name of the exposure column.
        confounder_cols: List of confounder column names.
        exposure_positive: Value indicating positive exposure (for binarization).

    Returns:
        Dict with model_type, n_observations, n_dropped, converged,
        exposure_effect, coefficients, model_fit, and interpretation.

    Raises:
        ValueError: If outcome is not numeric.
    """
    n_total = len(df)

    # Validate numeric outcome before preparation
    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
        raise ValueError(
            f"Outcome '{outcome_col}' must be numeric for linear regression."
        )

    y, X, feature_names = _prepare_regression_data(
        df, outcome_col, exposure_col, confounder_cols,
        exposure_positive=exposure_positive,
    )

    n_obs = len(y)
    n_dropped = n_total - n_obs

    n_params = X.shape[1]
    if n_obs <= n_params:
        raise ValueError(
            f"Not enough observations ({n_obs}) for the number of parameters ({n_params})."
        )

    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=Warning)
        try:
            model = sm.GLM(y, X, family=Gaussian())
            fit_result = model.fit()
        except Warning as w:
            raise ValueError(f"Model fitting warning: {w}")
        except Exception as e:
            raise ValueError(f"Model fitting failed: {e}")

    extracted = _extract_glm_results(fit_result, feature_names, "linear", exposure_col)

    confounder_names = confounder_cols if confounder_cols else []
    exp_eff = extracted["exposure_effect"]
    interpretation = interpret_linear_regression(
        exposure_name=exposure_col,
        beta=exp_eff["effect"],
        ci_lower=exp_eff["ci_lower"],
        ci_upper=exp_eff["ci_upper"],
        p_value=exp_eff["p_value"],
        confounder_names=confounder_names,
        n_obs=n_obs,
        r_squared=extracted["model_fit"].get("r_squared", 0.0),
    )

    return {
        "model_type": "linear",
        "n_observations": n_obs,
        "n_dropped": n_dropped,
        **extracted,
        "interpretation": interpretation,
    }


def run_poisson_regression(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    confounder_cols: list[str],
    exposure_positive: object | None = None,
) -> dict:
    """Run Poisson regression (adjusted incidence rate ratios).

    Args:
        df: Input DataFrame.
        outcome_col: Name of the outcome column.
        exposure_col: Name of the exposure column.
        confounder_cols: List of confounder column names.
        exposure_positive: Value indicating positive exposure (for binarization).

    Returns:
        Dict with model_type, n_observations, n_dropped, converged,
        exposure_effect, coefficients, model_fit, and interpretation.

    Raises:
        ValueError: If outcome contains negative values or is not numeric.
    """
    n_total = len(df)

    # Validate numeric outcome
    if not pd.api.types.is_numeric_dtype(df[outcome_col]):
        raise ValueError(
            f"Outcome '{outcome_col}' must be numeric for Poisson regression."
        )

    # Check for negative values
    if (df[outcome_col].dropna() < 0).any():
        raise ValueError(
            f"Outcome '{outcome_col}' contains negative values. "
            f"Poisson regression requires non-negative counts."
        )

    y, X, feature_names = _prepare_regression_data(
        df, outcome_col, exposure_col, confounder_cols,
        exposure_positive=exposure_positive,
    )

    n_obs = len(y)
    n_dropped = n_total - n_obs

    n_params = X.shape[1]
    if n_obs <= n_params:
        raise ValueError(
            f"Not enough observations ({n_obs}) for the number of parameters ({n_params})."
        )

    # Fit model
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=Warning)
        try:
            model = sm.GLM(y, X, family=Poisson())
            fit_result = model.fit()
        except Warning as w:
            raise ValueError(f"Model fitting warning: {w}")
        except Exception as e:
            raise ValueError(f"Model fitting failed: {e}")

    extracted = _extract_glm_results(fit_result, feature_names, "poisson", exposure_col)

    confounder_names = confounder_cols if confounder_cols else []
    exp_eff = extracted["exposure_effect"]
    interpretation = interpret_poisson_regression(
        exposure_name=exposure_col,
        irr=exp_eff["effect"],
        ci_lower=exp_eff["ci_lower"],
        ci_upper=exp_eff["ci_upper"],
        p_value=exp_eff["p_value"],
        confounder_names=confounder_names,
        n_obs=n_obs,
    )

    return {
        "model_type": "poisson",
        "n_observations": n_obs,
        "n_dropped": n_dropped,
        **extracted,
        "interpretation": interpretation,
    }
