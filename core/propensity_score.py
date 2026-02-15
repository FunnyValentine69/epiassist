"""Propensity score analysis via IPTW for causal inference.

Provides Inverse Probability of Treatment Weighting (IPTW) for
estimating average treatment effects adjusted for measured confounders.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial, Gaussian

from core.regression import _fit_glm, _is_categorical
from utils.constants import SMD_BALANCE_THRESHOLD
from utils.interpretations import interpret_propensity_score

# Epsilon to clip PS away from 0 and 1 (prevents infinite weights)
_PS_CLIP_EPSILON: float = 1e-6


def _prepare_ps_data(
    df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: list[str],
    treatment_positive: object,
    weight_col: str | None = None,
) -> tuple[pd.Series, pd.DataFrame, list[str], pd.Series | None]:
    """Prepare data for propensity score logistic regression.

    Binarizes treatment, encodes categorical confounders as dummies,
    drops NaN rows, and adds a constant to the design matrix.

    Args:
        df: Input DataFrame.
        treatment_col: Column name for binary treatment.
        confounder_cols: List of confounder column names.
        treatment_positive: Value indicating treated group (coded as 1).
        weight_col: Optional survey weight column.

    Returns:
        Tuple of (treatment_binary, X_with_constant, feature_names, weights).
    """
    if treatment_positive is None:
        raise ValueError(
            f"treatment_positive must be specified. Set the positive value "
            f"for '{treatment_col}' to define the treated group."
        )

    relevant_cols = [treatment_col] + list(confounder_cols)
    if weight_col is not None:
        relevant_cols.append(weight_col)
    subset = df[relevant_cols].dropna().copy()

    # Binarize treatment
    treatment = (subset[treatment_col] == treatment_positive).astype(int)

    # Build covariate matrix
    predictor_parts = []
    for col in confounder_cols:
        if _is_categorical(subset[col]):
            dummies = pd.get_dummies(subset[col], prefix=col, drop_first=True)
            dummies = dummies.astype(int)
            predictor_parts.append(dummies)
        else:
            predictor_parts.append(subset[[col]])

    X = pd.concat(predictor_parts, axis=1)
    feature_names = list(X.columns)
    X = sm.add_constant(X)

    aligned_weights = subset[weight_col] if weight_col is not None else None
    if aligned_weights is not None and (aligned_weights <= 0).any():
        raise ValueError("Weight column must contain only positive (> 0) values.")

    return treatment, X, feature_names, aligned_weights


def estimate_propensity_scores(
    df: pd.DataFrame,
    treatment_col: str,
    confounder_cols: list[str],
    treatment_positive: object,
    weight_col: str | None = None,
) -> dict:
    """Estimate propensity scores via logistic regression.

    Fits P(treatment=1 | confounders) using statsmodels GLM(Binomial).

    Args:
        df: Input DataFrame.
        treatment_col: Name of the binary treatment/exposure column.
        confounder_cols: List of confounder column names.
        treatment_positive: Value indicating the treated group.
        weight_col: Optional survey weight column.

    Returns:
        Dict with ps_scores, treatment_binary, n_observations, n_dropped,
        converged, common_support, model_coefficients, and survey_weights.

    Raises:
        ValueError: If no confounders or treatment has fewer than 2 levels.
    """
    if not confounder_cols:
        raise ValueError("At least one confounder is required for propensity score estimation.")

    n_total = len(df)
    treatment, X, feature_names, aligned_weights = _prepare_ps_data(
        df, treatment_col, confounder_cols, treatment_positive, weight_col
    )

    if treatment.nunique() < 2:
        raise ValueError(
            f"Treatment '{treatment_col}' has only one level after binarization. "
            f"Both treated and control groups must be present."
        )

    n_obs = len(treatment)
    n_dropped = n_total - n_obs

    if n_obs <= X.shape[1]:
        raise ValueError(
            f"Not enough observations ({n_obs}) for the number of parameters ({X.shape[1]}). "
            f"Reduce confounders or increase sample size."
        )

    fit_result = _fit_glm(treatment, X, Binomial(), aligned_weights)
    ps_scores = fit_result.predict(X)
    ps_scores = pd.Series(ps_scores, index=treatment.index, name="propensity_score")

    # Extract model coefficients
    coefficients = [
        {
            "variable": name,
            "coef": round(float(fit_result.params[name]), 4),
            "or": round(float(np.exp(fit_result.params[name])), 4),
            "p_value": round(float(fit_result.pvalues[name]), 6),
        }
        for name in feature_names
    ]

    return {
        "ps_scores": ps_scores,
        "treatment_binary": treatment,
        "n_observations": n_obs,
        "n_dropped": n_dropped,
        "converged": bool(fit_result.converged),
        "model_coefficients": coefficients,
        "survey_weights": aligned_weights,
    }


def assess_common_support(
    ps_treated: np.ndarray,
    ps_control: np.ndarray,
) -> dict:
    """Assess overlap of propensity score distributions between groups.

    Args:
        ps_treated: Propensity scores for treated group.
        ps_control: Propensity scores for control group.

    Returns:
        Dict with treated_range, control_range, overlap_range,
        overlap_pct, sufficient, and warning.
    """
    t_min, t_max = float(np.min(ps_treated)), float(np.max(ps_treated))
    c_min, c_max = float(np.min(ps_control)), float(np.max(ps_control))

    overlap_min = max(t_min, c_min)
    overlap_max = min(t_max, c_max)

    if overlap_max <= overlap_min:
        return {
            "treated_range": (t_min, t_max),
            "control_range": (c_min, c_max),
            "overlap_range": (0.0, 0.0),
            "overlap_pct": 0.0,
            "sufficient": False,
            "warning": "No overlap between treated and control PS distributions. "
                       "IPTW estimates are unreliable.",
        }

    all_ps = np.concatenate([ps_treated, ps_control])
    in_overlap = np.sum((all_ps >= overlap_min) & (all_ps <= overlap_max))
    overlap_pct = float(in_overlap / len(all_ps) * 100)

    sufficient = overlap_pct >= 90.0
    warning_msg = None
    if not sufficient:
        warning_msg = (
            f"Only {overlap_pct:.0f}% of observations fall within the region of "
            f"common support. IPTW estimates may be unreliable."
        )

    return {
        "treated_range": (t_min, t_max),
        "control_range": (c_min, c_max),
        "overlap_range": (overlap_min, overlap_max),
        "overlap_pct": round(overlap_pct, 1),
        "sufficient": sufficient,
        "warning": warning_msg,
    }


def calculate_iptw_weights(
    ps: pd.Series,
    treatment: pd.Series,
    estimand: str = "ATE",
    stabilized: bool = True,
    trim_quantile: float = 0.0,
) -> dict:
    """Calculate Inverse Probability of Treatment Weights.

    ATE weights:  treated -> 1/PS,      control -> 1/(1-PS)
    ATT weights:  treated -> 1,          control -> PS/(1-PS)

    Stabilized weights multiply by P(T=1) and P(T=0) respectively.

    Args:
        ps: Propensity scores (0 < ps < 1).
        treatment: Binary treatment indicator (0/1).
        estimand: "ATE" or "ATT".
        stabilized: If True, use stabilized weights (recommended).
        trim_quantile: Trim extreme PS at this quantile (0 = no trimming).

    Returns:
        Dict with weights, estimand, stabilized, trimmed_n, weight_summary.

    Raises:
        ValueError: If estimand is not ATE or ATT.
    """
    if estimand not in ("ATE", "ATT"):
        raise ValueError(f"estimand must be 'ATE' or 'ATT', got '{estimand}'")

    ps_clipped = ps.clip(lower=_PS_CLIP_EPSILON, upper=1 - _PS_CLIP_EPSILON)

    # Optional trimming
    trimmed_n = 0
    if trim_quantile > 0:
        lower_bound = ps_clipped.quantile(trim_quantile)
        upper_bound = ps_clipped.quantile(1 - trim_quantile)
        mask = (ps_clipped >= lower_bound) & (ps_clipped <= upper_bound)
        trimmed_n = int((~mask).sum())
        ps_clipped = ps_clipped.clip(lower=lower_bound, upper=upper_bound)

    # Marginal treatment probability (for stabilization)
    p_treat = treatment.mean()

    if estimand == "ATE":
        if stabilized:
            weights = treatment * p_treat / ps_clipped + (1 - treatment) * (1 - p_treat) / (1 - ps_clipped)
        else:
            weights = treatment / ps_clipped + (1 - treatment) / (1 - ps_clipped)
    else:  # ATT
        if stabilized:
            weights = treatment + (1 - treatment) * p_treat * ps_clipped / ((1 - p_treat) * (1 - ps_clipped))
        else:
            weights = treatment + (1 - treatment) * ps_clipped / (1 - ps_clipped)

    # Weight summary
    sum_w = weights.sum()
    effective_n = float(sum_w**2 / (weights**2).sum()) if (weights**2).sum() > 0 else 0.0

    return {
        "weights": weights,
        "estimand": estimand,
        "stabilized": stabilized,
        "trimmed_n": trimmed_n,
        "weight_summary": {
            "mean": round(float(weights.mean()), 4),
            "sd": round(float(weights.std()), 4),
            "min": round(float(weights.min()), 4),
            "max": round(float(weights.max()), 4),
            "effective_n": round(effective_n, 1),
        },
    }


def calculate_smd(
    treated_values: pd.Series,
    control_values: pd.Series,
    treated_weights: pd.Series | None = None,
    control_weights: pd.Series | None = None,
) -> float:
    """Calculate Standardized Mean Difference between two groups.

    For continuous variables:
        SMD = |mean_t - mean_c| / sqrt((sd_t^2 + sd_c^2) / 2)
    For binary variables:
        SMD = |p_t - p_c| / sqrt((p_t*(1-p_t) + p_c*(1-p_c)) / 2)

    Args:
        treated_values: Values for treated group.
        control_values: Values for control group.
        treated_weights: Optional weights for treated group.
        control_weights: Optional weights for control group.

    Returns:
        Absolute value of the standardized mean difference.
    """
    def _weighted_mean(vals: pd.Series, w: pd.Series | None) -> float:
        if w is not None:
            return float(np.average(vals, weights=w))
        return float(vals.mean())

    def _weighted_var(vals: pd.Series, w: pd.Series | None) -> float:
        if w is not None:
            avg = np.average(vals, weights=w)
            return float(np.average((vals - avg) ** 2, weights=w))
        return float(vals.var(ddof=0))

    mean_t = _weighted_mean(treated_values, treated_weights)
    mean_c = _weighted_mean(control_values, control_weights)
    var_t = _weighted_var(treated_values, treated_weights)
    var_c = _weighted_var(control_values, control_weights)

    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd < 1e-10:
        return 0.0

    return abs(mean_t - mean_c) / pooled_sd


def balance_diagnostics(
    df: pd.DataFrame,
    confounder_cols: list[str],
    treatment_binary: pd.Series,
    iptw_weights: pd.Series | None = None,
) -> dict:
    """Calculate balance diagnostics before and after IPTW adjustment.

    Args:
        df: DataFrame aligned with treatment_binary and iptw_weights.
        confounder_cols: List of confounder column names to check.
        treatment_binary: Binary treatment indicator (0/1).
        iptw_weights: IPTW weights for "after" comparison. None for before-only.

    Returns:
        Dict with diagnostics list, summary, threshold, and all_balanced.
    """
    treated_mask = treatment_binary == 1
    control_mask = treatment_binary == 0

    diagnostics = []

    for col in confounder_cols:
        series = df[col]

        # Expand categorical to dummies for SMD calculation
        if _is_categorical(series):
            dummies = pd.get_dummies(series, prefix=col, drop_first=True).astype(int)
            cols_to_check = list(dummies.columns)
            data_for_check = dummies
        else:
            cols_to_check = [col]
            data_for_check = df[[col]]

        for check_col in cols_to_check:
            vals = data_for_check[check_col]
            t_vals = vals[treated_mask]
            c_vals = vals[control_mask]

            smd_before = calculate_smd(t_vals, c_vals)

            smd_after = None
            balanced_after = None
            if iptw_weights is not None:
                t_w = iptw_weights[treated_mask]
                c_w = iptw_weights[control_mask]
                smd_after = calculate_smd(t_vals, c_vals, t_w, c_w)
                balanced_after = smd_after < SMD_BALANCE_THRESHOLD

            diagnostics.append({
                "variable": check_col,
                "smd_before": round(smd_before, 4),
                "smd_after": round(smd_after, 4) if smd_after is not None else None,
                "balanced_before": smd_before < SMD_BALANCE_THRESHOLD,
                "balanced_after": balanced_after,
            })

    n_total = len(diagnostics)
    n_balanced_before = sum(1 for d in diagnostics if d["balanced_before"])
    n_balanced_after = (
        sum(1 for d in diagnostics if d["balanced_after"])
        if iptw_weights is not None
        else None
    )
    all_balanced = n_balanced_after == n_total if n_balanced_after is not None else False

    return {
        "diagnostics": diagnostics,
        "summary": {
            "n_balanced_before": n_balanced_before,
            "n_balanced_after": n_balanced_after,
            "n_total": n_total,
        },
        "threshold": SMD_BALANCE_THRESHOLD,
        "all_balanced": all_balanced,
    }


def estimate_treatment_effect(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_binary: pd.Series,
    iptw_weights: pd.Series,
    outcome_type: str = "binary",
    outcome_positive: object | None = None,
    survey_weights: pd.Series | None = None,
    n_bootstrap: int = 200,
) -> dict:
    """Estimate treatment effect using IPTW-weighted outcome model.

    For binary outcomes: weighted logistic regression -> OR.
    For continuous outcomes: weighted linear regression -> mean difference.

    CIs via bootstrap (resample observations, re-estimate weighted outcome model).

    Args:
        df: Input DataFrame aligned with treatment_binary and iptw_weights.
        outcome_col: Name of the outcome column.
        treatment_binary: Binary treatment indicator (0/1).
        iptw_weights: IPTW weights.
        outcome_type: "binary" or "continuous".
        outcome_positive: For binary outcomes, the value coded as 1.
        survey_weights: Optional survey weights (multiplied with IPTW).
        n_bootstrap: Number of bootstrap iterations for CI.

    Returns:
        Dict with value, ci_lower, ci_upper, outcome_type,
        n_observations, and effective_n.

    Raises:
        ValueError: If outcome_type is not 'binary' or 'continuous'.
    """
    if outcome_type not in ("binary", "continuous"):
        raise ValueError(
            f"outcome_type must be 'binary' or 'continuous', got '{outcome_type}'"
        )

    # Combine IPTW with survey weights if present
    final_weights = iptw_weights.copy()
    if survey_weights is not None:
        final_weights = final_weights * survey_weights

    # Prepare outcome
    outcome = df[outcome_col].loc[treatment_binary.index]
    if outcome_type == "binary" and outcome_positive is not None:
        outcome = (outcome == outcome_positive).astype(int)

    # Point estimate via weighted GLM
    family = Binomial() if outcome_type == "binary" else Gaussian()
    point_estimate = _fit_weighted_outcome(
        outcome, treatment_binary, final_weights, family, outcome_type
    )

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    boot_estimates = []
    n = len(treatment_binary)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_y = outcome.iloc[idx].reset_index(drop=True)
        boot_t = treatment_binary.iloc[idx].reset_index(drop=True)
        boot_w = final_weights.iloc[idx].reset_index(drop=True)

        try:
            est = _fit_weighted_outcome(boot_y, boot_t, boot_w, family, outcome_type)
            if np.isfinite(est):
                boot_estimates.append(est)
        except (ValueError, np.linalg.LinAlgError):
            continue

    if len(boot_estimates) < 10:
        ci_lower = float("nan")
        ci_upper = float("nan")
    else:
        ci_lower = float(np.percentile(boot_estimates, 2.5))
        ci_upper = float(np.percentile(boot_estimates, 97.5))

    sum_w = final_weights.sum()
    effective_n = float(sum_w**2 / (final_weights**2).sum()) if (final_weights**2).sum() > 0 else 0.0

    return {
        "value": round(point_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "outcome_type": outcome_type,
        "n_observations": n,
        "effective_n": round(effective_n, 1),
    }


def _fit_weighted_outcome(
    outcome: pd.Series,
    treatment: pd.Series,
    weights: pd.Series,
    family: sm.families.Family,
    outcome_type: str,
) -> float:
    """Fit a weighted outcome model and return the treatment effect.

    Args:
        outcome: Outcome variable (0/1 or continuous).
        treatment: Binary treatment indicator.
        weights: Combined IPTW (+ survey) weights.
        family: GLM family.
        outcome_type: "binary" or "continuous".

    Returns:
        Treatment effect (OR for binary, beta for continuous).
    """
    X = sm.add_constant(treatment)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.GLM(outcome, X, family=family, freq_weights=weights.values)
            result = model.fit()
        except Exception as e:
            raise ValueError(f"Outcome model fitting failed: {e}")

    treatment_coef = float(result.params.iloc[-1])

    if outcome_type == "binary":
        return float(np.exp(treatment_coef))
    return treatment_coef


def run_propensity_score_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    confounder_cols: list[str],
    treatment_positive: object,
    outcome_positive: object | None = None,
    outcome_type: str = "binary",
    estimand: str = "ATE",
    stabilized: bool = True,
    trim_quantile: float = 0.0,
    weight_col: str | None = None,
    n_bootstrap: int = 200,
) -> dict:
    """Run the full propensity score IPTW analysis pipeline.

    Pipeline:
    1. Estimate propensity scores (logistic regression)
    2. Assess common support
    3. Calculate IPTW weights
    4. Compute balance diagnostics (before/after)
    5. Estimate treatment effect with bootstrap CI

    Args:
        df: Input DataFrame.
        outcome_col: Name of the outcome column.
        treatment_col: Name of the treatment/exposure column.
        confounder_cols: Confounder column names.
        treatment_positive: Value for treated group.
        outcome_positive: Value for positive outcome (binary only).
        outcome_type: "binary" or "continuous".
        estimand: "ATE" or "ATT".
        stabilized: Use stabilized weights.
        trim_quantile: PS trimming quantile (0 = none).
        weight_col: Optional survey weight column.
        n_bootstrap: Number of bootstrap iterations.

    Returns:
        Dict with ps_model, common_support, iptw, balance,
        treatment_effect, and interpretation.
    """
    # 1. Estimate propensity scores
    ps_result = estimate_propensity_scores(
        df, treatment_col, confounder_cols, treatment_positive, weight_col
    )
    ps_scores = ps_result["ps_scores"]
    treatment_binary = ps_result["treatment_binary"]

    # 2. Common support
    treated_mask = treatment_binary == 1
    common_support = assess_common_support(
        ps_scores[treated_mask].values,
        ps_scores[~treated_mask].values,
    )

    # 3. IPTW weights
    iptw_result = calculate_iptw_weights(
        ps_scores, treatment_binary, estimand, stabilized, trim_quantile
    )

    # 4. Balance diagnostics — use aligned subset of df
    aligned_df = df.loc[treatment_binary.index]
    balance = balance_diagnostics(
        aligned_df, confounder_cols, treatment_binary, iptw_result["weights"]
    )

    # 5. Treatment effect
    te_result = estimate_treatment_effect(
        aligned_df, outcome_col, treatment_binary, iptw_result["weights"],
        outcome_type=outcome_type,
        outcome_positive=outcome_positive,
        survey_weights=ps_result["survey_weights"],
        n_bootstrap=n_bootstrap,
    )

    # Add estimand to treatment effect
    te_result["estimand"] = estimand

    # 6. Interpretation
    interpretation = interpret_propensity_score(
        estimand=estimand,
        effect_value=te_result["value"],
        ci_lower=te_result["ci_lower"],
        ci_upper=te_result["ci_upper"],
        outcome_type=outcome_type,
        treatment_name=treatment_col,
        confounder_names=confounder_cols,
        n_obs=ps_result["n_observations"],
        effective_n=iptw_result["weight_summary"]["effective_n"],
        all_balanced=balance["all_balanced"],
        n_balanced=balance["summary"]["n_balanced_after"] or 0,
        n_total_covariates=balance["summary"]["n_total"],
        weighted=ps_result["survey_weights"] is not None,
    )

    return {
        "ps_model": ps_result,
        "common_support": common_support,
        "iptw": iptw_result,
        "balance": balance,
        "treatment_effect": te_result,
        "interpretation": interpretation,
    }
