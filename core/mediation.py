"""Mediation analysis using Baron-Kenny's 3-step regression.

Provides indirect/direct effect decomposition, Sobel test, bootstrap CIs,
and proportion mediated for epidemiological mediation analysis.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial, Gaussian

from core.regression import _fit_glm, _is_categorical
from utils.interpretations import interpret_mediation


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_mediation_data(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    mediator_col: str,
    confounder_cols: list[str],
    exposure_positive: object,
    binarize_outcome: bool = False,
    outcome_positive: object | None = None,
    weight_col: str | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series | None, int, int]:
    """Prepare data for mediation models with a single listwise deletion.

    Drops NaN across outcome, exposure, mediator, and all confounders at once
    so all three Baron-Kenny models use identical rows.

    Args:
        df: Input DataFrame.
        outcome_col: Outcome column name.
        exposure_col: Exposure column name.
        mediator_col: Mediator column name.
        confounder_cols: Confounder column names.
        exposure_positive: Value that indicates exposed group.
        binarize_outcome: Whether to binarize the outcome.
        outcome_positive: Positive outcome value (required if binarize_outcome).
        weight_col: Optional survey weight column.

    Returns:
        Tuple of (outcome, exposure, mediator, confounders_X, weights, n_obs, n_dropped).
        confounders_X is a DataFrame with dummy-encoded categoricals and a constant.

    Raises:
        ValueError: If fewer than 10 observations remain or exposure has only one level.
    """
    n_total = len(df)

    relevant_cols = [outcome_col, exposure_col, mediator_col] + list(confounder_cols)
    if weight_col is not None:
        relevant_cols.append(weight_col)

    subset = df[relevant_cols].dropna().copy()
    n_obs = len(subset)
    n_dropped = n_total - n_obs

    if n_obs < 10:
        raise ValueError(
            f"Only {n_obs} complete observations after dropping missing values. "
            f"At least 10 are required."
        )

    # Binarize outcome if needed
    if binarize_outcome:
        subset[outcome_col] = (subset[outcome_col] == outcome_positive).astype(int)

    # Binarize exposure
    if exposure_positive is not None and _is_categorical(subset[exposure_col]):
        subset[exposure_col] = (subset[exposure_col] == exposure_positive).astype(int)

    # Check exposure has two levels
    if subset[exposure_col].nunique() < 2:
        raise ValueError(
            "Exposure variable has only one level after data preparation. "
            "Cannot fit mediation models."
        )

    outcome = subset[outcome_col]
    exposure = subset[exposure_col]
    mediator = subset[mediator_col]

    # Build confounder matrix with dummy encoding
    confounder_parts: list[pd.DataFrame] = []
    for col in confounder_cols:
        if _is_categorical(subset[col]):
            dummies = pd.get_dummies(subset[col], prefix=col, drop_first=True)
            dummies = dummies.astype(int)
            confounder_parts.append(dummies)
        else:
            confounder_parts.append(subset[[col]])

    if confounder_parts:
        confounders_X = pd.concat(confounder_parts, axis=1)
    else:
        confounders_X = pd.DataFrame(index=subset.index)

    # Extract weights
    aligned_weights = None
    if weight_col is not None:
        aligned_weights = subset[weight_col]
        if (aligned_weights <= 0).any():
            raise ValueError("Weight column must contain only positive (> 0) values.")

    return outcome, exposure, mediator, confounders_X, aligned_weights, n_obs, n_dropped


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def _build_design(predictors: list[pd.Series | pd.DataFrame]) -> pd.DataFrame:
    """Concatenate predictor parts and add constant.

    Args:
        predictors: List of Series or DataFrames to combine as columns.

    Returns:
        Design matrix with constant column prepended.
    """
    parts = [p.to_frame() if isinstance(p, pd.Series) else p for p in predictors]
    return sm.add_constant(pd.concat(parts, axis=1))


def _fit_and_extract(
    y: pd.Series,
    X: pd.DataFrame,
    family: object,
    weights: pd.Series | None,
    coef_names: list[str],
    suffix_map: dict[str, str],
) -> dict:
    """Fit a GLM and extract named coefficients, or return NaN on failure.

    Args:
        y: Response variable.
        X: Design matrix.
        family: GLM family (Gaussian, Binomial, etc.).
        weights: Optional sample weights.
        coef_names: Column names to extract from the fitted model.
        suffix_map: Maps each coef_name to its output key suffix
            (e.g. {exposure_name: "c"} produces keys coef_c, se_c, p_c).

    Returns:
        Dict with coef/se/p for each name plus a "converged" flag.
    """
    try:
        fit = _fit_glm(y, X, family, weights)
        result = {"converged": bool(fit.converged)}
        for name in coef_names:
            suffix = suffix_map[name]
            result[f"coef_{suffix}"] = float(fit.params[name])
            result[f"se_{suffix}"] = float(fit.bse[name])
            result[f"p_{suffix}"] = float(fit.pvalues[name])
        return result
    except (ValueError, KeyError) as e:
        result = {"converged": False, "error": str(e)}
        for name in coef_names:
            suffix = suffix_map[name]
            result[f"coef_{suffix}"] = float("nan")
            result[f"se_{suffix}"] = float("nan")
            result[f"p_{suffix}"] = float("nan")
        return result


def fit_mediation_models(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    mediator_col: str,
    confounder_cols: list[str],
    exposure_positive: object,
    outcome_type: str = "continuous",
    binarize_outcome: bool = False,
    outcome_positive: object | None = None,
    weight_col: str | None = None,
) -> dict:
    """Fit all three Baron-Kenny mediation models.

    Model 1 (total): outcome ~ exposure + confounders (c path)
    Model 2 (a path): mediator ~ exposure + confounders
    Model 3 (direct): outcome ~ exposure + mediator + confounders (c' and b paths)

    Args:
        df: Input DataFrame.
        outcome_col: Outcome column name.
        exposure_col: Exposure column name.
        mediator_col: Mediator column name.
        confounder_cols: Confounder column names.
        exposure_positive: Value indicating exposed group.
        outcome_type: "binary" or "continuous".
        binarize_outcome: Whether to binarize the outcome.
        outcome_positive: Positive outcome value.
        weight_col: Optional survey weight column.

    Returns:
        Dict with keys: models, n_observations, n_dropped.

    Raises:
        ValueError: If any model fails to fit.
    """
    outcome, exposure, mediator, confounders_X, weights, n_obs, n_dropped = (
        _prepare_mediation_data(
            df, outcome_col, exposure_col, mediator_col, confounder_cols,
            exposure_positive, binarize_outcome, outcome_positive, weight_col,
        )
    )

    outcome_family = Binomial() if outcome_type == "binary" else Gaussian()
    mediator_family = Gaussian()  # Mediator always treated as continuous
    exposure_name = exposure.name

    # Models 1 and 2 share the same design matrix: exposure + confounders
    X_exposure_only = _build_design([exposure, confounders_X])

    # Model 1: Total effect -- outcome ~ exposure + confounders
    total = _fit_and_extract(
        outcome, X_exposure_only, outcome_family, weights,
        [exposure_name], {exposure_name: "c"},
    )

    # Model 2: a path -- mediator ~ exposure + confounders
    a_path = _fit_and_extract(
        mediator, X_exposure_only, mediator_family, weights,
        [exposure_name], {exposure_name: "a"},
    )

    # Model 3: Direct -- outcome ~ exposure + mediator + confounders
    X_direct = _build_design([exposure, mediator, confounders_X])
    direct = _fit_and_extract(
        outcome, X_direct, outcome_family, weights,
        [exposure_name, mediator_col],
        {exposure_name: "c_prime", mediator_col: "b"},
    )

    return {
        "models": {"total": total, "a_path": a_path, "direct": direct},
        "n_observations": n_obs,
        "n_dropped": n_dropped,
    }


# ---------------------------------------------------------------------------
# Effect calculation
# ---------------------------------------------------------------------------


def calculate_mediation_effects(
    a: float,
    b: float,
    c: float,
    c_prime: float,
    se_a: float,
    se_b: float,
    outcome_type: str = "continuous",
) -> dict:
    """Compute indirect, direct, total effects and Sobel test.

    Args:
        a: Coefficient from a-path (exposure → mediator).
        b: Coefficient from b-path (mediator → outcome, adjusted for exposure).
        c: Total effect coefficient (exposure → outcome).
        c_prime: Direct effect coefficient (exposure → outcome, adjusted for mediator).
        se_a: Standard error of a.
        se_b: Standard error of b.
        outcome_type: "continuous" (product method) or "binary" (difference method).

    Returns:
        Dict with indirect, direct, total, proportion_mediated, method,
        sobel_se, sobel_z, sobel_p.
    """
    from scipy import stats

    direct = c_prime
    total = c

    if outcome_type == "continuous":
        indirect = a * b
        method = "product"
        # Sobel test
        sobel_se = float(np.sqrt(a**2 * se_b**2 + b**2 * se_a**2))
        if sobel_se > 0:
            sobel_z = float(indirect / sobel_se)
            sobel_p = float(2 * (1 - stats.norm.cdf(abs(sobel_z))))
        else:
            sobel_z = None
            sobel_p = None
    else:
        # Binary outcome: difference method
        indirect = c - c_prime
        method = "difference"
        sobel_se = None
        sobel_z = None
        sobel_p = None

    # Proportion mediated (only when indirect and total have same sign)
    if total != 0 and np.sign(indirect) == np.sign(total):
        proportion_mediated = float(indirect / total)
        # Clamp to [0, 1] for interpretability
        proportion_mediated = max(0.0, min(1.0, proportion_mediated))
    else:
        proportion_mediated = None

    return {
        "indirect": float(indirect),
        "direct": float(direct),
        "total": float(total),
        "proportion_mediated": proportion_mediated,
        "method": method,
        "sobel_se": sobel_se,
        "sobel_z": sobel_z,
        "sobel_p": sobel_p,
    }


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def bootstrap_mediation_ci(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    mediator_col: str,
    confounder_cols: list[str],
    exposure_positive: object,
    outcome_type: str = "continuous",
    binarize_outcome: bool = False,
    outcome_positive: object | None = None,
    weight_col: str | None = None,
    n_boot: int = 200,
) -> dict:
    """Bootstrap confidence intervals for mediation effects.

    Resamples the full dataset and refits all 3 models per iteration.

    Args:
        df: Input DataFrame.
        outcome_col: Outcome column name.
        exposure_col: Exposure column name.
        mediator_col: Mediator column name.
        confounder_cols: Confounder column names.
        exposure_positive: Value indicating exposed group.
        outcome_type: "binary" or "continuous".
        binarize_outcome: Whether to binarize the outcome.
        outcome_positive: Positive outcome value.
        weight_col: Optional survey weight column.
        n_boot: Number of bootstrap iterations.

    Returns:
        Dict with indirect_ci, direct_ci, total_ci (each a tuple of floats).
    """
    # Prepare data once to get the clean subset
    outcome, exposure, mediator, confounders_X, weights, n_obs, _ = (
        _prepare_mediation_data(
            df, outcome_col, exposure_col, mediator_col, confounder_cols,
            exposure_positive, binarize_outcome, outcome_positive, weight_col,
        )
    )

    outcome_family = Binomial() if outcome_type == "binary" else Gaussian()
    mediator_family = Gaussian()

    # Reconstruct a clean DataFrame for resampling
    clean_df = pd.concat([outcome, exposure, mediator, confounders_X], axis=1)
    if weights is not None:
        clean_df = pd.concat([clean_df, weights.rename("__weight__")], axis=1)

    rng = np.random.default_rng(42)
    boot_indirect: list[float] = []
    boot_direct: list[float] = []
    boot_total: list[float] = []

    exposure_name = exposure.name
    confounder_x_cols = list(confounders_X.columns)

    for _ in range(n_boot):
        idx = rng.choice(n_obs, size=n_obs, replace=True)
        sample = clean_df.iloc[idx].reset_index(drop=True)

        s_outcome = sample[outcome_col]
        s_exposure = sample[exposure_col]
        s_mediator = sample[mediator_col]
        s_confounders = sample[confounder_x_cols] if confounder_x_cols else pd.DataFrame(index=sample.index)
        s_weights = sample["__weight__"] if weights is not None else None

        try:
            # Models 1 & 2 share design: exposure + confounders
            X_exp = _build_design([s_exposure, s_confounders])
            fit1 = _fit_glm(s_outcome, X_exp, outcome_family, s_weights)
            coef_c = float(fit1.params[exposure_name])

            fit2 = _fit_glm(s_mediator, X_exp, mediator_family, s_weights)
            coef_a = float(fit2.params[exposure_name])

            # Model 3: exposure + mediator + confounders
            X_full = _build_design([s_exposure, s_mediator, s_confounders])
            fit3 = _fit_glm(s_outcome, X_full, outcome_family, s_weights)
            coef_c_prime = float(fit3.params[exposure_name])
            coef_b = float(fit3.params[mediator_col])

            # Calculate effects
            if outcome_type == "continuous":
                indirect = coef_a * coef_b
            else:
                indirect = coef_c - coef_c_prime

            if np.isfinite(indirect) and np.isfinite(coef_c_prime) and np.isfinite(coef_c):
                boot_indirect.append(indirect)
                boot_direct.append(coef_c_prime)
                boot_total.append(coef_c)
        except (ValueError, np.linalg.LinAlgError):
            continue

    # Percentile CIs
    if len(boot_indirect) < 10:
        nan_ci = (float("nan"), float("nan"))
        return {"indirect_ci": nan_ci, "direct_ci": nan_ci, "total_ci": nan_ci}

    return {
        "indirect_ci": (
            float(np.percentile(boot_indirect, 2.5)),
            float(np.percentile(boot_indirect, 97.5)),
        ),
        "direct_ci": (
            float(np.percentile(boot_direct, 2.5)),
            float(np.percentile(boot_direct, 97.5)),
        ),
        "total_ci": (
            float(np.percentile(boot_total, 2.5)),
            float(np.percentile(boot_total, 97.5)),
        ),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_mediation_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    mediator_col: str,
    confounder_cols: list[str],
    exposure_positive: object,
    outcome_type: str = "continuous",
    binarize_outcome: bool = False,
    outcome_positive: object | None = None,
    weight_col: str | None = None,
    n_boot: int = 200,
) -> dict:
    """Run complete Baron-Kenny mediation analysis.

    Fits all 3 models, computes effects, bootstraps CIs, and generates
    a plain English interpretation.

    Args:
        df: Input DataFrame.
        outcome_col: Outcome column name.
        exposure_col: Exposure column name.
        mediator_col: Mediator column name.
        confounder_cols: Confounder column names.
        exposure_positive: Value indicating exposed group.
        outcome_type: "binary" or "continuous".
        binarize_outcome: Whether to binarize the outcome.
        outcome_positive: Positive outcome value.
        weight_col: Optional survey weight column.
        n_boot: Number of bootstrap iterations.

    Returns:
        Complete mediation analysis result dict.
    """
    # Step 1: Fit models
    model_result = fit_mediation_models(
        df, outcome_col, exposure_col, mediator_col, confounder_cols,
        exposure_positive, outcome_type, binarize_outcome, outcome_positive,
        weight_col,
    )
    models = model_result["models"]

    # Check for model failures before computing effects
    failed_models = {
        name: data["error"]
        for name, data in models.items()
        if "error" in data
    }
    if failed_models:
        error_details = "; ".join(
            f"{name}: {err}" for name, err in failed_models.items()
        )
        raise ValueError(
            f"Cannot compute mediation effects — model(s) failed to fit: "
            f"{error_details}"
        )

    # Extract coefficients
    a = models["a_path"]["coef_a"]
    se_a = models["a_path"]["se_a"]
    b = models["direct"]["coef_b"]
    se_b = models["direct"]["se_b"]
    c = models["total"]["coef_c"]
    c_prime = models["direct"]["coef_c_prime"]

    # Step 2: Calculate effects
    effects = calculate_mediation_effects(a, b, c, c_prime, se_a, se_b, outcome_type)

    # Step 3: Bootstrap CIs
    ci = bootstrap_mediation_ci(
        df, outcome_col, exposure_col, mediator_col, confounder_cols,
        exposure_positive, outcome_type, binarize_outcome, outcome_positive,
        weight_col, n_boot,
    )

    # Step 4: Interpretation
    interpretation = interpret_mediation(
        mediator_name=mediator_col,
        exposure_name=exposure_col,
        outcome_name=outcome_col,
        indirect=effects["indirect"],
        direct=effects["direct"],
        total=effects["total"],
        indirect_ci=ci["indirect_ci"],
        direct_ci=ci["direct_ci"],
        sobel_p=effects["sobel_p"],
        proportion_mediated=effects["proportion_mediated"],
        method=effects["method"],
        n_obs=model_result["n_observations"],
        confounder_names=confounder_cols,
        weighted=weight_col is not None,
    )

    return {
        "models": models,
        "effects": effects,
        "ci": ci,
        "n_observations": model_result["n_observations"],
        "n_dropped": model_result["n_dropped"],
        "n_boot": n_boot,
        "interpretation": interpretation,
    }
