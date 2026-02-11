"""Meta-analysis engine for pooling effect estimates.

This module implements inverse-variance fixed-effect and DerSimonian-Laird
random-effects meta-analysis for ratio and difference measures.
"""

import math

from scipy import stats

from utils.constants import DIFFERENCE_MEASURES, RATIO_MEASURES, Z_SCORE_95
from utils.interpretations import interpret_heterogeneity, interpret_meta_analysis


def validate_studies(studies: list[dict]) -> list[str]:
    """Validate study inputs for meta-analysis.

    Args:
        studies: List of study dicts with keys: name, effect, ci_lower, ci_upper.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []

    if len(studies) < 2:
        errors.append("At least 2 studies are required for meta-analysis.")
        return errors

    for i, s in enumerate(studies):
        label = s.get("name", f"Study {i + 1}")

        if s.get("effect") is None:
            errors.append(f"{label}: Effect estimate is required.")
            continue

        if not isinstance(s["effect"], (int, float)):
            errors.append(f"{label}: Effect must be a number.")
            continue

        if s.get("ci_lower") is None or s.get("ci_upper") is None:
            errors.append(f"{label}: Both CI bounds are required.")
            continue

        if not isinstance(s["ci_lower"], (int, float)) or not isinstance(s["ci_upper"], (int, float)):
            errors.append(f"{label}: CI bounds must be numbers.")
            continue

        if s["ci_lower"] >= s["ci_upper"]:
            errors.append(f"{label}: CI lower must be less than CI upper.")

        if s["ci_lower"] > s["effect"] or s["effect"] > s["ci_upper"]:
            errors.append(f"{label}: Effect must be within CI bounds.")

    return errors


def _calculate_se_from_ci(
    ci_lower: float, ci_upper: float, is_log_scale: bool
) -> float:
    """Calculate standard error from confidence interval bounds.

    Args:
        ci_lower: Lower CI bound (natural scale).
        ci_upper: Upper CI bound (natural scale).
        is_log_scale: If True, compute SE on log scale (for ratio measures).

    Returns:
        Standard error.

    Raises:
        ValueError: If CI bounds are equal (SE would be zero).
    """
    if ci_lower == ci_upper:
        raise ValueError("CI lower and upper bounds must differ to compute SE.")

    if is_log_scale and (ci_lower <= 0 or ci_upper <= 0):
        raise ValueError("CI bounds must be positive for log-scale calculation.")

    if is_log_scale:
        return (math.log(ci_upper) - math.log(ci_lower)) / (2 * Z_SCORE_95)
    return (ci_upper - ci_lower) / (2 * Z_SCORE_95)


def _prepare_studies(
    studies: list[dict], measure_type: str
) -> list[dict]:
    """Transform studies to analysis scale and compute SE/variance/weight.

    For ratio measures, converts to log scale. For difference measures,
    stays on natural scale.

    Args:
        studies: Validated study dicts.
        measure_type: Measure type string (e.g., "OR", "MD").

    Returns:
        List of prepared study dicts with added fields:
        theta (analysis-scale effect), se, variance, weight.
    """
    is_ratio = measure_type in RATIO_MEASURES
    prepared = []

    for s in studies:
        se = _calculate_se_from_ci(s["ci_lower"], s["ci_upper"], is_ratio)
        variance = se ** 2

        if is_ratio:
            theta = math.log(s["effect"])
        else:
            theta = s["effect"]

        prepared.append({
            "name": s.get("name", ""),
            "effect": s["effect"],
            "ci_lower": s["ci_lower"],
            "ci_upper": s["ci_upper"],
            "theta": theta,
            "se": se,
            "variance": variance,
            "weight": 1.0 / variance,
        })

    return prepared


def fixed_effect_meta(
    prepared: list[dict], measure_type: str
) -> dict:
    """Inverse-variance fixed-effect meta-analysis.

    Args:
        prepared: Studies from _prepare_studies.
        measure_type: Measure type string.

    Returns:
        Dict with pooled estimate, CI, z-value, p-value, weights.
    """
    is_ratio = measure_type in RATIO_MEASURES
    total_weight = sum(s["weight"] for s in prepared)
    pooled_theta = sum(s["weight"] * s["theta"] for s in prepared) / total_weight
    se_pooled = 1.0 / math.sqrt(total_weight)

    ci_lower_theta = pooled_theta - Z_SCORE_95 * se_pooled
    ci_upper_theta = pooled_theta + Z_SCORE_95 * se_pooled

    z_value = pooled_theta / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

    # Back-transform for ratio measures
    if is_ratio:
        try:
            pooled = math.exp(pooled_theta)
            ci_lower = math.exp(ci_lower_theta)
            ci_upper = math.exp(ci_upper_theta)
        except OverflowError:
            raise ValueError(
                "Effect sizes too extreme for back-transformation. "
                "Check that input values are on the correct scale."
            )
    else:
        pooled = pooled_theta
        ci_lower = ci_lower_theta
        ci_upper = ci_upper_theta

    # Calculate percentage weights
    weights = [(s["weight"] / total_weight) * 100 for s in prepared]

    interpretation = interpret_meta_analysis(
        pooled, ci_lower, ci_upper, measure_type, "fixed"
    )

    return {
        "value": pooled,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "z_value": z_value,
        "p_value": p_value,
        "weights": weights,
        "interpretation": interpretation,
    }


def heterogeneity_stats(prepared: list[dict]) -> dict:
    """Calculate heterogeneity statistics.

    Args:
        prepared: Studies from _prepare_studies.

    Returns:
        Dict with Q statistic, Q p-value, I-squared, tau-squared, interpretation.
    """
    k = len(prepared)

    # Fixed-effect pooled theta (on analysis scale)
    total_weight = sum(s["weight"] for s in prepared)
    pooled_theta = sum(s["weight"] * s["theta"] for s in prepared) / total_weight

    # Cochran's Q
    q_statistic = sum(
        s["weight"] * (s["theta"] - pooled_theta) ** 2 for s in prepared
    )

    # Q p-value from chi-squared distribution with k-1 df
    q_p_value = float(stats.chi2.sf(q_statistic, k - 1))

    # I-squared
    if q_statistic > 0 and k > 1:
        i_squared = max(0.0, (q_statistic - (k - 1)) / q_statistic * 100)
    else:
        i_squared = 0.0

    # Tau-squared (DerSimonian-Laird estimator)
    c = total_weight - sum(s["weight"] ** 2 for s in prepared) / total_weight
    if c > 0:
        tau_squared = max(0.0, (q_statistic - (k - 1)) / c)
    else:
        tau_squared = 0.0

    interpretation = interpret_heterogeneity(i_squared, q_p_value, k)

    return {
        "q_statistic": q_statistic,
        "q_p_value": q_p_value,
        "i_squared": i_squared,
        "tau_squared": tau_squared,
        "interpretation": interpretation,
    }


def random_effects_meta(
    prepared: list[dict], tau_squared: float, measure_type: str
) -> dict:
    """DerSimonian-Laird random-effects meta-analysis.

    Args:
        prepared: Studies from _prepare_studies.
        tau_squared: Between-study variance from heterogeneity_stats.
        measure_type: Measure type string.

    Returns:
        Dict with pooled estimate, CI, z-value, p-value, weights,
        tau_squared, prediction_interval.
    """
    is_ratio = measure_type in RATIO_MEASURES
    k = len(prepared)

    # Random-effects weights (epsilon floor prevents division by zero)
    re_weights = [1.0 / max(s["variance"] + tau_squared, 1e-10) for s in prepared]
    total_re_weight = sum(re_weights)

    pooled_theta = sum(w * s["theta"] for w, s in zip(re_weights, prepared)) / total_re_weight
    se_pooled = 1.0 / math.sqrt(total_re_weight)

    ci_lower_theta = pooled_theta - Z_SCORE_95 * se_pooled
    ci_upper_theta = pooled_theta + Z_SCORE_95 * se_pooled

    z_value = pooled_theta / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z_value)))

    # Prediction interval (where future studies might fall)
    pi_is_fallback = False
    if k > 2:
        t_crit = float(stats.t.ppf(0.975, k - 2))
        pi_se = math.sqrt(se_pooled ** 2 + tau_squared)
        pi_lower_theta = pooled_theta - t_crit * pi_se
        pi_upper_theta = pooled_theta + t_crit * pi_se
    else:
        pi_lower_theta = ci_lower_theta
        pi_upper_theta = ci_upper_theta
        pi_is_fallback = True

    # Back-transform
    if is_ratio:
        try:
            pooled = math.exp(pooled_theta)
            ci_lower = math.exp(ci_lower_theta)
            ci_upper = math.exp(ci_upper_theta)
            pi_lower = math.exp(pi_lower_theta)
            pi_upper = math.exp(pi_upper_theta)
        except OverflowError:
            raise ValueError(
                "Effect sizes too extreme for back-transformation. "
                "Check that input values are on the correct scale."
            )
    else:
        pooled = pooled_theta
        ci_lower = ci_lower_theta
        ci_upper = ci_upper_theta
        pi_lower = pi_lower_theta
        pi_upper = pi_upper_theta

    # Percentage weights
    weights = [(w / total_re_weight) * 100 for w in re_weights]

    interpretation = interpret_meta_analysis(
        pooled, ci_lower, ci_upper, measure_type, "random"
    )

    return {
        "value": pooled,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "z_value": z_value,
        "p_value": p_value,
        "weights": weights,
        "tau_squared": tau_squared,
        "prediction_interval": (pi_lower, pi_upper),
        "prediction_interval_note": (
            "Prediction interval requires >2 studies; showing CI instead."
            if pi_is_fallback else None
        ),
        "interpretation": interpretation,
    }


def run_meta_analysis(
    studies: list[dict],
    measure_type: str,
    model: str = "both",
) -> dict:
    """Run a complete meta-analysis.

    Args:
        studies: List of study dicts with keys: name, effect, ci_lower, ci_upper.
        measure_type: Measure type (e.g., "OR", "RR", "HR", "MD", "beta").
        model: "fixed", "random", or "both".

    Returns:
        Dict with studies, fixed/random results, heterogeneity, measure info.
    """
    # Validate model and measure_type parameters
    valid_models = {"fixed", "random", "both"}
    if model not in valid_models:
        return {"errors": [f"Invalid model '{model}'. Must be one of: {', '.join(sorted(valid_models))}"]}

    valid_measures = RATIO_MEASURES | DIFFERENCE_MEASURES
    if measure_type not in valid_measures:
        return {"errors": [f"Unknown measure type '{measure_type}'. Supported: {', '.join(sorted(valid_measures))}"]}

    is_ratio = measure_type in RATIO_MEASURES
    null_value = 1.0 if is_ratio else 0.0

    # Validate
    errors = validate_studies(studies)
    if errors:
        return {"errors": errors}

    # Validate ratio measures have positive values
    if is_ratio:
        ratio_errors = []
        for s in studies:
            if s["effect"] <= 0 or s["ci_lower"] <= 0 or s["ci_upper"] <= 0:
                ratio_errors.append(f"{s.get('name', 'Study')}: Ratio measures must be positive.")
        if ratio_errors:
            return {"errors": ratio_errors}

    # Prepare
    prepared = _prepare_studies(studies, measure_type)

    # Fixed effect
    fixed_result = None
    if model in ("fixed", "both"):
        fixed_result = fixed_effect_meta(prepared, measure_type)

    # Heterogeneity (always computed — needed for random effects)
    het = heterogeneity_stats(prepared)

    # Random effects
    random_result = None
    if model in ("random", "both"):
        random_result = random_effects_meta(prepared, het["tau_squared"], measure_type)

    return {
        "studies": prepared,
        "fixed": fixed_result,
        "random": random_result,
        "heterogeneity": het,
        "measure_type": measure_type,
        "is_ratio": is_ratio,
        "null_value": null_value,
    }
