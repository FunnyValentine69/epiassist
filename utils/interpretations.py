"""Natural language interpretations for statistical results.

This module provides functions that generate plain English explanations
of statistical measures for non-statisticians.
"""

from utils.constants import (
    ALPHA_DEFAULT,
    I_SQUARED_THRESHOLDS,
    META_MEASURE_LABELS,
    RATIO_MEASURES,
)


def interpret_odds_ratio(
    or_value: float, ci_lower: float, ci_upper: float
) -> str:
    """Generate plain English interpretation of an odds ratio.

    Args:
        or_value: The odds ratio point estimate.
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.

    Returns:
        Plain English interpretation string.
    """
    if or_value is None:
        return "Odds ratio could not be calculated."

    # Determine direction and magnitude
    if or_value > 1:
        direction = "higher"
        magnitude = or_value
        percent_change = (or_value - 1) * 100
    elif or_value < 1:
        direction = "lower"
        magnitude = 1 / or_value
        percent_change = (1 - or_value) * 100
    else:
        return "The odds ratio is 1.0, indicating no association between exposure and outcome."

    # Check statistical significance
    if ci_lower > 1:
        significance = "This association is statistically significant (CI excludes 1.0)."
    elif ci_upper < 1:
        significance = "This association is statistically significant (CI excludes 1.0)."
    else:
        significance = "This association is NOT statistically significant (CI includes 1.0)."

    # Build interpretation
    interpretation = (
        f"OR = {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}). "
        f"The exposed group has {magnitude:.1f}x {direction} odds of the outcome "
        f"compared to the unexposed group ({percent_change:.0f}% {'increase' if or_value > 1 else 'decrease'}). "
        f"{significance}"
    )

    return interpretation


def interpret_risk_ratio(
    rr_value: float, ci_lower: float, ci_upper: float
) -> str:
    """Generate plain English interpretation of a risk ratio.

    Args:
        rr_value: The risk ratio point estimate.
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.

    Returns:
        Plain English interpretation string.
    """
    if rr_value is None:
        return "Risk ratio could not be calculated."

    # Determine direction
    if rr_value > 1:
        direction = "higher"
        percent_change = (rr_value - 1) * 100
    elif rr_value < 1:
        direction = "lower"
        percent_change = (1 - rr_value) * 100
    else:
        return "The risk ratio is 1.0, indicating no difference in risk between groups."

    # Check statistical significance
    if ci_lower > 1 or ci_upper < 1:
        significance = "statistically significant"
    else:
        significance = "NOT statistically significant"

    interpretation = (
        f"RR = {rr_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}). "
        f"The exposed group has {percent_change:.0f}% {direction} risk of the outcome "
        f"compared to the unexposed group. "
        f"This association is {significance}."
    )

    return interpretation


def interpret_p_value(p: float, alpha: float = ALPHA_DEFAULT) -> str:
    """Generate plain English interpretation of a p-value.

    Args:
        p: The p-value.
        alpha: Significance threshold (default 0.05).

    Returns:
        Plain English interpretation string.
    """
    if p < 0.001:
        p_str = "< 0.001"
    else:
        p_str = f"= {p:.4f}"

    if p < alpha:
        interpretation = (
            f"p-value {p_str}. The result is statistically significant at alpha = {alpha}. "
            f"There is strong evidence against the null hypothesis."
        )
    else:
        interpretation = (
            f"p-value {p_str}. The result is NOT statistically significant at alpha = {alpha}. "
            f"There is insufficient evidence to reject the null hypothesis."
        )

    return interpretation


def interpret_power(power: float) -> str:
    """Generate plain English interpretation of statistical power.

    Args:
        power: The statistical power (probability of detecting effect if real).

    Returns:
        Plain English interpretation string.
    """
    power_pct = power * 100

    if power >= 0.80:
        adequacy = "adequate"
        recommendation = "This study is well-powered to detect the specified effect."
    elif power >= 0.60:
        adequacy = "marginally adequate"
        recommendation = "Consider increasing sample size for more reliable detection."
    else:
        adequacy = "inadequate"
        recommendation = "A larger sample size is strongly recommended."

    interpretation = (
        f"Power = {power_pct:.0f}%. "
        f"This power level is {adequacy}. "
        f"{recommendation} "
        f"There is a {power_pct:.0f}% chance of detecting a true effect if one exists."
    )

    return interpretation


def interpret_e_value(e_value: float) -> str:
    """Generate plain English interpretation of an E-value.

    The E-value quantifies how strong unmeasured confounding would need
    to be to explain away the observed association.

    Args:
        e_value: The E-value.

    Returns:
        Plain English interpretation string.
    """
    if e_value is None:
        return "E-value could not be calculated."

    if e_value >= 10:
        robustness = "very robust"
        explanation = (
            "Unmeasured confounding would need to be extremely strong "
            "to explain away this association."
        )
    elif e_value >= 5:
        robustness = "quite robust"
        explanation = (
            "It would take a fairly strong unmeasured confounder "
            "to fully explain this association."
        )
    elif e_value >= 3:
        robustness = "moderately robust"
        explanation = (
            "A moderately strong unmeasured confounder could potentially "
            "explain this association."
        )
    elif e_value >= 2:
        robustness = "somewhat vulnerable"
        explanation = (
            "A relatively weak unmeasured confounder might be sufficient "
            "to explain this association."
        )
    else:
        robustness = "vulnerable"
        explanation = (
            "Even a weak unmeasured confounder could explain away "
            "this association."
        )

    interpretation = (
        f"E-value = {e_value:.2f}. "
        f"This result is {robustness} to unmeasured confounding. "
        f"{explanation} "
        f"An unmeasured confounder would need associations of at least {e_value:.1f} "
        f"with both the exposure and outcome to fully explain the observed effect."
    )

    return interpretation


def interpret_heterogeneity(
    i_squared: float, q_p_value: float, num_studies: int
) -> str:
    """Generate plain English interpretation of heterogeneity statistics.

    Args:
        i_squared: I-squared statistic (0-100).
        q_p_value: P-value from Cochran's Q test.
        num_studies: Number of studies in the meta-analysis.

    Returns:
        Plain English interpretation string.
    """
    if i_squared < I_SQUARED_THRESHOLDS["low"]:
        level = "low"
        desc = "Studies show consistent results."
    elif i_squared < I_SQUARED_THRESHOLDS["moderate"]:
        level = "moderate"
        desc = "Some variability between studies exists."
    elif i_squared < I_SQUARED_THRESHOLDS["high"]:
        level = "substantial"
        desc = "Considerable variability between studies. Consider exploring sources of heterogeneity."
    else:
        level = "considerable"
        desc = "High variability between studies. Results should be interpreted with caution."

    q_sig = "significant" if q_p_value < 0.10 else "not significant"

    interpretation = (
        f"I² = {i_squared:.1f}% ({level} heterogeneity). "
        f"{desc} "
        f"Cochran's Q test is {q_sig} (p = {q_p_value:.4f}), "
        f"based on {num_studies} studies."
    )

    return interpretation


def interpret_meta_analysis(
    pooled: float,
    ci_lower: float,
    ci_upper: float,
    measure_type: str,
    model: str,
) -> str:
    """Generate plain English interpretation of a pooled meta-analysis result.

    Args:
        pooled: Pooled point estimate (on natural scale).
        ci_lower: Lower bound of 95% CI (natural scale).
        ci_upper: Upper bound of 95% CI (natural scale).
        measure_type: Measure type (e.g., "OR", "RR", "MD").
        model: Model used ("fixed" or "random").

    Returns:
        Plain English interpretation string.
    """
    label = META_MEASURE_LABELS.get(measure_type, measure_type)
    model_name = "fixed-effect" if model == "fixed" else "random-effects"
    is_ratio = measure_type in RATIO_MEASURES
    null_value = 1.0 if is_ratio else 0.0

    # Determine significance
    if is_ratio:
        significant = ci_lower > 1.0 or ci_upper < 1.0
    else:
        significant = ci_lower > 0.0 or ci_upper < 0.0

    sig_text = "statistically significant" if significant else "NOT statistically significant"

    # Describe direction
    if is_ratio:
        if pooled > 1.0:
            direction = f"a {(pooled - 1) * 100:.0f}% increase"
        elif pooled < 1.0:
            direction = f"a {(1 - pooled) * 100:.0f}% decrease"
        else:
            direction = "no effect"
    else:
        if pooled > 0:
            direction = f"an increase of {pooled:.2f}"
        elif pooled < 0:
            direction = f"a decrease of {abs(pooled):.2f}"
        else:
            direction = "no effect"

    interpretation = (
        f"Pooled {label} = {pooled:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}) "
        f"using a {model_name} model. "
        f"The pooled estimate suggests {direction}. "
        f"This result is {sig_text} (CI {'excludes' if significant else 'includes'} {null_value})."
    )

    return interpretation
