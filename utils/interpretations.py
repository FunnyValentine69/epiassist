"""Natural language interpretations for statistical results.

This module provides functions that generate plain English explanations
of statistical measures for non-statisticians.
"""

from utils.constants import (
    ALPHA_DEFAULT,
    I_SQUARED_THRESHOLDS,
    META_MEASURE_LABELS,
    RATIO_MEASURES,
    SMD_BALANCE_THRESHOLD,
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
    if ci_lower > 1 or ci_upper < 1:
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


def interpret_mantel_haenszel(
    or_value: float,
    or_ci_lower: float,
    or_ci_upper: float,
    homogeneity_p: float | None,
    n_strata: int,
    confounder_name: str,
) -> str:
    """Generate plain English interpretation of Mantel-Haenszel adjusted results.

    Args:
        or_value: MH-adjusted odds ratio.
        or_ci_lower: Lower bound of 95% CI for adjusted OR.
        or_ci_upper: Upper bound of 95% CI for adjusted OR.
        homogeneity_p: P-value from Breslow-Day test (None if < 2 strata).
        n_strata: Number of valid strata used.
        confounder_name: Name of the stratifying confounder.

    Returns:
        Plain English interpretation string.
    """
    # Direction and magnitude
    if or_value > 1:
        direction = "higher"
        percent_change = (or_value - 1) * 100
    elif or_value < 1:
        direction = "lower"
        percent_change = (1 - or_value) * 100
    else:
        direction = "equal"
        percent_change = 0.0

    # Significance
    if or_ci_lower > 1 or or_ci_upper < 1:
        significance = "statistically significant (CI excludes 1.0)"
    else:
        significance = "NOT statistically significant (CI includes 1.0)"

    parts = [
        f"Adjusted OR = {or_value:.2f} (95% CI: {or_ci_lower:.2f}-{or_ci_upper:.2f}), "
        f"stratified by {confounder_name} ({n_strata} strata)."
    ]

    if or_value != 1:
        parts.append(
            f"After adjusting for {confounder_name}, the exposed group has "
            f"{percent_change:.0f}% {direction} odds of the outcome. "
            f"This association is {significance}."
        )
    else:
        parts.append(
            f"After adjusting for {confounder_name}, there is no association "
            f"between exposure and outcome."
        )

    # Homogeneity assessment
    if homogeneity_p is not None:
        if homogeneity_p < ALPHA_DEFAULT:
            parts.append(
                f"Breslow-Day test (p = {homogeneity_p:.4f}) suggests the OR varies "
                f"across strata of {confounder_name} — possible effect modification. "
                f"The pooled estimate should be interpreted with caution."
            )
        else:
            parts.append(
                f"Breslow-Day test (p = {homogeneity_p:.4f}) shows no significant "
                f"variation in OR across strata — pooling is appropriate."
            )

    return " ".join(parts)


def interpret_smr(smr: float, ci_lower: float, ci_upper: float) -> str:
    """Generate plain English interpretation of a Standardized Mortality/Incidence Ratio.

    Args:
        smr: The SMR/SIR point estimate (null = 1.0).
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.

    Returns:
        Plain English interpretation string.
    """
    if smr is None:
        return "SMR/SIR could not be calculated."

    # Determine direction
    if smr > 1:
        percent_excess = (smr - 1) * 100
        direction = (
            f"The observed number of events is {percent_excess:.0f}% higher "
            f"than expected based on the reference population rates."
        )
    elif smr < 1:
        percent_deficit = (1 - smr) * 100
        direction = (
            f"The observed number of events is {percent_deficit:.0f}% lower "
            f"than expected based on the reference population rates."
        )
    else:
        return (
            "SMR/SIR = 1.00. The observed number of events exactly matches "
            "the expected number based on the reference population."
        )

    # Check statistical significance (CI excludes 1.0)
    if ci_lower > 1 or ci_upper < 1:
        significance = "This is statistically significant (CI excludes 1.0)."
    else:
        significance = "This is NOT statistically significant (CI includes 1.0)."

    return (
        f"SMR/SIR = {smr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}). "
        f"{direction} {significance}"
    )


def interpret_direct_standardized_rate(
    adjusted_rate: float,
    ci_lower: float,
    ci_upper: float,
    crude_rate: float,
    multiplier: int,
) -> str:
    """Generate plain English interpretation of a directly standardized rate.

    Args:
        adjusted_rate: The age-adjusted rate (per multiplier).
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.
        crude_rate: The unadjusted crude rate (per multiplier).
        multiplier: The rate multiplier (e.g. 100000 for "per 100,000").

    Returns:
        Plain English interpretation string.
    """
    multiplier_label = f"per {multiplier:,}"

    # Compare adjusted to crude
    if crude_rate < 1e-10:
        comparison = (
            "The crude rate is zero, so the adjusted rate reflects the "
            "weighted contribution of stratum-specific rates to the standard population."
        )
    else:
        pct_diff = ((adjusted_rate - crude_rate) / crude_rate) * 100
        abs_pct = abs(pct_diff)

        if abs_pct < 5:
            comparison = (
                f"The adjusted rate ({adjusted_rate:.2f}) is similar to the crude rate "
                f"({crude_rate:.2f}), suggesting the age distribution of the study "
                f"population is close to the standard population."
            )
        elif pct_diff > 0:
            comparison = (
                f"The adjusted rate ({adjusted_rate:.2f}) is {abs_pct:.0f}% higher than the "
                f"crude rate ({crude_rate:.2f}), indicating that the study population "
                f"has a younger age structure compared to the standard population."
            )
        else:
            comparison = (
                f"The adjusted rate ({adjusted_rate:.2f}) is {abs_pct:.0f}% lower than the "
                f"crude rate ({crude_rate:.2f}), indicating that the study population "
                f"has an older age structure compared to the standard population."
            )

    return (
        f"Age-adjusted rate = {adjusted_rate:.2f} {multiplier_label} "
        f"(95% CI: {ci_lower:.2f}-{ci_upper:.2f}). "
        f"{comparison} "
        f"This rate is comparable to other populations standardized "
        f"to the same reference population."
    )


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


def interpret_logistic_regression(
    exposure_name: str,
    or_value: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    confounder_names: list[str],
    n_obs: int,
    weighted: bool = False,
) -> str:
    """Generate plain English interpretation of logistic regression results.

    Args:
        exposure_name: Name of the exposure variable.
        or_value: Adjusted odds ratio for exposure.
        ci_lower: Lower bound of 95% CI for OR.
        ci_upper: Upper bound of 95% CI for OR.
        p_value: P-value for the exposure coefficient.
        confounder_names: List of confounder variable names adjusted for.
        n_obs: Number of observations used in the model.

    Returns:
        Plain English interpretation string.
    """
    if or_value > 1:
        direction = f"{(or_value - 1) * 100:.0f}% higher odds"
    elif or_value < 1:
        direction = f"{(1 - or_value) * 100:.0f}% lower odds"
    else:
        direction = "no difference in odds"

    sig = "statistically significant" if p_value < ALPHA_DEFAULT else "NOT statistically significant"
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.4f}"

    if confounder_names:
        adj_text = f"adjusted for {', '.join(confounder_names)}"
    else:
        adj_text = "unadjusted (no confounders)"

    prefix = "Survey-weighted adjusted" if weighted else "Adjusted"

    return (
        f"{prefix} OR = {or_value:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}, "
        f"p {p_str}), {adj_text}, based on {n_obs:,} observations. "
        f"Exposure to {exposure_name} is associated with {direction} "
        f"of the outcome. This association is {sig}."
    )


def interpret_linear_regression(
    exposure_name: str,
    beta: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    confounder_names: list[str],
    n_obs: int,
    r_squared: float,
    weighted: bool = False,
) -> str:
    """Generate plain English interpretation of linear regression results.

    Args:
        exposure_name: Name of the exposure variable.
        beta: Adjusted beta coefficient for exposure.
        ci_lower: Lower bound of 95% CI for beta.
        ci_upper: Upper bound of 95% CI for beta.
        p_value: P-value for the exposure coefficient.
        confounder_names: List of confounder variable names adjusted for.
        n_obs: Number of observations used in the model.
        r_squared: R-squared of the model.

    Returns:
        Plain English interpretation string.
    """
    if beta > 0:
        direction = f"an increase of {beta:.2f}"
    elif beta < 0:
        direction = f"a decrease of {abs(beta):.2f}"
    else:
        direction = "no change"

    sig = "statistically significant" if p_value < ALPHA_DEFAULT else "NOT statistically significant"
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.4f}"

    if confounder_names:
        adj_text = f"adjusted for {', '.join(confounder_names)}"
    else:
        adj_text = "unadjusted (no confounders)"

    prefix = "Survey-weighted adjusted" if weighted else "Adjusted"

    return (
        f"{prefix} β = {beta:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}, "
        f"p {p_str}), {adj_text}, based on {n_obs:,} observations. "
        f"A one-unit increase in {exposure_name} is associated with {direction} "
        f"in the outcome. This association is {sig}. "
        f"The model explains {r_squared * 100:.1f}% of variance (R² = {r_squared:.4f})."
    )


def interpret_poisson_regression(
    exposure_name: str,
    irr: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    confounder_names: list[str],
    n_obs: int,
    weighted: bool = False,
) -> str:
    """Generate plain English interpretation of Poisson regression results.

    Args:
        exposure_name: Name of the exposure variable.
        irr: Adjusted incidence rate ratio for exposure.
        ci_lower: Lower bound of 95% CI for IRR.
        ci_upper: Upper bound of 95% CI for IRR.
        p_value: P-value for the exposure coefficient.
        confounder_names: List of confounder variable names adjusted for.
        n_obs: Number of observations used in the model.

    Returns:
        Plain English interpretation string.
    """
    if irr > 1:
        direction = f"{(irr - 1) * 100:.0f}% higher rate"
    elif irr < 1:
        direction = f"{(1 - irr) * 100:.0f}% lower rate"
    else:
        direction = "no difference in rate"

    sig = "statistically significant" if p_value < ALPHA_DEFAULT else "NOT statistically significant"
    p_str = "< 0.001" if p_value < 0.001 else f"= {p_value:.4f}"

    if confounder_names:
        adj_text = f"adjusted for {', '.join(confounder_names)}"
    else:
        adj_text = "unadjusted (no confounders)"

    prefix = "Survey-weighted adjusted" if weighted else "Adjusted"

    return (
        f"{prefix} IRR = {irr:.2f} (95% CI: {ci_lower:.2f}-{ci_upper:.2f}, "
        f"p {p_str}), {adj_text}, based on {n_obs:,} observations. "
        f"Exposure to {exposure_name} is associated with {direction} "
        f"of the outcome. This association is {sig}."
    )


def interpret_propensity_score(
    estimand: str,
    effect_value: float,
    ci_lower: float,
    ci_upper: float,
    outcome_type: str,
    treatment_name: str,
    confounder_names: list[str],
    n_obs: int,
    effective_n: float,
    all_balanced: bool,
    n_balanced: int,
    n_total_covariates: int,
    weighted: bool = False,
) -> str:
    """Generate plain English interpretation of propensity score analysis.

    Args:
        estimand: "ATE" or "ATT".
        effect_value: Treatment effect (OR for binary, mean diff for continuous).
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.
        outcome_type: "binary" or "continuous".
        treatment_name: Name of the treatment variable.
        confounder_names: Confounder names used in PS model.
        n_obs: Number of observations.
        effective_n: Effective sample size after weighting.
        all_balanced: True if all covariates have SMD < threshold after weighting.
        n_balanced: Number of covariates with SMD < threshold after weighting.
        n_total_covariates: Total number of covariates checked.
        weighted: Whether survey weights were also applied.

    Returns:
        Plain English interpretation string.
    """
    import math

    is_binary = outcome_type == "binary"
    measure_label = "OR" if is_binary else "Mean Difference"
    prefix = "Survey-weighted IPTW-adjusted" if weighted else "IPTW-adjusted"

    # Guard against NaN inputs from bootstrap failure
    if not math.isfinite(effect_value) or not math.isfinite(ci_lower) or not math.isfinite(ci_upper):
        return (
            f"{prefix} analysis could not produce reliable estimates. "
            f"Bootstrap confidence intervals failed to converge. "
            f"Consider simplifying the model, trimming extreme propensity scores, "
            f"or checking data quality."
        )

    # Estimand explanation
    if estimand == "ATE":
        estimand_desc = "the average treatment effect in the full population"
    else:
        estimand_desc = "the average treatment effect among the treated"

    # Direction and magnitude
    if is_binary:
        if effect_value > 1:
            direction = f"{(effect_value - 1) * 100:.0f}% higher odds"
        elif effect_value < 1:
            direction = f"{(1 - effect_value) * 100:.0f}% lower odds"
        else:
            direction = "no difference in odds"
        significant = ci_lower > 1.0 or ci_upper < 1.0
    else:
        if effect_value > 0:
            direction = f"an increase of {effect_value:.2f}"
        elif effect_value < 0:
            direction = f"a decrease of {abs(effect_value):.2f}"
        else:
            direction = "no change"
        significant = ci_lower > 0.0 or ci_upper < 0.0

    sig = "statistically significant" if significant else "NOT statistically significant"

    # Confounder list
    if confounder_names:
        adj_text = f"adjusted for {', '.join(confounder_names)}"
    else:
        adj_text = "using propensity score weights"

    # Balance quality
    if all_balanced:
        balance_text = (
            f"All {n_total_covariates} covariates achieved adequate balance "
            f"(SMD < {SMD_BALANCE_THRESHOLD}) after IPTW weighting."
        )
    else:
        balance_text = (
            f"{n_balanced} of {n_total_covariates} covariates achieved adequate balance "
            f"(SMD < {SMD_BALANCE_THRESHOLD}) after IPTW weighting. "
            f"Residual imbalance may bias the treatment effect estimate."
        )

    # Effective N note
    eff_ratio = effective_n / n_obs if n_obs > 0 else 0
    if eff_ratio < 0.5:
        eff_note = (
            f"The effective sample size ({effective_n:.0f}) is substantially smaller "
            f"than the actual sample ({n_obs:,}), indicating high weight variability. "
            f"Consider trimming extreme propensity scores."
        )
    else:
        eff_note = f"Effective sample size: {effective_n:.0f} of {n_obs:,} observations."

    return (
        f"{prefix} {measure_label} = {effect_value:.2f} "
        f"(95% CI: {ci_lower:.2f}-{ci_upper:.2f}), "
        f"estimating {estimand_desc} ({estimand}), "
        f"{adj_text}, based on {n_obs:,} observations. "
        f"Exposure to {treatment_name} is associated with {direction} "
        f"of the outcome. This association is {sig}. "
        f"{balance_text} {eff_note}"
    )


def interpret_mediation(
    mediator_name: str,
    exposure_name: str,
    outcome_name: str,
    indirect: float,
    direct: float,
    total: float,
    indirect_ci: tuple[float, float],
    direct_ci: tuple[float, float],
    sobel_p: float | None,
    proportion_mediated: float | None,
    method: str,
    n_obs: int,
    confounder_names: list[str],
    weighted: bool = False,
) -> str:
    """Generate plain English interpretation of mediation analysis.

    Args:
        mediator_name: Name of the mediator variable.
        exposure_name: Name of the exposure variable.
        outcome_name: Name of the outcome variable.
        indirect: Indirect effect (a*b or c-c').
        direct: Direct effect (c').
        total: Total effect (c).
        indirect_ci: 95% CI for indirect effect (lower, upper).
        direct_ci: 95% CI for direct effect (lower, upper).
        sobel_p: Sobel test p-value (None for binary outcomes).
        proportion_mediated: Proportion mediated (None when signs differ).
        method: "product" or "difference".
        n_obs: Number of observations.
        confounder_names: Confounders adjusted for.
        weighted: Whether survey weights were applied.

    Returns:
        Plain English interpretation string.
    """
    prefix = "Survey-weighted Baron-Kenny" if weighted else "Baron-Kenny"

    # Indirect effect significance via bootstrap CI
    indirect_sig = indirect_ci[0] > 0 or indirect_ci[1] < 0

    # Direct effect significance via bootstrap CI
    direct_sig = direct_ci[0] > 0 or direct_ci[1] < 0

    # Classification
    if indirect_sig and not direct_sig:
        class_desc = (
            f"The indirect effect through {mediator_name} is statistically significant "
            f"while the direct effect is not, suggesting full mediation."
        )
    elif indirect_sig and direct_sig:
        class_desc = (
            f"Both the indirect effect through {mediator_name} and the direct effect "
            f"are statistically significant, suggesting partial mediation."
        )
    else:
        class_desc = (
            f"The indirect effect through {mediator_name} is not statistically significant, "
            f"suggesting no evidence of mediation."
        )

    # Proportion mediated text
    if proportion_mediated is not None and indirect_sig:
        prop_text = (
            f"Approximately {proportion_mediated * 100:.1f}% of the total effect "
            f"is mediated through {mediator_name}."
        )
    elif proportion_mediated is None and indirect_sig:
        prop_text = (
            "Proportion mediated could not be calculated because the indirect "
            "and total effects have different signs."
        )
    else:
        prop_text = ""

    # Sobel test text
    if sobel_p is not None:
        if sobel_p < ALPHA_DEFAULT:
            sobel_text = (
                f"The Sobel test confirms the indirect effect is statistically significant "
                f"(p = {sobel_p:.4f})."
            )
        else:
            sobel_text = (
                f"The Sobel test indicates the indirect effect is not statistically significant "
                f"(p = {sobel_p:.4f})."
            )
    else:
        sobel_text = (
            "The Sobel test is not applicable for binary outcomes; "
            "bootstrap confidence intervals are used instead."
        )

    # Adjustment text
    if confounder_names:
        adj_text = f"adjusted for {', '.join(confounder_names)}"
    else:
        adj_text = "unadjusted"

    # Method note
    if method == "difference":
        method_note = (
            "The difference method (c - c') was used for the indirect effect "
            "because the outcome is binary."
        )
    else:
        method_note = (
            "The product of coefficients method (a x b) was used for the indirect effect."
        )

    parts = [
        f"{prefix} mediation analysis ({adj_text}, n = {n_obs:,}) "
        f"examined whether {mediator_name} mediates the effect of "
        f"{exposure_name} on {outcome_name}.",
        f"Total effect = {total:.4f}, Direct effect = {direct:.4f} "
        f"(95% CI: {direct_ci[0]:.4f} to {direct_ci[1]:.4f}), "
        f"Indirect effect = {indirect:.4f} "
        f"(95% CI: {indirect_ci[0]:.4f} to {indirect_ci[1]:.4f}).",
        class_desc,
    ]

    if prop_text:
        parts.append(prop_text)
    parts.append(sobel_text)
    parts.append(method_note)

    return " ".join(parts)
