"""E-value calculations for sensitivity analysis.

The E-value quantifies how strong unmeasured confounding would need
to be to explain away an observed association between exposure and outcome.
"""

import math

from utils.interpretations import interpret_e_value


def calculate_e_value(point_estimate: float, ci_bound: float = None) -> dict:
    """Calculate E-value for an observed effect estimate.

    The E-value is the minimum strength of association that an unmeasured
    confounder would need to have with both the exposure and outcome to
    fully explain away the observed association.

    Args:
        point_estimate: The odds ratio or risk ratio point estimate.
        ci_bound: The confidence interval bound closest to 1.0 (optional).

    Returns:
        Dictionary with e_value, e_value_ci (if ci_bound provided), and interpretation.
    """
    if point_estimate is None or point_estimate <= 0:
        return {
            "e_value": None,
            "e_value_ci": None,
            "interpretation": "Cannot calculate E-value: invalid point estimate",
        }

    # E-value formula: E = RR + sqrt(RR * (RR - 1))
    # For OR, this is an approximation that works well when outcome is rare
    # or when RR approximates OR

    # Convert estimate to be >= 1 (E-value is symmetric)
    rr = point_estimate if point_estimate >= 1 else 1 / point_estimate

    # Calculate E-value for point estimate
    e_value = rr + math.sqrt(rr * (rr - 1))

    result = {
        "e_value": round(e_value, 2),
        "e_value_ci": None,
        "interpretation": interpret_e_value(e_value),
    }

    # Calculate E-value for CI bound if provided
    if ci_bound is not None and ci_bound > 0:
        # Use the CI bound closest to 1
        if ci_bound >= 1:
            rr_ci = ci_bound
        else:
            rr_ci = 1 / ci_bound if ci_bound > 0 else None

        if rr_ci is not None and rr_ci > 1:
            e_value_ci = rr_ci + math.sqrt(rr_ci * (rr_ci - 1))
            result["e_value_ci"] = round(e_value_ci, 2)
        else:
            # CI crosses 1, so E-value for CI is 1
            result["e_value_ci"] = 1.0

    return result


def calculate_e_value_for_or(
    odds_ratio: float, ci_lower: float = None, ci_upper: float = None
) -> dict:
    """Calculate E-value specifically for an odds ratio with CI.

    Args:
        odds_ratio: The odds ratio point estimate.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.

    Returns:
        Dictionary with e_value, e_value_ci, and interpretation.
    """
    if odds_ratio is None or odds_ratio <= 0:
        return {
            "e_value": None,
            "e_value_ci": None,
            "interpretation": "Cannot calculate E-value: invalid odds ratio",
        }

    # Determine which CI bound is closer to 1
    ci_bound = None
    if ci_lower is not None and ci_upper is not None:
        if odds_ratio >= 1:
            ci_bound = ci_lower  # For protective, use lower bound
        else:
            ci_bound = ci_upper  # For harmful, use upper bound

    return calculate_e_value(odds_ratio, ci_bound)


def calculate_confounding_strength_needed(
    observed_rr: float, true_rr: float = 1.0
) -> dict:
    """Calculate the confounding strength needed to shift observed RR to true RR.

    This helps understand how strong a confounder would need to be
    to reduce an observed association to a specific value.

    Args:
        observed_rr: The observed risk ratio or odds ratio.
        true_rr: The hypothesized true RR if confounding is removed (default 1.0).

    Returns:
        Dictionary with required_strength and interpretation.
    """
    if observed_rr is None or observed_rr <= 0:
        return {
            "required_strength": None,
            "interpretation": "Cannot calculate: invalid observed estimate",
        }

    if true_rr <= 0:
        return {
            "required_strength": None,
            "interpretation": "Cannot calculate: invalid true estimate",
        }

    # For the confounder to shift observed_rr to true_rr,
    # it needs association strength of observed_rr / true_rr with both
    # exposure and outcome
    bias_factor = observed_rr / true_rr if observed_rr >= 1 else true_rr / observed_rr

    if bias_factor <= 1:
        return {
            "required_strength": 1.0,
            "interpretation": "No confounding needed to explain the difference.",
        }

    # Required association with both exposure and outcome
    # Using the E-value formula in reverse
    required_strength = bias_factor + math.sqrt(bias_factor * (bias_factor - 1))

    interpretation = (
        f"To shift OR from {observed_rr:.2f} to {true_rr:.2f}, "
        f"an unmeasured confounder would need associations of at least "
        f"{required_strength:.1f} with both exposure and outcome."
    )

    return {
        "required_strength": round(required_strength, 2),
        "interpretation": interpretation,
    }
