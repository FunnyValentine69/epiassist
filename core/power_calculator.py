"""Power and sample size calculations for study design.

This module provides functions to calculate statistical power,
required sample sizes, and generate power curves.
"""

import math

import pandas as pd
from scipy import stats

from utils.constants import ALPHA_DEFAULT, POWER_DEFAULT


def calculate_sample_size(
    effect_size: float, alpha: float = ALPHA_DEFAULT, power: float = POWER_DEFAULT
) -> int:
    """Calculate required sample size for detecting an effect.

    Uses the formula for comparing two proportions.

    Args:
        effect_size: Cohen's h effect size (0.2 small, 0.5 medium, 0.8 large).
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).

    Returns:
        Required sample size per group.
    """
    # Get z-scores for alpha and power
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    # Sample size formula: n = 2 * ((z_alpha + z_beta) / effect_size)^2
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return math.ceil(n)


def calculate_power(
    n: int, effect_size: float, alpha: float = ALPHA_DEFAULT
) -> float:
    """Calculate statistical power given sample size and effect size.

    Args:
        n: Sample size per group.
        effect_size: Cohen's h effect size.
        alpha: Significance level (default 0.05).

    Returns:
        Statistical power (probability of detecting effect if real).
    """
    if n <= 0 or effect_size <= 0:
        return 0.0

    # Get z-score for alpha
    z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Calculate z_beta from sample size formula
    # n = 2 * ((z_alpha + z_beta) / effect_size)^2
    # Solving for z_beta:
    z_beta = effect_size * math.sqrt(n / 2) - z_alpha

    # Convert z_beta to power
    power = stats.norm.cdf(z_beta)

    return min(max(power, 0.0), 1.0)  # Clamp to [0, 1]


def generate_power_curve(
    effect_size: float,
    alpha: float = ALPHA_DEFAULT,
    n_range: tuple[int, int] = (10, 500),
) -> pd.DataFrame:
    """Generate power curve data for a range of sample sizes.

    Args:
        effect_size: Cohen's h effect size.
        alpha: Significance level (default 0.05).
        n_range: Tuple of (min_n, max_n) for sample size range.

    Returns:
        DataFrame with columns 'n' and 'power'.
    """
    n_values = list(range(n_range[0], n_range[1] + 1, max(1, (n_range[1] - n_range[0]) // 50)))
    power_values = [calculate_power(n, effect_size, alpha) for n in n_values]

    return pd.DataFrame({"n": n_values, "power": power_values})


def calculate_sample_size_for_or(
    p0: float,
    odds_ratio: float,
    alpha: float = ALPHA_DEFAULT,
    power: float = POWER_DEFAULT,
    ratio: float = 1.0,
) -> dict:
    """Calculate sample size for detecting a specific odds ratio.

    Args:
        p0: Baseline probability in unexposed group.
        odds_ratio: Expected odds ratio to detect.
        alpha: Significance level (default 0.05).
        power: Desired statistical power (default 0.80).
        ratio: Ratio of unexposed to exposed (n0/n1).

    Returns:
        Dictionary with n_exposed, n_unexposed, n_total.
    """
    # Convert OR to p1
    odds0 = p0 / (1 - p0)
    odds1 = odds0 * odds_ratio
    p1 = odds1 / (1 + odds1)

    # Effect size using arcsine transformation (Cohen's h)
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))
    h = abs(h)

    if h == 0:
        return {
            "n_exposed": None,
            "n_unexposed": None,
            "n_total": None,
            "interpretation": "Cannot calculate: effect size is zero",
        }

    # Get z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Sample size formula for unequal groups
    p_bar = (p0 + p1) / 2
    q_bar = 1 - p_bar

    numerator = (z_alpha * math.sqrt(2 * p_bar * q_bar) + z_beta * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2
    denominator = (p1 - p0) ** 2

    n1 = numerator / denominator
    n0 = n1 * ratio

    n_exposed = math.ceil(n1)
    n_unexposed = math.ceil(n0)

    interpretation = (
        f"To detect OR = {odds_ratio:.2f} with {power*100:.0f}% power at α = {alpha}, "
        f"you need {n_exposed} exposed and {n_unexposed} unexposed participants "
        f"(total N = {n_exposed + n_unexposed})."
    )

    return {
        "n_exposed": n_exposed,
        "n_unexposed": n_unexposed,
        "n_total": n_exposed + n_unexposed,
        "interpretation": interpretation,
    }


def effect_size_from_proportions(p1: float, p0: float) -> float:
    """Calculate Cohen's h effect size from two proportions.

    Args:
        p1: Proportion in group 1.
        p0: Proportion in group 2.

    Returns:
        Cohen's h effect size.
    """
    h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p0)))
    return abs(h)


def classify_effect_size(h: float) -> str:
    """Classify Cohen's h effect size.

    Args:
        h: Cohen's h value.

    Returns:
        Classification string (small, medium, large).
    """
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"
