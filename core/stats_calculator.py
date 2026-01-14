"""Statistical calculations for epidemiological measures.

This module provides functions to calculate odds ratios, risk ratios,
risk differences, and chi-square tests from 2x2 contingency tables.
"""

import math

from scipy import stats

from utils.constants import CI_LEVEL_DEFAULT, Z_SCORE_95


def calculate_confidence_interval(
    estimate: float, se: float, level: float = CI_LEVEL_DEFAULT
) -> tuple[float, float]:
    """Calculate confidence interval for an estimate.

    Args:
        estimate: The point estimate.
        se: Standard error of the estimate.
        level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    z = stats.norm.ppf(1 - (1 - level) / 2)
    margin = z * se
    return (estimate - margin, estimate + margin)


def calculate_odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    """Calculate odds ratio from a 2x2 table.

    Table layout:
                    Outcome+  Outcome-
        Exposed        a         b
        Unexposed      c         d

    Args:
        a: Exposed with outcome (top-left cell).
        b: Exposed without outcome (top-right cell).
        c: Unexposed with outcome (bottom-left cell).
        d: Unexposed without outcome (bottom-right cell).

    Returns:
        Dictionary with value, ci_lower, ci_upper, interpretation, and p_value.
    """
    # Handle zero cells by adding 0.5 (Haldane-Anscombe correction)
    a_adj = a + 0.5 if a == 0 or b == 0 or c == 0 or d == 0 else a
    b_adj = b + 0.5 if a == 0 or b == 0 or c == 0 or d == 0 else b
    c_adj = c + 0.5 if a == 0 or b == 0 or c == 0 or d == 0 else c
    d_adj = d + 0.5 if a == 0 or b == 0 or c == 0 or d == 0 else d

    # Calculate OR
    or_value = (a_adj * d_adj) / (b_adj * c_adj)

    # Calculate SE of log(OR)
    se_log_or = math.sqrt(1 / a_adj + 1 / b_adj + 1 / c_adj + 1 / d_adj)

    # Calculate 95% CI on log scale, then exponentiate
    log_or = math.log(or_value)
    ci_log = calculate_confidence_interval(log_or, se_log_or)
    ci_lower = math.exp(ci_log[0])
    ci_upper = math.exp(ci_log[1])

    # Calculate p-value using chi-square
    chi_result = calculate_chi_square(a, b, c, d)
    p_value = chi_result["p_value"]

    # Generate interpretation
    from utils.interpretations import interpret_odds_ratio

    interpretation = interpret_odds_ratio(or_value, ci_lower, ci_upper)

    return {
        "value": round(or_value, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "interpretation": interpretation,
        "p_value": p_value,
        "se": round(se_log_or, 4),
    }


def calculate_risk_ratio(a: int, b: int, c: int, d: int) -> dict:
    """Calculate risk ratio (relative risk) from a 2x2 table.

    Args:
        a: Exposed with outcome.
        b: Exposed without outcome.
        c: Unexposed with outcome.
        d: Unexposed without outcome.

    Returns:
        Dictionary with value, ci_lower, ci_upper, interpretation.
    """
    # Calculate risks
    n1 = a + b  # Total exposed
    n0 = c + d  # Total unexposed

    if n1 == 0 or n0 == 0:
        return {
            "value": None,
            "ci_lower": None,
            "ci_upper": None,
            "interpretation": "Cannot calculate: division by zero",
        }

    risk_exposed = a / n1
    risk_unexposed = c / n0

    if risk_unexposed == 0:
        return {
            "value": None,
            "ci_lower": None,
            "ci_upper": None,
            "interpretation": "Cannot calculate: unexposed risk is zero",
        }

    rr_value = risk_exposed / risk_unexposed

    # Calculate SE of log(RR)
    # SE(log(RR)) = sqrt((1-p1)/(n1*p1) + (1-p0)/(n0*p0))
    if risk_exposed == 0:
        se_log_rr = float("inf")
    else:
        se_log_rr = math.sqrt(
            (1 - risk_exposed) / (n1 * risk_exposed)
            + (1 - risk_unexposed) / (n0 * risk_unexposed)
        )

    # Calculate 95% CI
    if se_log_rr == float("inf"):
        ci_lower, ci_upper = 0, float("inf")
    else:
        log_rr = math.log(rr_value)
        ci_log = calculate_confidence_interval(log_rr, se_log_rr)
        ci_lower = math.exp(ci_log[0])
        ci_upper = math.exp(ci_log[1])

    # Generate interpretation
    from utils.interpretations import interpret_risk_ratio

    interpretation = interpret_risk_ratio(rr_value, ci_lower, ci_upper)

    return {
        "value": round(rr_value, 3),
        "ci_lower": round(ci_lower, 3),
        "ci_upper": round(ci_upper, 3),
        "interpretation": interpretation,
    }


def calculate_risk_difference(a: int, b: int, c: int, d: int) -> dict:
    """Calculate risk difference (attributable risk) from a 2x2 table.

    Args:
        a: Exposed with outcome.
        b: Exposed without outcome.
        c: Unexposed with outcome.
        d: Unexposed without outcome.

    Returns:
        Dictionary with value, ci_lower, ci_upper, interpretation.
    """
    n1 = a + b  # Total exposed
    n0 = c + d  # Total unexposed

    if n1 == 0 or n0 == 0:
        return {
            "value": None,
            "ci_lower": None,
            "ci_upper": None,
            "interpretation": "Cannot calculate: division by zero",
        }

    risk_exposed = a / n1
    risk_unexposed = c / n0

    rd_value = risk_exposed - risk_unexposed

    # Calculate SE of RD
    # SE(RD) = sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)
    se_rd = math.sqrt(
        (risk_exposed * (1 - risk_exposed)) / n1
        + (risk_unexposed * (1 - risk_unexposed)) / n0
    )

    # Calculate 95% CI
    ci_lower, ci_upper = calculate_confidence_interval(rd_value, se_rd)

    # Generate interpretation
    interpretation = (
        f"The risk difference is {rd_value:.1%}. "
        f"The exposed group has a {abs(rd_value):.1%} "
        f"{'higher' if rd_value > 0 else 'lower'} risk than the unexposed group."
    )

    return {
        "value": round(rd_value, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "interpretation": interpretation,
    }


def calculate_chi_square(a: int, b: int, c: int, d: int) -> dict:
    """Calculate chi-square test for a 2x2 table.

    Args:
        a: Exposed with outcome.
        b: Exposed without outcome.
        c: Unexposed with outcome.
        d: Unexposed without outcome.

    Returns:
        Dictionary with chi2 statistic, p_value, df, and interpretation.
    """
    # Create observed table
    observed = [[a, b], [c, d]]

    # Calculate chi-square
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Generate interpretation
    from utils.interpretations import interpret_p_value

    interpretation = interpret_p_value(p_value)

    return {
        "value": round(chi2, 3),
        "p_value": p_value,
        "df": dof,
        "interpretation": interpretation,
        "expected": expected,
    }


def calculate_nnt(risk_difference: float) -> dict:
    """Calculate Number Needed to Treat (or Harm) from risk difference.

    Args:
        risk_difference: The absolute risk difference.

    Returns:
        Dictionary with NNT value and interpretation.
    """
    if risk_difference == 0:
        return {
            "value": None,
            "interpretation": "Risk difference is zero; NNT is undefined.",
        }

    nnt = 1 / abs(risk_difference)

    if risk_difference > 0:
        interpretation = (
            f"Number Needed to Harm (NNH): {nnt:.1f}. "
            f"For every {nnt:.0f} people exposed, one additional person "
            "experiences the outcome compared to unexposed."
        )
    else:
        interpretation = (
            f"Number Needed to Treat (NNT): {nnt:.1f}. "
            f"For every {nnt:.0f} people exposed, one fewer person "
            "experiences the outcome compared to unexposed."
        )

    return {
        "value": round(nnt, 2),
        "interpretation": interpretation,
    }
