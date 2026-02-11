"""Standardized Mortality/Incidence Ratio (SMR/SIR) calculator.

Compares observed events in a study population to expected events
based on reference population rates. SMR = Observed / Expected (null = 1.0).
Confidence intervals use the exact Poisson method via chi-squared distribution.
"""

from scipy.stats import chi2

from utils.interpretations import interpret_smr


def calculate_smr(
    observed: int, expected: float, ci_level: float = 0.95
) -> dict:
    """Calculate the Standardized Mortality/Incidence Ratio.

    Args:
        observed: Number of observed events (non-negative integer).
        expected: Number of expected events (positive float).
        ci_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dictionary with value, ci_lower, ci_upper, interpretation,
        observed, and expected.

    Raises:
        ValueError: If observed < 0 or expected <= 0.
    """
    if observed < 0:
        raise ValueError("Observed count must be non-negative.")
    if expected <= 0:
        raise ValueError("Expected count must be positive.")
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be between 0 and 1 (exclusive).")

    smr = observed / expected
    alpha = 1 - ci_level

    # Exact Poisson CI via chi-squared relationship
    # Lower: chi2(alpha/2, 2*observed) / (2*expected)
    # Upper: chi2(1 - alpha/2, 2*(observed+1)) / (2*expected)
    if observed == 0:
        ci_lower = 0.0
    else:
        ci_lower = chi2.ppf(alpha / 2, 2 * observed) / (2 * expected)

    ci_upper = chi2.ppf(1 - alpha / 2, 2 * (observed + 1)) / (2 * expected)

    interpretation = interpret_smr(smr, ci_lower, ci_upper)

    return {
        "value": round(smr, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "interpretation": interpretation,
        "observed": observed,
        "expected": round(expected, 4),
    }


def calculate_expected_events(strata: list[dict]) -> dict:
    """Calculate total expected events from age/sex-stratified data.

    Args:
        strata: List of dicts, each with keys:
            - stratum_name (str): Label for the stratum (e.g. "40-49 males")
            - person_time (float): Person-time in the study population
            - reference_rate (float): Event rate in the reference population
            - observed (int, optional): Observed events in this stratum

    Returns:
        Dictionary with expected, strata_details, total_person_time,
        and total_observed.

    Raises:
        ValueError: If strata is empty or contains invalid values.
    """
    if not strata:
        raise ValueError("Strata list must not be empty.")

    total_expected = 0.0
    total_person_time = 0.0
    total_observed = 0
    details = []

    for s in strata:
        pt = s["person_time"]
        rate = s["reference_rate"]

        if pt < 0:
            raise ValueError(
                f"Person-time must be non-negative (stratum: {s.get('stratum_name', '?')})."
            )
        if rate < 0:
            raise ValueError(
                f"Reference rate must be non-negative (stratum: {s.get('stratum_name', '?')})."
            )

        stratum_expected = pt * rate
        total_expected += stratum_expected
        total_person_time += pt

        obs = s.get("observed", 0)
        if obs < 0:
            raise ValueError(
                f"Observed count must be non-negative (stratum: {s.get('stratum_name', '?')})."
            )
        total_observed += obs

        details.append({
            "stratum_name": s.get("stratum_name", ""),
            "person_time": pt,
            "reference_rate": rate,
            "expected": round(stratum_expected, 4),
            "observed": obs,
        })

    return {
        "expected": round(total_expected, 4),
        "strata_details": details,
        "total_person_time": round(total_person_time, 4),
        "total_observed": total_observed,
    }


def calculate_smr_stratified(
    strata: list[dict], ci_level: float = 0.95
) -> dict:
    """Calculate SMR/SIR from stratified person-time and reference rates.

    Pipeline: calculate_expected_events → sum observed → calculate_smr.

    Args:
        strata: List of stratum dicts (see calculate_expected_events).
            Each must include 'observed' key.
        ci_level: Confidence level (default 0.95).

    Returns:
        SMR result dict plus strata_details and total_person_time.

    Raises:
        ValueError: If strata is empty or expected events is zero.
    """
    exp_result = calculate_expected_events(strata)
    total_observed = exp_result["total_observed"]
    total_expected = exp_result["expected"]

    if total_expected <= 0:
        raise ValueError(
            "Total expected events must be positive. "
            "Check that person-time and reference rates are non-zero."
        )

    smr_result = calculate_smr(total_observed, total_expected, ci_level)
    smr_result["strata_details"] = exp_result["strata_details"]
    smr_result["total_person_time"] = exp_result["total_person_time"]

    return smr_result
