"""Direct standardization for age-adjusted rates.

Computes adjusted rates by applying study-specific stratum rates to a
standard population. Confidence intervals use the Fay-Feuer method
(gamma distribution), the standard used by SEER/NCI.
"""

from scipy.stats import gamma

from utils.interpretations import interpret_direct_standardized_rate


def calculate_stratum_rates(strata: list[dict]) -> list[dict]:
    """Calculate rate and weighted events for each stratum.

    Args:
        strata: List of dicts, each with keys:
            - stratum_name (str): Label for the stratum (e.g. "45-54")
            - events (int): Number of events in this stratum
            - population (int): Population count in this stratum
            - standard_weight (int): Weight from the standard population

    Returns:
        Enriched list of stratum dicts with added 'rate' and 'weighted_events'.

    Raises:
        ValueError: If strata is empty or contains invalid values.
    """
    if not strata:
        raise ValueError("Strata list must not be empty.")

    results = []
    for s in strata:
        events = s["events"]
        population = s["population"]
        weight = s["standard_weight"]

        if events < 0:
            raise ValueError(
                f"Events must be non-negative (stratum: {s.get('stratum_name', '?')})."
            )
        if population <= 0:
            raise ValueError(
                f"Population must be positive (stratum: {s.get('stratum_name', '?')})."
            )
        if weight < 0:
            raise ValueError(
                f"Standard weight must be non-negative (stratum: {s.get('stratum_name', '?')})."
            )

        rate = events / population
        weighted_events = rate * weight

        results.append({
            "stratum_name": s.get("stratum_name", ""),
            "events": events,
            "population": population,
            "standard_weight": weight,
            "rate": rate,
            "weighted_events": weighted_events,
        })

    return results


def calculate_direct_standardized_rate(
    strata: list[dict],
    multiplier: int = 100_000,
    ci_level: float = 0.95,
) -> dict:
    """Calculate a directly standardized rate with Fay-Feuer CI.

    Pipeline: calculate_stratum_rates -> sum weighted events ->
    divide by total standard weight -> multiply by rate multiplier.

    Args:
        strata: List of stratum dicts (see calculate_stratum_rates).
        multiplier: Rate multiplier (e.g. 100000 for "per 100,000").
        ci_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dictionary with value, ci_lower, ci_upper, interpretation,
        strata_details, total_standard_pop, total_events,
        total_population, crude_rate, and multiplier.

    Raises:
        ValueError: If strata is empty, ci_level invalid, or
            total standard weight is zero.
    """
    if multiplier <= 0:
        raise ValueError("Rate multiplier must be a positive integer.")
    if not (0 < ci_level < 1):
        raise ValueError("ci_level must be between 0 and 1 (exclusive).")

    enriched = calculate_stratum_rates(strata)

    total_weight = sum(s["standard_weight"] for s in enriched)
    if total_weight == 0:
        raise ValueError("Total standard population weight must be positive.")

    total_weighted_events = sum(s["weighted_events"] for s in enriched)
    total_events = sum(s["events"] for s in enriched)
    total_population = sum(s["population"] for s in enriched)

    # Adjusted rate
    adjusted_rate = (total_weighted_events / total_weight) * multiplier

    # Crude rate
    crude_rate = (total_events / total_population) * multiplier if total_population > 0 else 0.0

    # Handle zero events edge case
    if total_events == 0:
        interpretation = interpret_direct_standardized_rate(
            0.0, 0.0, 0.0, crude_rate, multiplier
        )
        return {
            "value": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "interpretation": interpretation,
            "strata_details": enriched,
            "total_standard_pop": total_weight,
            "total_events": total_events,
            "total_population": total_population,
            "crude_rate": round(crude_rate, 4),
            "multiplier": multiplier,
        }

    # Fay-Feuer CI (gamma distribution)
    alpha = 1 - ci_level

    # Variance: sum((w_i / W)^2 * d_i / n_i^2)
    variance = sum(
        (s["standard_weight"] / total_weight) ** 2
        * s["events"]
        / s["population"] ** 2
        for s in enriched
    )

    # Gamma parameters
    weighted_rate = total_weighted_events / total_weight  # unscaled adjusted rate

    if weighted_rate <= 0 or variance <= 0:
        # Degenerate case: events exist but only in zero-weight strata,
        # or variance is zero. CI cannot be computed meaningfully.
        ci_lower = adjusted_rate
        ci_upper = adjusted_rate
    else:
        theta = variance / weighted_rate
        kappa = weighted_rate / theta

        ci_lower = gamma.ppf(alpha / 2, kappa, scale=theta) * multiplier
        ci_upper = gamma.ppf(1 - alpha / 2, kappa + 1, scale=theta) * multiplier

    interpretation = interpret_direct_standardized_rate(
        adjusted_rate, ci_lower, ci_upper, crude_rate, multiplier
    )

    return {
        "value": round(adjusted_rate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "interpretation": interpretation,
        "strata_details": enriched,
        "total_standard_pop": total_weight,
        "total_events": total_events,
        "total_population": total_population,
        "crude_rate": round(crude_rate, 4),
        "multiplier": multiplier,
    }
