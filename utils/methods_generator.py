"""Manuscript Methods section generator.

Produces template-based, publication-ready Methods text from
the current session state.  Each analysis type has a private
section generator that returns ``str | None``; sections are
concatenated in standard epidemiological convention order.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _english_list(items: list[str]) -> str:
    """Format a list as an English phrase with Oxford comma.

    Examples:
        >>> _english_list(["age"])
        'age'
        >>> _english_list(["age", "sex"])
        'age and sex'
        >>> _english_list(["age", "sex", "education"])
        'age, sex, and education'
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# ---------------------------------------------------------------------------
# Private section generators – each returns str | None
# ---------------------------------------------------------------------------

def _section_study_design(state: dict) -> str | None:
    """Study design paragraph: data source, N, variable roles."""
    df = state.get("data_df")
    if df is None:
        return None

    source = state.get("data_source_name", "the uploaded dataset")
    n = len(df)
    outcome = state.get("data_outcome_col")
    exposure = state.get("data_exposure_col")
    confounders = state.get("data_confounder_cols", [])
    weight_col = state.get("data_weight_col")

    parts: list[str] = []
    parts.append(
        f"Data were obtained from {source} (N = {n:,})."
    )

    if outcome and exposure:
        parts.append(
            f"The outcome variable was {outcome} and the primary "
            f"exposure was {exposure}."
        )
    elif outcome:
        parts.append(f"The outcome variable was {outcome}.")
    elif exposure:
        parts.append(f"The primary exposure was {exposure}.")

    if confounders:
        parts.append(
            f"Potential confounders included {_english_list(confounders)}."
        )

    if weight_col:
        parts.append(
            f"Survey sampling weights ({weight_col}) were applied to "
            f"account for the complex survey design."
        )

    return "### Study Design\n\n" + " ".join(parts)


def _section_descriptive(state: dict) -> str | None:
    """Brief mention that descriptive statistics were computed."""
    if state.get("data_df") is None:
        return None
    return (
        "### Descriptive Statistics\n\n"
        "Descriptive statistics (means, standard deviations, and "
        "frequencies) were computed for all study variables to "
        "characterize the analytic sample."
    )


def _section_cross_tabulation(state: dict) -> str | None:
    """2x2 table methods with effect measures and chi-square."""
    outcome = state.get("data_outcome_col")
    exposure = state.get("data_exposure_col")
    if not outcome or not exposure:
        return None

    return (
        "### Cross-Tabulation\n\n"
        f"A 2\u00d72 contingency table was constructed for {exposure} "
        f"and {outcome}. The odds ratio (OR), risk ratio (RR), and "
        f"risk difference (RD) were calculated with 95% confidence "
        f"intervals. Association was assessed using the Pearson "
        f"chi-square test."
    )


def _section_regression(state: dict) -> str | None:
    """Regression model description with confounders."""
    reg = state.get("data_reg_result")
    if reg is None:
        return None

    model_type = reg.get("model_type", "logistic")
    n_obs = reg.get("n_observations", "")
    weighted = reg.get("weighted", False)
    outcome = state.get("data_outcome_col", "the outcome")
    exposure = state.get("data_exposure_col", "the exposure")
    confounders = state.get("data_confounder_cols", [])

    model_map = {
        "logistic": ("logistic regression", "adjusted odds ratios (aOR)"),
        "linear": ("linear regression", "adjusted beta coefficients"),
        "poisson": ("Poisson regression", "adjusted incidence rate ratios (aIRR)"),
    }
    model_name, effect_label = model_map.get(
        model_type, ("regression", "adjusted effect estimates")
    )

    parts: list[str] = []
    parts.append(
        f"Multivariable {model_name} was used to estimate "
        f"{effect_label} for the association between {exposure} "
        f"and {outcome}."
    )

    if confounders:
        parts.append(
            f"Models were adjusted for {_english_list(confounders)}."
        )

    if weighted:
        parts.append("Survey sampling weights were incorporated.")

    if n_obs:
        parts.append(f"The analytic sample included {n_obs:,} observations.")

    parts.append("All models were fit using generalized linear models (GLM).")

    return "### Regression Analysis\n\n" + " ".join(parts)


def _section_propensity_score(state: dict) -> str | None:
    """IPTW propensity score methods."""
    ps = state.get("data_ps_result")
    if ps is None:
        return None

    exposure = state.get("data_exposure_col", "treatment")
    confounders = state.get("data_confounder_cols", [])

    parts: list[str] = []
    if confounders:
        parts.append(
            f"Propensity scores for {exposure} were estimated using "
            f"logistic regression with {_english_list(confounders)} as predictors."
        )
    else:
        parts.append(
            f"Propensity scores for {exposure} were estimated using "
            f"logistic regression."
        )

    parts.append(
        "Inverse probability of treatment weights (IPTW) were "
        "computed to create a pseudo-population in which exposure "
        "was independent of measured confounders. "
        "Covariate balance was assessed using standardized mean "
        "differences (SMD), with a threshold of < 0.1 indicating "
        "adequate balance (Austin, 2011). "
        "Bootstrap resampling was used to construct 95% confidence "
        "intervals for the treatment effect estimate."
    )

    return "### Propensity Score Analysis\n\n" + " ".join(parts)


def _section_mediation(state: dict) -> str | None:
    """Baron-Kenny mediation analysis methods."""
    med = state.get("data_med_result")
    if med is None:
        return None

    mediators = state.get("data_mediator_cols", [])
    exposure = state.get("data_exposure_col", "the exposure")
    outcome = state.get("data_outcome_col", "the outcome")

    mediator_text = _english_list(mediators) if mediators else "the mediator"

    parts: list[str] = []
    parts.append(
        f"Mediation analysis was conducted to assess whether "
        f"{mediator_text} mediated the relationship between "
        f"{exposure} and {outcome}, following the Baron and Kenny "
        f"(1986) causal steps approach."
    )
    parts.append(
        "The indirect effect was estimated as the product of "
        "coefficients (a \u00d7 b) and verified using the difference "
        "method (c \u2212 c'). "
        "Statistical significance was evaluated using the Sobel test "
        "and nonparametric bootstrap confidence intervals."
    )

    return "### Mediation Analysis\n\n" + " ".join(parts)


def _section_meta_analysis(state: dict) -> str | None:
    """Meta-analysis methods."""
    meta = state.get("meta_results")
    if meta is None:
        return None

    measure = state.get("meta_measure_type", "the effect measure")

    parts: list[str] = []

    has_fixed = meta.get("fixed") is not None
    has_random = meta.get("random") is not None

    if has_fixed and has_random:
        parts.append(
            f"Both fixed-effect and random-effects meta-analyses "
            f"were performed to pool {measure} estimates across studies."
        )
    elif has_random:
        parts.append(
            f"A random-effects meta-analysis was performed to pool "
            f"{measure} estimates across studies."
        )
    else:
        parts.append(
            f"A fixed-effect meta-analysis was performed to pool "
            f"{measure} estimates across studies."
        )

    if has_random:
        parts.append(
            "Random-effects models used the DerSimonian and Laird (1986) "
            "method for estimating between-study variance."
        )

    parts.append(
        "Heterogeneity was assessed using Cochran's Q test and the "
        "I\u00b2 statistic. "
        "Results are presented with 95% confidence intervals."
    )

    return "### Meta-Analysis\n\n" + " ".join(parts)


def _section_sensitivity(state: dict) -> str | None:
    """E-value sensitivity analysis methods."""
    ev = state.get("e_value_result")
    if ev is None:
        return None

    return (
        "### Sensitivity Analysis\n\n"
        "The E-value was calculated to quantify the minimum strength "
        "of association that an unmeasured confounder would need to "
        "have with both the exposure and the outcome to fully explain "
        "away the observed association (VanderWeele & Ding, 2017)."
    )


def _section_software(state: dict) -> str | None:  # noqa: ARG001
    """Software and tools statement (always included)."""
    return (
        "### Software\n\n"
        "All analyses were conducted using EpiAssist, a web-based "
        "epidemiological analysis platform built with Python. "
        "Statistical computations used the statsmodels and SciPy "
        "libraries. Significance was assessed at the \u03b1 = 0.05 "
        "level unless otherwise noted."
    )


# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------

_SECTION_GENERATORS = [
    _section_study_design,
    _section_descriptive,
    _section_cross_tabulation,
    _section_regression,
    _section_propensity_score,
    _section_mediation,
    _section_meta_analysis,
    _section_sensitivity,
    _section_software,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_methods_section(state: dict) -> str:
    """Generate a publication-ready Methods section from analysis results.

    Iterates through all section generators in standard Methods
    convention order.  Only sections whose prerequisite session-state
    keys are present are included.  The Software section is always
    appended.

    Args:
        state: A dict snapshot of ``st.session_state`` (or a plain
               dict for testing).

    Returns:
        Markdown-formatted Methods section string.
    """
    sections: list[str] = []
    for gen in _SECTION_GENERATORS:
        text = gen(state)
        if text is not None:
            sections.append(text)

    if not sections:
        # Fallback: only software statement
        sw = _section_software(state)
        if sw is not None:
            sections.append(sw)

    return "## Methods\n\n" + "\n\n".join(sections) + "\n"
