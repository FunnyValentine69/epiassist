"""Tests for the manuscript Methods section generator."""

import pandas as pd
import pytest

from utils.methods_generator import _english_list, generate_methods_section


# ---------------------------------------------------------------------------
# Fixtures — plain dicts mimicking session state (no Streamlit needed)
# ---------------------------------------------------------------------------

def _minimal_data_state() -> dict:
    """State with just a DataFrame and variable roles."""
    df = pd.DataFrame({
        "outcome": [0, 1, 1, 0, 1],
        "exposure": [0, 1, 1, 0, 0],
        "age": [25, 30, 45, 50, 35],
        "sex": [0, 1, 0, 1, 1],
    })
    return {
        "data_df": df,
        "data_source_name": "NHANES 2017-2018",
        "data_outcome_col": "outcome",
        "data_exposure_col": "exposure",
        "data_confounder_cols": ["age", "sex"],
    }


def _regression_state(model_type: str = "logistic") -> dict:
    """State with regression results."""
    state = _minimal_data_state()
    state["data_reg_result"] = {
        "model_type": model_type,
        "weighted": False,
        "n_observations": 5,
        "n_dropped": 0,
        "exposure_effect": {
            "variable": "exposure",
            "coef": 0.5,
            "effect": 1.65,
            "ci_lower": 1.10,
            "ci_upper": 2.47,
            "p_value": 0.015,
            "se": 0.20,
        },
        "coefficients": [],
        "model_fit": {"aic": 100.0, "bic": 105.0},
        "converged": True,
        "interpretation": "Test interpretation.",
    }
    return state


def _propensity_score_state() -> dict:
    """State with propensity score results."""
    state = _minimal_data_state()
    state["data_ps_result"] = {
        "ps_model": {},
        "common_support": {},
        "iptw": {},
        "balance": {"all_balanced": True},
        "treatment_effect": {"value": 1.5, "ci_lower": 1.1, "ci_upper": 2.0},
        "interpretation": "Test.",
    }
    return state


def _mediation_state() -> dict:
    """State with mediation results."""
    state = _minimal_data_state()
    state["data_mediator_cols"] = ["stress"]
    state["data_med_result"] = {
        "models": {},
        "effects": {"indirect": 0.3, "direct": 0.7, "total": 1.0},
        "ci": {},
        "n_observations": 5,
        "interpretation": "Test.",
    }
    return state


def _meta_analysis_state() -> dict:
    """State with meta-analysis results (both models)."""
    return {
        "meta_results": {
            "fixed": {"value": 1.5},
            "random": {"value": 1.4, "tau_squared": 0.02},
            "heterogeneity": {"i_squared": 45.0},
            "studies": [],
        },
        "meta_measure_type": "OR",
    }


def _sensitivity_state() -> dict:
    """State with E-value results."""
    return {
        "e_value_result": {
            "e_value": 2.5,
            "e_value_ci": 1.8,
            "interpretation": "Test.",
        },
    }


def _full_state() -> dict:
    """State with every analysis populated."""
    state = _minimal_data_state()
    state["data_weight_col"] = "survey_wt"
    state.update(_regression_state())
    state.update(_propensity_score_state())
    state.update(_mediation_state())
    state.update(_meta_analysis_state())
    state.update(_sensitivity_state())
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnglishList:
    """Tests for the _english_list helper."""

    def test_empty(self):
        assert _english_list([]) == ""

    def test_single(self):
        assert _english_list(["age"]) == "age"

    def test_two(self):
        assert _english_list(["age", "sex"]) == "age and sex"

    def test_three_oxford_comma(self):
        result = _english_list(["age", "sex", "education"])
        assert result == "age, sex, and education"

    def test_four(self):
        result = _english_list(["a", "b", "c", "d"])
        assert result == "a, b, c, and d"


class TestEmptyState:
    """Empty or minimal state handling."""

    def test_empty_state_returns_stub(self):
        text = generate_methods_section({})
        assert "## Methods" in text
        assert "EpiAssist" in text
        assert "statsmodels" in text

    def test_software_always_present(self):
        text = generate_methods_section({})
        assert "### Software" in text
        assert "EpiAssist" in text


class TestStudyDesignSection:

    def test_source_name_and_n(self):
        state = _minimal_data_state()
        text = generate_methods_section(state)
        assert "NHANES 2017-2018" in text
        assert "N = 5" in text

    def test_variable_names_appear(self):
        state = _minimal_data_state()
        text = generate_methods_section(state)
        assert "outcome" in text
        assert "exposure" in text

    def test_confounders_listed(self):
        state = _minimal_data_state()
        text = generate_methods_section(state)
        assert "age" in text
        assert "sex" in text

    def test_weighted_analysis_noted(self):
        state = _minimal_data_state()
        state["data_weight_col"] = "survey_wt"
        text = generate_methods_section(state)
        assert "survey_wt" in text
        assert "survey" in text.lower()


class TestRegressionSection:

    def test_logistic(self):
        state = _regression_state("logistic")
        text = generate_methods_section(state)
        assert "logistic regression" in text
        assert "odds ratio" in text.lower()

    def test_linear(self):
        state = _regression_state("linear")
        text = generate_methods_section(state)
        assert "linear regression" in text
        assert "beta" in text.lower()

    def test_poisson(self):
        state = _regression_state("poisson")
        text = generate_methods_section(state)
        assert "Poisson regression" in text
        assert "incidence rate ratio" in text.lower()

    def test_confounders_in_regression(self):
        state = _regression_state()
        text = generate_methods_section(state)
        assert "age" in text
        assert "sex" in text


class TestPropensityScoreSection:

    def test_iptw_and_citation(self):
        state = _propensity_score_state()
        text = generate_methods_section(state)
        assert "IPTW" in text
        assert "Austin" in text

    def test_balance_threshold(self):
        state = _propensity_score_state()
        text = generate_methods_section(state)
        assert "0.1" in text

    def test_no_confounders_no_extra_space(self):
        """PS section with no confounders should not have 'regression .' spacing."""
        state = _propensity_score_state()
        state["data_confounder_cols"] = []
        text = generate_methods_section(state)
        assert "regression ." not in text
        assert "logistic regression." in text


class TestMediationSection:

    def test_baron_kenny_citation(self):
        state = _mediation_state()
        text = generate_methods_section(state)
        assert "Baron and Kenny" in text
        assert "1986" in text

    def test_mediator_name(self):
        state = _mediation_state()
        text = generate_methods_section(state)
        assert "stress" in text


class TestMetaAnalysisSection:

    def test_dersimonian_laird_citation(self):
        state = _meta_analysis_state()
        text = generate_methods_section(state)
        assert "DerSimonian and Laird" in text
        assert "1986" in text

    def test_both_models_mentioned(self):
        state = _meta_analysis_state()
        text = generate_methods_section(state)
        assert "fixed-effect" in text
        assert "random-effects" in text

    def test_measure_type_appears(self):
        state = _meta_analysis_state()
        text = generate_methods_section(state)
        assert "OR" in text

    def test_fixed_only_no_dersimonian(self):
        """Fixed-only meta should NOT mention DerSimonian and Laird."""
        state = {
            "meta_results": {
                "fixed": {"value": 1.5},
                "random": None,
                "heterogeneity": {"i_squared": 0.0},
                "studies": [],
            },
            "meta_measure_type": "OR",
        }
        text = generate_methods_section(state)
        assert "fixed-effect" in text
        assert "DerSimonian" not in text

    def test_random_only_includes_dersimonian(self):
        """Random-only meta SHOULD mention DerSimonian and Laird."""
        state = {
            "meta_results": {
                "fixed": None,
                "random": {"value": 1.4, "tau_squared": 0.02},
                "heterogeneity": {"i_squared": 45.0},
                "studies": [],
            },
            "meta_measure_type": "RR",
        }
        text = generate_methods_section(state)
        assert "random-effects" in text
        assert "DerSimonian and Laird" in text


class TestSensitivitySection:

    def test_e_value_and_citation(self):
        state = _sensitivity_state()
        text = generate_methods_section(state)
        assert "E-value" in text
        assert "VanderWeele" in text
        assert "2017" in text


class TestPartialState:

    def test_partial_state_no_errors(self):
        """Subset of keys should not crash."""
        state = {"data_df": pd.DataFrame({"x": [1, 2, 3]})}
        text = generate_methods_section(state)
        assert "## Methods" in text

    def test_only_meta(self):
        """Only meta-analysis keys present."""
        state = _meta_analysis_state()
        text = generate_methods_section(state)
        assert "### Meta-Analysis" in text
        assert "### Study Design" not in text


class TestAllSectionsCombined:

    def test_all_sections_present(self):
        state = _full_state()
        text = generate_methods_section(state)

        assert "### Study Design" in text
        assert "### Descriptive Statistics" in text
        assert "### Cross-Tabulation" in text
        assert "### Regression Analysis" in text
        assert "### Propensity Score Analysis" in text
        assert "### Mediation Analysis" in text
        assert "### Meta-Analysis" in text
        assert "### Sensitivity Analysis" in text
        assert "### Software" in text

    def test_section_ordering(self):
        """Study Design comes before Regression, which comes before Software."""
        state = _full_state()
        text = generate_methods_section(state)

        idx_design = text.index("### Study Design")
        idx_regression = text.index("### Regression Analysis")
        idx_software = text.index("### Software")

        assert idx_design < idx_regression < idx_software

    def test_section_ordering_detailed(self):
        """All sections appear in the correct order."""
        state = _full_state()
        text = generate_methods_section(state)

        expected_order = [
            "### Study Design",
            "### Descriptive Statistics",
            "### Cross-Tabulation",
            "### Regression Analysis",
            "### Propensity Score Analysis",
            "### Mediation Analysis",
            "### Meta-Analysis",
            "### Sensitivity Analysis",
            "### Software",
        ]

        indices = [text.index(heading) for heading in expected_order]
        assert indices == sorted(indices)
