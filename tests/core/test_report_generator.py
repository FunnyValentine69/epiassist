"""Tests for the PDF report generator."""

import fitz  # PyMuPDF
import pandas as pd
import pytest

from core.report_generator import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf_text(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF byte string using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def _pdf_page_count(pdf_bytes: bytes) -> int:
    """Count pages in a PDF byte string."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    count = len(doc)
    doc.close()
    return count


# ---------------------------------------------------------------------------
# Fixtures — plain dicts mimicking session state
# ---------------------------------------------------------------------------

def _data_state() -> dict:
    """State with a DataFrame and variable roles."""
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
        "data_col_summary": [
            {"name": "outcome", "type": "int64", "non_null": 5, "unique": 2},
            {"name": "exposure", "type": "int64", "non_null": 5, "unique": 2},
            {"name": "age", "type": "int64", "non_null": 5, "unique": 5},
            {"name": "sex", "type": "int64", "non_null": 5, "unique": 2},
        ],
    }


def _regression_state() -> dict:
    """State with regression results and many coefficients."""
    state = _data_state()
    coefficients = [
        {
            "variable": f"var_{i}",
            "coef": 0.1 * i,
            "effect": 1.0 + 0.1 * i,
            "ci_lower": 0.9 + 0.1 * i,
            "ci_upper": 1.1 + 0.1 * i,
            "p_value": 0.05 / (i + 1),
            "se": 0.05,
        }
        for i in range(5)
    ]
    state["data_reg_result"] = {
        "model_type": "logistic",
        "weighted": False,
        "n_observations": 5,
        "n_dropped": 0,
        "exposure_effect": coefficients[0],
        "coefficients": coefficients,
        "model_fit": {"aic": 100.0, "bic": 105.0, "deviance": 80.0, "pseudo_r_squared": 0.15},
        "converged": True,
        "interpretation": "The exposure was significantly associated with the outcome.",
    }
    return state


def _propensity_score_state() -> dict:
    """State with propensity score results."""
    state = _data_state()
    state["data_ps_result"] = {
        "ps_model": {},
        "common_support": {},
        "iptw": {
            "weight_summary": {
                "mean": 1.2, "median": 1.0, "min": 0.5,
                "max": 3.0, "effective_n": 4.5,
            },
        },
        "balance": {
            "all_balanced": True,
            "covariates": [
                {"name": "age", "smd_raw": 0.15, "smd_weighted": 0.03, "balanced": True},
                {"name": "sex", "smd_raw": 0.20, "smd_weighted": 0.05, "balanced": True},
            ],
            "summary": {"n_balanced_after": 2, "n_total": 2},
        },
        "treatment_effect": {
            "value": 1.5, "ci_lower": 1.1, "ci_upper": 2.0,
            "estimand": "ATE",
        },
        "interpretation": "The treatment effect was significant.",
    }
    return state


def _mediation_state() -> dict:
    """State with mediation results."""
    state = _data_state()
    state["data_mediator_cols"] = ["stress"]
    state["data_med_result"] = {
        "models": {
            "a_path": {"coef_a": 0.5, "se_a": 0.1},
            "direct": {"coef_b": 0.3, "se_b": 0.08, "coef_c_prime": 0.7},
            "total": {"coef_c": 1.0},
        },
        "effects": {
            "indirect": 0.15,
            "direct": 0.70,
            "total": 0.85,
            "proportion_mediated": 0.176,
            "sobel_p": 0.02,
            "method": "product",
        },
        "ci": {
            "indirect_ci": (0.05, 0.25),
            "direct_ci": (0.40, 1.00),
            "total_ci": (0.50, 1.20),
        },
        "n_observations": 5,
        "n_dropped": 0,
        "n_boot": 200,
        "interpretation": "Mediation was partially significant.",
    }
    return state


def _meta_analysis_state() -> dict:
    """State with meta-analysis results."""
    return {
        "meta_results": {
            "fixed": {"value": 1.50, "ci_lower": 1.20, "ci_upper": 1.88, "p_value": 0.001},
            "random": {"value": 1.45, "ci_lower": 1.10, "ci_upper": 1.91, "p_value": 0.008,
                       "tau_squared": 0.02},
            "heterogeneity": {
                "q_value": 8.5, "q_p_value": 0.13,
                "i_squared": 45.0, "tau_squared": 0.02,
            },
            "studies": [
                {"name": "Smith 2020", "effect": 1.3, "ci_lower": 1.0, "ci_upper": 1.7, "weight": 30.0},
                {"name": "Jones 2021", "effect": 1.8, "ci_lower": 1.2, "ci_upper": 2.7, "weight": 25.0},
                {"name": "Lee 2022", "effect": 1.5, "ci_lower": 1.1, "ci_upper": 2.0, "weight": 45.0},
            ],
        },
        "meta_measure_type": "OR",
    }


def _full_state() -> dict:
    """State with every analysis populated."""
    state = _data_state()
    state.update(_regression_state())
    state.update(_propensity_score_state())
    state.update(_mediation_state())
    state.update(_meta_analysis_state())
    state["export_methods_text"] = "## Methods\n\nAll analyses used EpiAssist.\n"
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicGeneration:

    def test_empty_session_valid_pdf(self):
        """Empty state produces valid PDF bytes."""
        pdf_bytes = generate_report({})
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 100
        assert pdf_bytes[:5] == b"%PDF-"

    def test_title_page_content(self):
        """Report title and date are present."""
        pdf_bytes = generate_report({})
        text = _pdf_text(pdf_bytes)
        assert "EpiAssist Analysis Report" in text


class TestDataSummary:

    def test_data_summary_section(self):
        """Source name and N appear in PDF."""
        state = _data_state()
        text = _pdf_text(generate_report(state))
        assert "NHANES 2017-2018" in text
        assert "5" in text  # N=5

    def test_variable_roles_in_summary(self):
        """Variable roles appear in data summary."""
        state = _data_state()
        text = _pdf_text(generate_report(state))
        assert "Outcome" in text
        assert "Exposure" in text


class TestRegressionSection:

    def test_regression_section(self):
        """Model type and coefficient table appear."""
        state = _regression_state()
        text = _pdf_text(generate_report(state))
        assert "Logistic" in text or "logistic" in text
        assert "Coefficients" in text

    def test_model_fit_in_regression(self):
        """Model fit statistics appear."""
        state = _regression_state()
        text = _pdf_text(generate_report(state))
        assert "AIC" in text


class TestPropensityScoreSection:

    def test_propensity_score_section(self):
        """Treatment effect and balance table appear."""
        state = _propensity_score_state()
        text = _pdf_text(generate_report(state))
        assert "Propensity Score" in text
        assert "Treatment Effect" in text

    def test_balance_diagnostics(self):
        """Balance diagnostics table is present."""
        state = _propensity_score_state()
        text = _pdf_text(generate_report(state))
        assert "Balance" in text


class TestMediationSection:

    def test_mediation_section(self):
        """Effect decomposition and path coefficients appear."""
        state = _mediation_state()
        text = _pdf_text(generate_report(state))
        assert "Mediation" in text
        assert "Indirect" in text or "indirect" in text.lower()

    def test_path_coefficients(self):
        """Path coefficient values are present."""
        state = _mediation_state()
        text = _pdf_text(generate_report(state))
        assert "Path" in text


class TestMetaAnalysisSection:

    def test_meta_analysis_section(self):
        """Pooled estimates and study names appear."""
        state = _meta_analysis_state()
        text = _pdf_text(generate_report(state))
        assert "Meta-Analysis" in text
        assert "Smith 2020" in text

    def test_heterogeneity_stats(self):
        """Heterogeneity statistics are present."""
        state = _meta_analysis_state()
        text = _pdf_text(generate_report(state))
        assert "Heterogeneity" in text


class TestPartialAndCombined:

    def test_partial_session(self):
        """Only some keys present -> only those sections, no errors."""
        state = _meta_analysis_state()
        pdf_bytes = generate_report(state)
        assert pdf_bytes[:5] == b"%PDF-"
        text = _pdf_text(pdf_bytes)
        assert "Meta-Analysis" in text
        assert "Regression" not in text

    def test_all_sections_combined(self):
        """Full state -> multi-page PDF with all sections."""
        state = _full_state()
        pdf_bytes = generate_report(state)
        text = _pdf_text(pdf_bytes)
        assert "Data Summary" in text
        assert "Regression" in text
        assert "Propensity Score" in text
        assert "Mediation" in text
        assert "Meta-Analysis" in text
        assert "Methods" in text

    def test_none_values_handled(self):
        """None in result dicts -> 'N/A' not crash."""
        state = _data_state()
        state["data_reg_result"] = {
            "model_type": "logistic",
            "weighted": False,
            "n_observations": None,
            "n_dropped": 0,
            "exposure_effect": None,
            "coefficients": [
                {"variable": "x", "coef": None, "effect": None,
                 "ci_lower": None, "ci_upper": None, "p_value": None, "se": None},
            ],
            "model_fit": {"aic": None},
            "converged": True,
            "interpretation": None,
        }
        pdf_bytes = generate_report(state)
        text = _pdf_text(pdf_bytes)
        assert "N/A" in text


class TestLongTable:

    def test_long_coefficient_table(self):
        """20+ coefficients -> table spans pages cleanly."""
        state = _data_state()
        coefficients = [
            {
                "variable": f"covariate_{i:02d}",
                "coef": 0.1 * i,
                "effect": 1.0 + 0.05 * i,
                "ci_lower": 0.9 + 0.05 * i,
                "ci_upper": 1.1 + 0.05 * i,
                "p_value": 0.05 / (i + 1),
                "se": 0.05,
            }
            for i in range(25)
        ]
        state["data_reg_result"] = {
            "model_type": "logistic",
            "weighted": False,
            "n_observations": 100,
            "n_dropped": 0,
            "exposure_effect": coefficients[0],
            "coefficients": coefficients,
            "model_fit": {"aic": 200.0},
            "converged": True,
            "interpretation": "Many covariates tested.",
        }
        pdf_bytes = generate_report(state)
        assert pdf_bytes[:5] == b"%PDF-"
        text = _pdf_text(pdf_bytes)
        assert "covariate_24" in text


class TestMethodsSection:

    def test_methods_section_included(self):
        """export_methods_text -> appears in PDF."""
        state = {"export_methods_text": "All analyses used EpiAssist software."}
        text = _pdf_text(generate_report(state))
        assert "EpiAssist" in text


class TestPageNumbers:

    def test_pdf_page_numbers(self):
        """Footer has page numbers on multi-page report."""
        state = _full_state()
        pdf_bytes = generate_report(state)
        page_count = _pdf_page_count(pdf_bytes)
        assert page_count >= 3  # Title + several sections
        text = _pdf_text(pdf_bytes)
        assert "Page" in text
