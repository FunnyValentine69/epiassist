"""PDF report generator for EpiAssist analysis sessions.

Uses fpdf2 to produce a structured PDF with text and tables
(no figures in V1).  Each section renderer checks for its
session-state key and skips silently if absent.
"""

from __future__ import annotations

from datetime import date

from fpdf import FPDF


# ---------------------------------------------------------------------------
# Custom FPDF subclass
# ---------------------------------------------------------------------------

class EpiAssistReport(FPDF):
    """PDF document with branded header/footer."""

    def header(self) -> None:
        """Add header on every page except the first (title page)."""
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 10)
        self.cell(w=0, h=8, text="EpiAssist Analysis Report", align="L")
        self.cell(w=0, h=8, text=date.today().strftime("%Y-%m-%d"), align="R")
        self.ln(12)

    def footer(self) -> None:
        """Page N / Total in the footer."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(w=0, h=10, text=f"Page {self.page_no()}/{{nb}}", align="C")


# ---------------------------------------------------------------------------
# Table helper
# ---------------------------------------------------------------------------

def _render_table(
    pdf: FPDF,
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
) -> None:
    """Render a table using fpdf2's table() context manager.

    Args:
        pdf: The FPDF instance.
        headers: Column header strings.
        rows: List of row data (each row is a list of strings).
        col_widths: Optional explicit column widths.
    """
    if not headers:
        return

    widths = tuple(int(w) for w in col_widths) if col_widths else tuple(
        int(pdf.epw / len(headers)) for _ in headers
    )
    with pdf.table(col_widths=widths) as table:
        # Header row
        header_row = table.row()
        for h in headers:
            header_row.cell(h)

        # Data rows
        for row_data in rows:
            row = table.row()
            for cell_val in row_data:
                row.cell(str(cell_val) if cell_val is not None else "N/A")


def _safe(value: object, fmt: str = "") -> str:
    """Safely format a value for PDF display."""
    if value is None:
        return "N/A"
    if fmt and isinstance(value, (int, float)):
        return f"{value:{fmt}}"
    return str(value)


# ---------------------------------------------------------------------------
# Section renderers — each takes (pdf, session_state) -> None
# ---------------------------------------------------------------------------

def _add_title_page(pdf: FPDF, state: dict) -> None:
    """Title page with report name, date, data source, N."""
    pdf.add_page()

    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(w=0, h=15, text="EpiAssist Analysis Report", align="C")
    pdf.ln(20)

    pdf.set_font("Helvetica", size=14)
    pdf.cell(w=0, h=10, text=date.today().strftime("%B %d, %Y"), align="C")
    pdf.ln(20)

    pdf.set_font("Helvetica", size=12)
    source = state.get("data_source_name")
    if source:
        pdf.cell(w=0, h=10, text=f"Data Source: {source}", align="C")
        pdf.ln(8)

    df = state.get("data_df")
    if df is not None:
        pdf.cell(w=0, h=10, text=f"Sample Size: N = {len(df):,}", align="C")
        pdf.ln(8)


def _add_data_summary(pdf: FPDF, state: dict) -> None:
    """Data summary: source, N, column types, variable roles."""
    if "data_df" not in state:
        return

    df = state["data_df"]
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Data Summary")
    pdf.ln(12)

    pdf.set_font("Helvetica", size=11)
    source = state.get("data_source_name", "Unknown")
    pdf.cell(w=0, h=8, text=f"Source: {source}")
    pdf.ln(6)
    pdf.cell(w=0, h=8, text=f"Observations: {len(df):,}")
    pdf.ln(6)
    pdf.cell(w=0, h=8, text=f"Variables: {len(df.columns)}")
    pdf.ln(10)

    # Variable roles
    outcome = state.get("data_outcome_col")
    exposure = state.get("data_exposure_col")
    confounders = state.get("data_confounder_cols", [])
    weight_col = state.get("data_weight_col")

    if outcome or exposure:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Variable Roles")
        pdf.ln(8)
        roles = []
        if outcome:
            roles.append(["Outcome", outcome])
        if exposure:
            roles.append(["Exposure", exposure])
        if confounders:
            roles.append(["Confounders", ", ".join(confounders)])
        if weight_col:
            roles.append(["Weight Column", weight_col])
        _render_table(pdf, ["Role", "Variable(s)"], roles, [40, pdf.epw - 40])

    # Column summary table
    col_summary = state.get("data_col_summary")
    if col_summary:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Column Summary")
        pdf.ln(8)
        headers = ["Column", "Type", "Non-null", "Unique"]
        rows = []
        for col in col_summary:
            rows.append([
                _safe(col.get("name")),
                _safe(col.get("type")),
                _safe(col.get("non_null")),
                _safe(col.get("unique")),
            ])
        _render_table(pdf, headers, rows)


def _add_dag_summary(pdf: FPDF, state: dict) -> None:
    """DAG summary: nodes, edges, adjustment set."""
    engine = state.get("dag_engine")
    if engine is None:
        return

    nodes = engine.nodes
    edges = engine.edges
    if not nodes:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="DAG Summary")
    pdf.ln(12)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(w=0, h=8, text=f"Nodes: {len(nodes)}")
    pdf.ln(6)
    pdf.cell(w=0, h=8, text=f"Edges: {len(edges)}")
    pdf.ln(10)

    # Node table
    node_rows = []
    for name, data in nodes:
        node_rows.append([name, data.get("node_type", "unknown")])
    _render_table(pdf, ["Node", "Type"], node_rows)


def _add_effect_estimates(pdf: FPDF, state: dict) -> None:
    """2x2 table effect estimates: OR, RR, RD, Chi-square."""
    results = state.get("stats_results")
    if results is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Effect Estimates (2x2 Table)")
    pdf.ln(12)

    rows = []
    for key, label in [
        ("or", "Odds Ratio"),
        ("rr", "Risk Ratio"),
        ("rd", "Risk Difference"),
    ]:
        data = results.get(key)
        if data:
            rows.append([
                label,
                _safe(data.get("value"), ".4f"),
                f"({_safe(data.get('ci_lower'), '.4f')}, {_safe(data.get('ci_upper'), '.4f')})",
            ])

    chi = results.get("chi_square")
    if chi:
        rows.append([
            "Chi-square",
            _safe(chi.get("statistic"), ".4f"),
            f"p = {_safe(chi.get('p_value'), '.6f')}",
        ])

    if rows:
        _render_table(pdf, ["Measure", "Value", "95% CI / p-value"], rows)


def _add_regression(pdf: FPDF, state: dict) -> None:
    """Regression results: coefficient table, model fit."""
    reg = state.get("data_reg_result")
    if reg is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    model_type = reg.get("model_type", "Regression")
    pdf.cell(w=0, h=10, text=f"{model_type.capitalize()} Regression Results")
    pdf.ln(12)

    # Model info
    pdf.set_font("Helvetica", size=11)
    pdf.cell(w=0, h=8, text=f"Model type: {model_type}")
    pdf.ln(6)
    pdf.cell(w=0, h=8, text=f"Observations: {_safe(reg.get('n_observations'))}")
    pdf.ln(6)
    if reg.get("weighted"):
        pdf.cell(w=0, h=8, text="Survey weights: Yes")
        pdf.ln(6)
    pdf.ln(6)

    # Coefficient table
    coefficients = reg.get("coefficients", [])
    if coefficients:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Coefficients")
        pdf.ln(8)
        headers = ["Variable", "Coef", "Effect", "95% CI", "p-value"]
        rows = []
        for c in coefficients:
            rows.append([
                _safe(c.get("variable")),
                _safe(c.get("coef"), ".4f"),
                _safe(c.get("effect"), ".4f"),
                f"({_safe(c.get('ci_lower'), '.4f')}, {_safe(c.get('ci_upper'), '.4f')})",
                _safe(c.get("p_value"), ".6f"),
            ])
        _render_table(pdf, headers, rows)

    # Model fit
    model_fit = reg.get("model_fit", {})
    if model_fit:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Model Fit")
        pdf.ln(8)
        fit_rows = []
        for key, label in [
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("deviance", "Deviance"),
            ("r_squared", "R-squared"),
            ("pseudo_r_squared", "Pseudo R-squared"),
        ]:
            val = model_fit.get(key)
            if val is not None:
                fit_rows.append([label, _safe(val, ".4f")])
        if fit_rows:
            _render_table(pdf, ["Statistic", "Value"], fit_rows)

    # Interpretation
    interp = reg.get("interpretation")
    if interp:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Interpretation")
        pdf.ln(8)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(w=0, h=6, text=interp)


def _add_propensity_score(pdf: FPDF, state: dict) -> None:
    """Propensity score: treatment effect, balance, weight summary."""
    ps = state.get("data_ps_result")
    if ps is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Propensity Score Analysis")
    pdf.ln(12)

    # Treatment effect
    te = ps.get("treatment_effect", {})
    if te:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Treatment Effect")
        pdf.ln(8)
        _render_table(
            pdf,
            ["Estimand", "Effect", "95% CI"],
            [[
                _safe(te.get("estimand", "ATE")),
                _safe(te.get("value"), ".4f"),
                f"({_safe(te.get('ci_lower'), '.4f')}, {_safe(te.get('ci_upper'), '.4f')})",
            ]],
        )

    # Balance diagnostics
    balance = ps.get("balance", {})
    covariates = balance.get("covariates", [])
    if covariates:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Balance Diagnostics")
        pdf.ln(8)
        headers = ["Covariate", "SMD (raw)", "SMD (weighted)", "Balanced"]
        rows = []
        for cov in covariates:
            rows.append([
                _safe(cov.get("name")),
                _safe(cov.get("smd_raw"), ".4f"),
                _safe(cov.get("smd_weighted"), ".4f"),
                "Yes" if cov.get("balanced") else "No",
            ])
        _render_table(pdf, headers, rows)

    # Weight summary
    iptw = ps.get("iptw", {})
    ws = iptw.get("weight_summary", {})
    if ws:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Weight Summary")
        pdf.ln(8)
        ws_rows = []
        for key, label in [
            ("mean", "Mean"),
            ("median", "Median"),
            ("min", "Min"),
            ("max", "Max"),
            ("effective_n", "Effective N"),
        ]:
            val = ws.get(key)
            if val is not None:
                ws_rows.append([label, _safe(val, ".2f")])
        if ws_rows:
            _render_table(pdf, ["Statistic", "Value"], ws_rows)

    # Interpretation
    interp = ps.get("interpretation")
    if interp:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Interpretation")
        pdf.ln(8)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(w=0, h=6, text=interp)


def _add_mediation(pdf: FPDF, state: dict) -> None:
    """Mediation: effect decomposition, path coefficients."""
    med = state.get("data_med_result")
    if med is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Mediation Analysis")
    pdf.ln(12)

    # Effect decomposition
    effects = med.get("effects", {})
    if effects:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Effect Decomposition")
        pdf.ln(8)
        ci = med.get("ci", {})
        rows = []
        for key, label in [
            ("indirect", "Indirect Effect"),
            ("direct", "Direct Effect"),
            ("total", "Total Effect"),
        ]:
            val = effects.get(key)
            ci_key = f"{key}_ci"
            ci_bounds = ci.get(ci_key)
            ci_str = (
                f"({ci_bounds[0]:.4f}, {ci_bounds[1]:.4f})"
                if ci_bounds and len(ci_bounds) == 2
                else "N/A"
            )
            rows.append([label, _safe(val, ".4f"), ci_str])
        _render_table(pdf, ["Effect", "Estimate", "95% CI"], rows)

    # Proportion mediated
    prop_med = effects.get("proportion_mediated")
    if prop_med is not None:
        pdf.ln(6)
        pdf.set_font("Helvetica", size=11)
        pdf.cell(w=0, h=8, text=f"Proportion mediated: {prop_med:.1%}")
        pdf.ln(6)

    # Path coefficients
    models = med.get("models", {})
    if models:
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Path Coefficients")
        pdf.ln(8)
        path_rows = []
        a_path = models.get("a_path", {})
        if "coef_a" in a_path:
            path_rows.append(["a (X -> M)", _safe(a_path["coef_a"], ".4f")])
        direct = models.get("direct", {})
        if "coef_b" in direct:
            path_rows.append(["b (M -> Y)", _safe(direct["coef_b"], ".4f")])
        if "coef_c_prime" in direct:
            path_rows.append(["c' (X -> Y, direct)", _safe(direct["coef_c_prime"], ".4f")])
        total = models.get("total", {})
        if "coef_c" in total:
            path_rows.append(["c (X -> Y, total)", _safe(total["coef_c"], ".4f")])
        if path_rows:
            _render_table(pdf, ["Path", "Coefficient"], path_rows)

    # Interpretation
    interp = med.get("interpretation")
    if interp:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Interpretation")
        pdf.ln(8)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(w=0, h=6, text=interp)


def _add_meta_analysis(pdf: FPDF, state: dict) -> None:
    """Meta-analysis: pooled estimates, heterogeneity, study weights."""
    meta = state.get("meta_results")
    if meta is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Meta-Analysis Results")
    pdf.ln(12)

    measure = state.get("meta_measure_type", "Effect")

    # Pooled estimates
    for model_key, label in [("fixed", "Fixed-Effect"), ("random", "Random-Effects")]:
        result = meta.get(model_key)
        if result is None:
            continue
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text=f"{label} Model")
        pdf.ln(8)
        _render_table(
            pdf,
            [measure, "95% CI", "p-value"],
            [[
                _safe(result.get("value"), ".4f"),
                f"({_safe(result.get('ci_lower'), '.4f')}, {_safe(result.get('ci_upper'), '.4f')})",
                _safe(result.get("p_value"), ".6f"),
            ]],
        )
        pdf.ln(8)

    # Heterogeneity
    het = meta.get("heterogeneity", {})
    if het:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Heterogeneity")
        pdf.ln(8)
        het_rows = []
        for key, label in [
            ("q_value", "Cochran's Q"),
            ("q_p_value", "Q p-value"),
            ("i_squared", "I-squared (%)"),
            ("tau_squared", "Tau-squared"),
        ]:
            val = het.get(key)
            if val is not None:
                het_rows.append([label, _safe(val, ".4f")])
        if het_rows:
            _render_table(pdf, ["Statistic", "Value"], het_rows)

    # Study weights table
    studies = meta.get("studies", [])
    if studies:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Individual Studies")
        pdf.ln(8)
        headers = ["Study", measure, "95% CI", "Weight (%)"]
        rows = []
        for s in studies:
            weight = s.get("weight")
            rows.append([
                _safe(s.get("name")),
                _safe(s.get("effect"), ".4f"),
                f"({_safe(s.get('ci_lower'), '.4f')}, {_safe(s.get('ci_upper'), '.4f')})",
                _safe(weight, ".1f") if weight is not None else "N/A",
            ])
        _render_table(pdf, headers, rows)


def _add_paper_summary(pdf: FPDF, state: dict) -> None:
    """Paper Analyzer: counts by category, top effect measures."""
    results = state.get("paper_results")
    if results is None:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Paper Analysis Summary")
    pdf.ln(12)

    # Counts
    pdf.set_font("Helvetica", size=11)
    filename = state.get("paper_filename", "Unknown")
    pdf.cell(w=0, h=8, text=f"Paper: {filename}")
    pdf.ln(8)

    for category, label in [
        ("effect_measures", "Effect Measures"),
        ("confidence_intervals", "Confidence Intervals"),
        ("p_values", "P-values"),
        ("sample_sizes", "Sample Sizes"),
    ]:
        items = results.get(category, [])
        pdf.cell(w=0, h=6, text=f"  {label}: {len(items)}")
        pdf.ln(5)

    # Top effect measures table
    ems = results.get("effect_measures", [])
    if ems:
        pdf.ln(8)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(w=0, h=8, text="Extracted Effect Measures")
        pdf.ln(8)
        headers = ["Type", "Value", "95% CI", "Adjusted"]
        rows = []
        for em in ems[:20]:  # Limit to first 20
            rows.append([
                _safe(em.get("type")),
                _safe(em.get("value"), ".4f"),
                f"({_safe(em.get('ci_lower'), '.4f')}, {_safe(em.get('ci_upper'), '.4f')})",
                "Yes" if em.get("adjusted") else "No",
            ])
        _render_table(pdf, headers, rows)


def _add_methods_section(pdf: FPDF, state: dict) -> None:
    """Include the generated manuscript Methods text."""
    methods = state.get("export_methods_text")
    if not methods:
        return

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(w=0, h=10, text="Generated Methods Section")
    pdf.ln(12)

    pdf.set_font("Helvetica", size=11)
    # Strip markdown headers for plain-text rendering
    clean_text = methods.replace("## ", "").replace("### ", "")
    pdf.multi_cell(w=0, h=6, text=clean_text)


# ---------------------------------------------------------------------------
# Section ordering
# ---------------------------------------------------------------------------

_SECTION_RENDERERS = [
    _add_title_page,
    _add_data_summary,
    _add_dag_summary,
    _add_effect_estimates,
    _add_regression,
    _add_propensity_score,
    _add_mediation,
    _add_meta_analysis,
    _add_paper_summary,
    _add_methods_section,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(session_state: dict) -> bytes:
    """Generate a PDF report from the current session state.

    Iterates through all section renderers.  Each renderer checks
    for its prerequisite keys and skips if absent.  The title page
    is always included.

    Args:
        session_state: A dict snapshot of ``st.session_state``
                       (or a plain dict for testing).

    Returns:
        PDF file contents as bytes.
    """
    pdf = EpiAssistReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    for renderer in _SECTION_RENDERERS:
        renderer(pdf, session_state)

    return bytes(pdf.output())
