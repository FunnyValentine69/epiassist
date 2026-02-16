"""Export & Report — manuscript methods generator and PDF report export."""

from datetime import date

import streamlit as st

from core.report_generator import generate_report
from utils.methods_generator import generate_methods_section

st.set_page_config(page_title="Export & Report - EpiAssist", layout="wide")

st.title("Export & Report")
st.markdown("""
Generate publication-ready text and downloadable reports from your
analysis session.  Sections are included automatically based on which
analyses you have completed across all pages.
""")

st.divider()

# -----------------------------------------------------------------------
# Section 1 — Manuscript Methods Generator
# -----------------------------------------------------------------------

st.header("Manuscript Methods Section")
st.caption(
    "Template-based generation — produces standard epidemiological "
    "Methods language from your session.  Review and adapt before use."
)

if st.button("Generate Methods Section", type="primary"):
    methods_text = generate_methods_section(dict(st.session_state))
    st.session_state.export_methods_text = methods_text

if "export_methods_text" in st.session_state:
    text = st.session_state.export_methods_text

    st.markdown(text)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download as Markdown",
            data=text,
            file_name=f"methods_section_{date.today().isoformat()}.md",
            mime="text/markdown",
        )
    with col2:
        st.text_area(
            "Copy to clipboard",
            value=text,
            height=200,
            help="Select all text and copy.",
        )

st.divider()

# -----------------------------------------------------------------------
# Section 2 — PDF Report Export
# -----------------------------------------------------------------------

st.header("PDF Report Export")

# Check if at least one analysis result exists
_result_keys = [
    "data_df", "stats_results", "data_reg_result",
    "data_ps_result", "data_med_result", "meta_results",
    "paper_results", "e_value_result",
]
_has_results = any(k in st.session_state for k in _result_keys)

if _has_results:
    st.download_button(
        label="Download PDF Report",
        data=generate_report(dict(st.session_state)),
        file_name=f"epiassist_report_{date.today().isoformat()}.pdf",
        mime="application/pdf",
        type="primary",
    )
else:
    st.info("Run at least one analysis to enable PDF export.")
