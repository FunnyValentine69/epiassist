"""Statistics Calculator page for epidemiological measures."""

import streamlit as st
import plotly.graph_objects as go

from core.stats_calculator import (
    calculate_odds_ratio,
    calculate_risk_ratio,
    calculate_risk_difference,
    calculate_chi_square,
)
from utils.constants import DEMO_2X2_TABLE

st.set_page_config(page_title="Statistics Calculator - EpiAssist", layout="wide")

st.title("Statistics Calculator")
st.markdown("""
Calculate Odds Ratios, Risk Ratios, Risk Differences, and Chi-square tests
from 2x2 contingency tables with 95% confidence intervals.
""")

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 2x2 Contingency Table")

    st.markdown("""
    Enter counts from your study:

    |  | Outcome (+) | Outcome (-) |
    |---|---|---|
    | **Exposed** | a | b |
    | **Unexposed** | c | d |
    """)

    # Load demo data button
    if st.button("Load Demo Data"):
        st.session_state.stats_a = DEMO_2X2_TABLE["a"]
        st.session_state.stats_b = DEMO_2X2_TABLE["b"]
        st.session_state.stats_c = DEMO_2X2_TABLE["c"]
        st.session_state.stats_d = DEMO_2X2_TABLE["d"]
        st.rerun()

    # Initialize session state
    if "stats_a" not in st.session_state:
        st.session_state.stats_a = DEMO_2X2_TABLE["a"]
    if "stats_b" not in st.session_state:
        st.session_state.stats_b = DEMO_2X2_TABLE["b"]
    if "stats_c" not in st.session_state:
        st.session_state.stats_c = DEMO_2X2_TABLE["c"]
    if "stats_d" not in st.session_state:
        st.session_state.stats_d = DEMO_2X2_TABLE["d"]

    subcol1, subcol2 = st.columns(2)
    with subcol1:
        a = st.number_input(
            "a (Exposed, Outcome+)",
            min_value=0,
            value=st.session_state.stats_a,
            key="input_a",
        )
        c = st.number_input(
            "c (Unexposed, Outcome+)",
            min_value=0,
            value=st.session_state.stats_c,
            key="input_c",
        )
    with subcol2:
        b = st.number_input(
            "b (Exposed, Outcome-)",
            min_value=0,
            value=st.session_state.stats_b,
            key="input_b",
        )
        d = st.number_input(
            "d (Unexposed, Outcome-)",
            min_value=0,
            value=st.session_state.stats_d,
            key="input_d",
        )

    # Update session state
    st.session_state.stats_a = a
    st.session_state.stats_b = b
    st.session_state.stats_c = c
    st.session_state.stats_d = d

    # Display table summary
    n1 = a + b
    n0 = c + d
    total = n1 + n0

    st.markdown(f"""
    **Summary:**
    - Exposed: {n1} ({a} with outcome, {b} without)
    - Unexposed: {n0} ({c} with outcome, {d} without)
    - Total: {total}
    """)

    calculate_btn = st.button("Calculate Statistics", type="primary")

with col2:
    st.markdown("### Results")

    if calculate_btn or "stats_results" in st.session_state:
        # Calculate all statistics
        or_result = calculate_odds_ratio(a, b, c, d)
        rr_result = calculate_risk_ratio(a, b, c, d)
        rd_result = calculate_risk_difference(a, b, c, d)
        chi_result = calculate_chi_square(a, b, c, d)

        # Store results
        st.session_state.stats_results = {
            "or": or_result,
            "rr": rr_result,
            "rd": rd_result,
            "chi": chi_result,
        }

        # Display results in metrics
        col_or, col_rr = st.columns(2)

        with col_or:
            st.metric(
                "Odds Ratio (OR)",
                f"{or_result['value']:.2f}" if or_result["value"] else "N/A",
                f"95% CI: {or_result['ci_lower']:.2f}-{or_result['ci_upper']:.2f}"
                if or_result["ci_lower"]
                else "",
            )

        with col_rr:
            st.metric(
                "Risk Ratio (RR)",
                f"{rr_result['value']:.2f}" if rr_result["value"] else "N/A",
                f"95% CI: {rr_result['ci_lower']:.2f}-{rr_result['ci_upper']:.2f}"
                if rr_result["ci_lower"]
                else "",
            )

        col_rd, col_chi = st.columns(2)

        with col_rd:
            st.metric(
                "Risk Difference",
                f"{rd_result['value']:.1%}" if rd_result["value"] else "N/A",
                f"95% CI: {rd_result['ci_lower']:.1%}-{rd_result['ci_upper']:.1%}"
                if rd_result["ci_lower"]
                else "",
            )

        with col_chi:
            p_val = chi_result["p_value"]
            p_display = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            st.metric(
                "Chi-square",
                f"{chi_result['value']:.2f}",
                f"p-value: {p_display}",
            )

        # Forest plot-style visualization
        st.markdown("### Effect Estimate Visualization")

        fig = go.Figure()

        # Add OR point and CI
        fig.add_trace(
            go.Scatter(
                x=[or_result["value"]],
                y=[2],
                mode="markers",
                marker=dict(size=12, color="#FF6B6B"),
                name="OR",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[or_result["ci_upper"] - or_result["value"]],
                    arrayminus=[or_result["value"] - or_result["ci_lower"]],
                ),
            )
        )

        # Add RR point and CI
        if rr_result["value"]:
            fig.add_trace(
                go.Scatter(
                    x=[rr_result["value"]],
                    y=[1],
                    mode="markers",
                    marker=dict(size=12, color="#4ECDC4"),
                    name="RR",
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[rr_result["ci_upper"] - rr_result["value"]],
                        arrayminus=[rr_result["value"] - rr_result["ci_lower"]],
                    ),
                )
            )

        # Add reference line at 1
        fig.add_vline(x=1, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Effect Estimates with 95% CI",
            xaxis_title="Effect Estimate",
            yaxis=dict(
                ticktext=["RR", "OR"],
                tickvals=[1, 2],
                range=[0, 3],
            ),
            showlegend=False,
            height=250,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretations
        st.markdown("### Interpretations")

        with st.expander("Odds Ratio Interpretation", expanded=True):
            st.markdown(or_result["interpretation"])

        with st.expander("Risk Ratio Interpretation"):
            st.markdown(rr_result["interpretation"])

        with st.expander("Risk Difference Interpretation"):
            st.markdown(rd_result["interpretation"])

        with st.expander("Chi-square Test Interpretation"):
            st.markdown(chi_result["interpretation"])

    else:
        st.info("Enter your data and click Calculate to see results.")

st.divider()

st.markdown("### Demo Data Reference")
st.markdown("""
The default values show data from the **Hearing Loss and Unemployment** study:

| | Unemployed (+) | Employed (-) |
|---|---|---|
| **Hearing Loss** | 80 | 150 |
| **No Hearing Loss** | 70 | 400 |

- **OR = 3.05**: Individuals with hearing loss have 3x higher odds of unemployment
- **95% CI**: (2.10, 4.42) - statistically significant
- **p < 0.001**: Strong evidence of association
""")
