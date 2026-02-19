"""Statistics Calculator page for epidemiological measures."""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from core.stats_calculator import (
    calculate_odds_ratio,
    calculate_risk_ratio,
    calculate_risk_difference,
    calculate_chi_square,
)
from core.smr_calculator import calculate_smr, calculate_smr_stratified
from core.direct_standardization import calculate_direct_standardized_rate
from utils.constants import DEMO_2X2_TABLE, STANDARD_POPULATIONS, RATE_MULTIPLIERS
from utils.ui_helpers import plot_download_button

st.set_page_config(page_title="Statistics Calculator - EpiAssist", layout="wide")

st.title("Statistics Calculator")
st.markdown("""
Calculate epidemiological measures from contingency tables (OR, RR, RD, Chi-square),
compute Standardized Mortality/Incidence Ratios (SMR/SIR), or calculate directly
standardized (age-adjusted) rates.
""")

st.divider()

tab_2x2, tab_smr, tab_direct = st.tabs([
    "2x2 Table (OR/RR/RD)", "SMR/SIR Calculator", "Direct Standardization"
])

# ── Tab 1: 2x2 Table ──────────────────────────────────────────────────────
with tab_2x2:
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
        for key in ("a", "b", "c", "d"):
            if f"stats_{key}" not in st.session_state:
                st.session_state[f"stats_{key}"] = DEMO_2X2_TABLE[key]

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

            st.plotly_chart(fig, width="stretch")
            plot_download_button(fig, filename="effect_estimates")

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

# ── Tab 2: SMR/SIR Calculator ─────────────────────────────────────────────
with tab_smr:
    st.markdown("""
    The **Standardized Mortality Ratio (SMR)** or **Standardized Incidence Ratio (SIR)**
    compares observed events to expected events based on reference population rates.

    **SMR = Observed / Expected** (null value = 1.0)
    """)

    smr_mode = st.radio(
        "Input mode",
        ["Simple (totals only)", "Stratified (by age/sex groups)"],
        key="smr_mode",
        horizontal=True,
    )

    if smr_mode == "Simple (totals only)":
        # ── Simple mode ────────────────────────────────────────────────
        col_in, col_out = st.columns([1, 1])

        with col_in:
            st.markdown("### Input")
            smr_observed = st.number_input(
                "Observed events",
                min_value=0,
                value=45,
                key="smr_observed",
            )
            smr_expected = st.number_input(
                "Expected events",
                min_value=0.01,
                value=30.0,
                step=0.1,
                format="%.2f",
                key="smr_expected",
            )
            smr_calc_btn = st.button("Calculate SMR/SIR", type="primary")

        with col_out:
            st.markdown("### Results")
            if smr_calc_btn or "smr_result" in st.session_state:
                try:
                    result = calculate_smr(smr_observed, smr_expected)
                    st.session_state.smr_result = result

                    st.metric(
                        "SMR/SIR",
                        f"{result['value']:.4f}",
                        f"95% CI: {result['ci_lower']:.4f}-{result['ci_upper']:.4f}",
                    )
                    st.markdown(
                        f"**Observed:** {result['observed']} | "
                        f"**Expected:** {result['expected']:.2f}"
                    )
                    with st.expander("Interpretation", expanded=True):
                        st.markdown(result["interpretation"])

                except ValueError as e:
                    st.error(str(e))
            else:
                st.info("Enter observed and expected counts, then click Calculate.")

    else:
        # ── Stratified mode ────────────────────────────────────────────
        st.markdown("### Stratified Data")
        st.markdown(
            "Enter person-time and reference rates by stratum. "
            "The expected events are computed as `person_time * reference_rate`."
        )

        # Default demo strata
        if "smr_strata_df" not in st.session_state:
            st.session_state.smr_strata_df = pd.DataFrame({
                "Stratum": ["20-39", "40-59", "60-79"],
                "Person-Time": [5000.0, 8000.0, 3000.0],
                "Reference Rate": [0.001, 0.004, 0.012],
                "Observed": [8, 40, 42],
            })

        edited_df = st.data_editor(
            st.session_state.smr_strata_df,
            num_rows="dynamic",
            key="smr_editor",
        )

        smr_strat_btn = st.button("Calculate Stratified SMR/SIR", type="primary")

        if smr_strat_btn or "smr_strat_result" in st.session_state:
            # Validate no empty cells before building strata
            if edited_df[["Person-Time", "Reference Rate", "Observed"]].isna().any().any():
                st.error("All strata rows must have values. Please fill in empty cells.")
            else:
                try:
                    strata = [
                        {
                            "stratum_name": row["Stratum"],
                            "person_time": float(row["Person-Time"]),
                            "reference_rate": float(row["Reference Rate"]),
                            "observed": int(row["Observed"]),
                        }
                        for _, row in edited_df.iterrows()
                    ]

                    result = calculate_smr_stratified(strata)
                    st.session_state.smr_strat_result = result

                    st.markdown("### Results")

                    st.metric(
                        "SMR/SIR",
                        f"{result['value']:.4f}",
                        f"95% CI: {result['ci_lower']:.4f}-{result['ci_upper']:.4f}",
                    )
                    st.markdown(
                        f"**Total Observed:** {result['observed']} | "
                        f"**Total Expected:** {result['expected']:.2f} | "
                        f"**Total Person-Time:** {result['total_person_time']:,.0f}"
                    )

                    with st.expander("Interpretation", expanded=True):
                        st.markdown(result["interpretation"])

                    with st.expander("Strata Breakdown"):
                        breakdown = pd.DataFrame(result["strata_details"])
                        st.dataframe(breakdown, use_container_width=True)

                except (ValueError, KeyError) as e:
                    st.error(f"Calculation error: {e}")

# ── Tab 3: Direct Standardization ─────────────────────────────────────────
with tab_direct:
    st.markdown("""
    **Direct standardization** computes age-adjusted rates by applying your study's
    stratum-specific rates to a standard population. This removes the effect of
    differing age distributions and lets you compare rates across populations.
    """)

    col_opts, _ = st.columns([1, 1])
    with col_opts:
        std_pop_name = st.selectbox(
            "Standard population",
            list(STANDARD_POPULATIONS.keys()) + ["Custom"],
            key="direct_std_pop",
        )
        multiplier_label = st.selectbox(
            "Rate multiplier",
            list(RATE_MULTIPLIERS.keys()),
            index=2,  # default "100,000"
            key="direct_multiplier",
        )

    # Demo data matching the US 2000 standard population strata
    _DEMO_EVENTS = [50, 30, 40, 60, 90, 150, 250, 400, 500, 300]
    _DEMO_POP = [25000, 45000, 50000, 55000, 48000, 42000, 35000, 25000, 15000, 5000]

    # Build default DataFrame from selected standard population
    if "direct_strata_df" not in st.session_state or st.session_state.get("_direct_last_pop") != std_pop_name:
        if std_pop_name != "Custom":
            pop = STANDARD_POPULATIONS[std_pop_name]
            st.session_state.direct_strata_df = pd.DataFrame({
                "Stratum": [s["stratum_name"] for s in pop],
                "Events": _DEMO_EVENTS[:len(pop)],
                "Population": _DEMO_POP[:len(pop)],
                "Standard Weight": [s["weight"] for s in pop],
            })
        else:
            st.session_state.direct_strata_df = pd.DataFrame({
                "Stratum": ["Group 1", "Group 2", "Group 3"],
                "Events": [0, 0, 0],
                "Population": [0, 0, 0],
                "Standard Weight": [0, 0, 0],
            })
        st.session_state._direct_last_pop = std_pop_name

    st.markdown("### Stratum Data")
    edited_direct_df = st.data_editor(
        st.session_state.direct_strata_df,
        num_rows="dynamic",
        key="direct_editor",
        column_config={
            "Events": st.column_config.NumberColumn("Events", min_value=0, step=1, format="%d"),
            "Population": st.column_config.NumberColumn("Population", min_value=0, step=1, format="%d"),
        },
    )

    direct_calc_btn = st.button("Calculate Adjusted Rate", type="primary")

    if direct_calc_btn or "direct_result" in st.session_state:
        # NaN guard
        if edited_direct_df[["Events", "Population", "Standard Weight"]].isna().any().any():
            st.error("All rows must have values. Please fill in empty cells.")
        else:
            try:
                strata = [
                    {
                        "stratum_name": row["Stratum"],
                        "events": int(row["Events"]),
                        "population": int(row["Population"]),
                        "standard_weight": int(row["Standard Weight"]),
                    }
                    for _, row in edited_direct_df.iterrows()
                ]

                multiplier = RATE_MULTIPLIERS[multiplier_label]
                result = calculate_direct_standardized_rate(strata, multiplier=multiplier)
                st.session_state.direct_result = result

                st.markdown("### Results")

                col_adj, col_crude = st.columns(2)
                with col_adj:
                    st.metric(
                        "Adjusted Rate",
                        f"{result['value']:.2f}",
                        f"95% CI: {result['ci_lower']:.2f}-{result['ci_upper']:.2f}",
                    )
                with col_crude:
                    st.metric(
                        "Crude Rate",
                        f"{result['crude_rate']:.2f}",
                        f"{multiplier_label}",
                    )

                with st.expander("Strata Breakdown"):
                    breakdown = pd.DataFrame(result["strata_details"])
                    st.dataframe(breakdown, use_container_width=True)

                with st.expander("Interpretation", expanded=True):
                    st.markdown(result["interpretation"])

            except (ValueError, KeyError) as e:
                st.error(f"Calculation error: {e}")
