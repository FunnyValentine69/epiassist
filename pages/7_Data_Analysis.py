"""Data Analysis page for uploading and analyzing epidemiological datasets."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_analyzer import (
    build_contingency_table,
    descriptive_stats_categorical,
    descriptive_stats_numeric,
    grouped_descriptive_stats,
    load_data,
    summarize_columns,
)
from core.e_value import calculate_e_value_for_or
from core.stats_calculator import (
    calculate_chi_square,
    calculate_mantel_haenszel,
    calculate_odds_ratio,
    calculate_risk_difference,
    calculate_risk_ratio,
)

NUMERIC_STAT_KEYS = ["n", "Mean", "Median", "SD", "Q1", "Q3", "Min", "Max"]
NUMERIC_STAT_FIELDS = ["n", "mean", "median", "sd", "q1", "q3", "min", "max"]


def _numeric_stats_df(stats: dict) -> pd.DataFrame:
    """Build a single-row DataFrame from numeric descriptive stats."""
    return pd.DataFrame([dict(zip(NUMERIC_STAT_KEYS, [stats[f] for f in NUMERIC_STAT_FIELDS]))])

st.set_page_config(page_title="Data Analysis - EpiAssist", layout="wide")

st.title("Data Analysis")
st.markdown("""
Upload a dataset (CSV, Excel, or paste), assign variable roles, explore descriptive
statistics, and generate 2x2 cross-tabulations with effect estimates.
""")

st.divider()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload & Preview", "Variable Roles", "Descriptive Statistics", "Cross-Tabulation"
])

# --- Tab 1: Upload & Preview ---
with tab1:
    upload_mode = st.radio(
        "Data source",
        ["Upload CSV/Excel", "Paste data"],
        horizontal=True,
        key="data_upload_mode",
    )

    df = None

    if upload_mode == "Upload CSV/Excel":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            key="data_file_uploader",
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                fmt = "excel" if uploaded_file.name.endswith((".xlsx", ".xls")) else "csv"
                df = load_data(file_bytes, fmt)
                st.session_state.data_source_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        pasted = st.text_area(
            "Paste your data (tab or comma separated)",
            height=200,
            key="data_paste_area",
        )
        if pasted.strip():
            try:
                df = load_data(pasted, "paste")
                st.session_state.data_source_name = "Pasted data"
            except Exception as e:
                st.error(f"Error parsing pasted data: {e}")

    # Store loaded data and clear stale role assignments
    if df is not None:
        st.session_state.data_df = df
        st.session_state.data_col_summary = summarize_columns(df)
        for key in ["data_outcome_col", "data_outcome_positive",
                     "data_exposure_col", "data_exposure_positive",
                     "data_confounder_cols"]:
            st.session_state.pop(key, None)

    # Clear button
    if "data_df" in st.session_state:
        if st.button("Clear data", key="data_clear_btn"):
            for key in list(st.session_state.keys()):
                if key.startswith("data_"):
                    del st.session_state[key]
            st.rerun()

    # Preview
    if "data_df" in st.session_state:
        df_display = st.session_state.data_df
        source_name = st.session_state.get("data_source_name", "Dataset")

        st.markdown(f"### Preview: {source_name}")
        col_info, row_info = st.columns(2)
        with col_info:
            st.metric("Columns", len(df_display.columns))
        with row_info:
            st.metric("Rows", len(df_display))

        st.dataframe(df_display.head(50), use_container_width=True)

        # Column summary table
        st.markdown("### Column Summary")
        summary_data = [
            {
                "Column": s["column"],
                "Type": s["type"],
                "Missing": f"{s['n_missing']} ({s['pct_missing']}%)",
                "Unique": s["n_unique"],
                "Samples": ", ".join(str(v) for v in s["samples"][:3]),
            }
            for s in st.session_state.data_col_summary
        ]
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Upload or paste data to get started.")


# --- Tab 2: Variable Roles ---
with tab2:
    if "data_df" not in st.session_state:
        st.info("Upload data in the first tab to assign variable roles.")
    else:
        df_roles = st.session_state.data_df
        all_columns = list(df_roles.columns)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Assign Roles")

            # Outcome
            outcome_col = st.selectbox(
                "Outcome variable",
                ["(none)"] + all_columns,
                key="data_outcome_col_select",
            )
            if outcome_col != "(none)":
                st.session_state.data_outcome_col = outcome_col
                outcome_values = sorted(df_roles[outcome_col].dropna().unique().tolist(), key=str)
                outcome_positive = st.selectbox(
                    "Positive outcome value",
                    outcome_values,
                    key="data_outcome_positive_select",
                )
                st.session_state.data_outcome_positive = outcome_positive
            else:
                st.session_state.pop("data_outcome_col", None)
                st.session_state.pop("data_outcome_positive", None)

            # Exposure
            exposure_col = st.selectbox(
                "Exposure variable",
                ["(none)"] + all_columns,
                key="data_exposure_col_select",
            )
            if exposure_col != "(none)":
                st.session_state.data_exposure_col = exposure_col
                exposure_values = sorted(df_roles[exposure_col].dropna().unique().tolist(), key=str)
                exposure_positive = st.selectbox(
                    "Positive exposure value",
                    exposure_values,
                    key="data_exposure_positive_select",
                )
                st.session_state.data_exposure_positive = exposure_positive
            else:
                st.session_state.pop("data_exposure_col", None)
                st.session_state.pop("data_exposure_positive", None)

            # Confounders
            remaining = [c for c in all_columns
                         if c != st.session_state.get("data_outcome_col")
                         and c != st.session_state.get("data_exposure_col")]
            confounder_cols = st.multiselect(
                "Confounder variables (optional)",
                remaining,
                key="data_confounder_cols_select",
            )
            st.session_state.data_confounder_cols = confounder_cols

        with col_right:
            st.markdown("### Variable Preview")

            for role, col_key, pos_key in [
                ("Outcome", "data_outcome_col", "data_outcome_positive"),
                ("Exposure", "data_exposure_col", "data_exposure_positive"),
            ]:
                if col_key in st.session_state:
                    col_name = st.session_state[col_key]
                    pos_val = st.session_state.get(pos_key)
                    st.markdown(f"**{role}: {col_name}**")
                    cat_stats = descriptive_stats_categorical(df_roles[col_name])
                    for cat in cat_stats["categories"][:5]:
                        marker = " **(+)**" if cat["value"] == pos_val else ""
                        st.write(f"- {cat['value']}: {cat['count']} ({cat['proportion']:.1%}){marker}")

            if confounder_cols:
                st.markdown(f"**Confounders:** {', '.join(confounder_cols)}")


# --- Tab 3: Descriptive Statistics ---
with tab3:
    if "data_df" not in st.session_state:
        st.info("Upload data in the first tab to view descriptive statistics.")
    else:
        df_desc = st.session_state.data_df
        col_summary = st.session_state.data_col_summary

        for s in col_summary:
            col_name = s["column"]
            series = df_desc[col_name]

            with st.expander(f"**{col_name}** ({s['type']})", expanded=False):
                if s["type"] == "numeric":
                    # Check if exposure is assigned for grouped stats
                    if "data_exposure_col" in st.session_state:
                        exp_col = st.session_state.data_exposure_col
                        grouped = grouped_descriptive_stats(df_desc, col_name, exp_col)

                        # Side-by-side
                        cols = st.columns(len(grouped))
                        for i, (group, stats) in enumerate(grouped.items()):
                            with cols[i]:
                                st.markdown(f"**{exp_col} = {group}**")
                                if "mean" in stats:
                                    st.dataframe(_numeric_stats_df(stats), hide_index=True)

                        # Histogram overlaid by exposure group
                        fig = go.Figure()
                        for group in grouped:
                            subset = df_desc[df_desc[exp_col] == group][col_name].dropna()
                            fig.add_trace(go.Histogram(
                                x=subset,
                                name=f"{exp_col}={group}",
                                opacity=0.6,
                            ))
                        fig.update_layout(
                            barmode="overlay",
                            title=f"{col_name} by {exp_col}",
                            xaxis_title=col_name,
                            yaxis_title="Count",
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        stats = descriptive_stats_numeric(series)
                        st.dataframe(_numeric_stats_df(stats), hide_index=True)

                        # Simple histogram
                        fig = go.Figure(go.Histogram(x=series.dropna()))
                        fig.update_layout(
                            title=col_name,
                            xaxis_title=col_name,
                            yaxis_title="Count",
                            height=300,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Categorical: frequency table
                    stats = descriptive_stats_categorical(series)
                    freq_df = pd.DataFrame(stats["categories"])
                    if not freq_df.empty:
                        freq_df.columns = ["Value", "Count", "Proportion"]
                        freq_df["Proportion"] = freq_df["Proportion"].apply(lambda x: f"{x:.1%}")
                        st.dataframe(freq_df, hide_index=True, use_container_width=True)
                    if stats["n_missing"] > 0:
                        st.caption(f"Missing: {stats['n_missing']}")


# --- Tab 4: Cross-Tabulation ---
with tab4:
    if "data_df" not in st.session_state:
        st.info("Upload data in the first tab to perform cross-tabulation.")
    elif "data_outcome_col" not in st.session_state or "data_exposure_col" not in st.session_state:
        st.warning("Assign both outcome and exposure variables in the Variable Roles tab first.")
    elif "data_outcome_positive" not in st.session_state or "data_exposure_positive" not in st.session_state:
        st.warning("Select positive values for both outcome and exposure in the Variable Roles tab.")
    else:
        df_ct = st.session_state.data_df
        outcome_col = st.session_state.data_outcome_col
        exposure_col = st.session_state.data_exposure_col
        outcome_pos = st.session_state.data_outcome_positive
        exposure_pos = st.session_state.data_exposure_positive

        try:
            ct = build_contingency_table(df_ct, outcome_col, exposure_col, outcome_pos, exposure_pos)
            a, b, c, d = ct["a"], ct["b"], ct["c"], ct["d"]
        except Exception as e:
            st.error(f"Error building contingency table: {e}")
            st.stop()

        # Display 2x2 table
        st.markdown("### 2x2 Contingency Table")

        table_df = pd.DataFrame(
            [[a, b, a + b], [c, d, c + d], [a + c, b + d, a + b + c + d]],
            columns=[f"{outcome_col}={outcome_pos}", f"{outcome_col}≠{outcome_pos}", "Total"],
            index=[f"{exposure_col}={exposure_pos}", f"{exposure_col}≠{exposure_pos}", "Total"],
        )
        st.dataframe(table_df, use_container_width=True)

        if ct["n_excluded"] > 0:
            st.warning(f"{ct['n_excluded']} rows excluded due to missing values.")

        # Calculate effect estimates
        st.markdown("### Effect Estimates")

        try:
            or_result = calculate_odds_ratio(a, b, c, d)
            rr_result = calculate_risk_ratio(a, b, c, d)
            rd_result = calculate_risk_difference(a, b, c, d)
            chi_result = calculate_chi_square(a, b, c, d)
        except Exception as e:
            st.error(f"Error calculating statistics: {e}")
            st.stop()

        # Metrics
        col_or, col_rr = st.columns(2)
        with col_or:
            st.metric(
                "Odds Ratio (OR)",
                f"{or_result['value']:.2f}" if or_result["value"] is not None else "N/A",
                f"95% CI: {or_result['ci_lower']:.2f}-{or_result['ci_upper']:.2f}"
                if or_result["ci_lower"] is not None else "",
            )
        with col_rr:
            st.metric(
                "Risk Ratio (RR)",
                f"{rr_result['value']:.2f}" if rr_result["value"] is not None else "N/A",
                f"95% CI: {rr_result['ci_lower']:.2f}-{rr_result['ci_upper']:.2f}"
                if rr_result["ci_lower"] is not None else "",
            )

        col_rd, col_chi = st.columns(2)
        with col_rd:
            st.metric(
                "Risk Difference",
                f"{rd_result['value']:.1%}" if rd_result["value"] is not None else "N/A",
                f"95% CI: {rd_result['ci_lower']:.1%}-{rd_result['ci_upper']:.1%}"
                if rd_result["ci_lower"] is not None else "",
            )
        with col_chi:
            p_val = chi_result["p_value"]
            p_display = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
            st.metric(
                "Chi-square",
                f"{chi_result['value']:.2f}",
                f"p-value: {p_display}",
            )

        # Effect estimate chart
        st.markdown("### Effect Estimate Visualization")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
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
        ))

        if rr_result["value"] is not None:
            fig.add_trace(go.Scatter(
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
            ))

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

        # --- Stratified Analysis (Mantel-Haenszel) ---
        mh_result = None
        confounders = st.session_state.get("data_confounder_cols", [])
        if confounders:
            col_summary = st.session_state.data_col_summary
            cat_confounders = [
                c for c in confounders
                if any(s["column"] == c and s["type"] == "categorical" for s in col_summary)
            ]

            if not cat_confounders:
                st.info(
                    "Stratified analysis requires categorical confounders. "
                    "All assigned confounders are numeric."
                )
            else:
                st.divider()
                st.markdown("### Stratified Analysis (Mantel-Haenszel)")

                mh_confounder = st.selectbox(
                    "Stratify by", cat_confounders, key="data_mh_confounder"
                )

                # Build per-stratum 2x2 tables
                strata_list = []
                skipped_strata = []
                for stratum_val, stratum_df in df_ct.groupby(mh_confounder):
                    try:
                        stratum_ct = build_contingency_table(
                            stratum_df, outcome_col, exposure_col,
                            outcome_pos, exposure_pos,
                        )
                        stratum_ct["stratum"] = stratum_val
                        stratum_ct["confounder_name"] = mh_confounder
                        strata_list.append(stratum_ct)
                    except ValueError as e:
                        skipped_strata.append((stratum_val, str(e)))

                if skipped_strata:
                    details = "; ".join(f"{name}: {reason}" for name, reason in skipped_strata)
                    st.warning(f"Excluded {len(skipped_strata)} strata: {details}")

                if len(strata_list) < 2:
                    st.warning(
                        "Need at least 2 valid strata for Mantel-Haenszel analysis. "
                        f"Only {len(strata_list)} stratum has enough data."
                    )
                else:
                    # Stratum-specific tables
                    with st.expander(
                        f"Stratum-specific tables ({len(strata_list)} strata)",
                        expanded=False,
                    ):
                        for s in strata_list:
                            st.markdown(f"**{mh_confounder} = {s['stratum']}**")
                            s_df = pd.DataFrame(
                                [[s["a"], s["b"]], [s["c"], s["d"]]],
                                columns=[f"{outcome_col}+", f"{outcome_col}-"],
                                index=[f"{exposure_col}+", f"{exposure_col}-"],
                            )
                            st.dataframe(s_df, use_container_width=True)

                    try:
                        mh = calculate_mantel_haenszel(strata_list)
                    except Exception as e:
                        st.error(f"Error in Mantel-Haenszel calculation: {e}")
                        st.stop()

                    mh_result = mh

                    # Adjusted metrics
                    st.markdown("#### Adjusted Effect Estimates")
                    col_adj_or, col_adj_rr = st.columns(2)
                    with col_adj_or:
                        st.metric(
                            "MH-Adjusted OR",
                            f"{mh['or_value']:.2f}",
                            f"95% CI: {mh['or_ci_lower']:.2f}-{mh['or_ci_upper']:.2f}",
                        )
                    with col_adj_rr:
                        if mh["rr_value"] is not None:
                            rr_ci = ""
                            if mh["rr_ci_lower"] is not None:
                                rr_ci = f"95% CI: {mh['rr_ci_lower']:.2f}-{mh['rr_ci_upper']:.2f}"
                            st.metric("MH-Adjusted RR", f"{mh['rr_value']:.2f}", rr_ci)
                        else:
                            st.metric("MH-Adjusted RR", "N/A")

                    # Crude vs Adjusted comparison
                    st.markdown("#### Crude vs Adjusted Comparison")
                    col_crude, col_adj = st.columns(2)
                    with col_crude:
                        st.metric("Crude OR", f"{or_result['value']:.2f}")
                    with col_adj:
                        crude_or = or_result["value"]
                        if crude_or is not None and crude_or != 0:
                            pct_change = ((mh["or_value"] - crude_or) / crude_or) * 100
                            delta_text = f"{pct_change:+.1f}% change from crude"
                        else:
                            delta_text = ""
                        st.metric(
                            "Adjusted OR",
                            f"{mh['or_value']:.2f}",
                            delta_text,
                        )

                    # Homogeneity test
                    st.markdown("#### Breslow-Day Homogeneity Test")
                    if mh["homogeneity_p_value"] is not None:
                        bd_p = mh["homogeneity_p_value"]
                        bd_display = "< 0.001" if bd_p < 0.001 else f"{bd_p:.4f}"
                        st.metric(
                            "Breslow-Day Statistic",
                            f"{mh['homogeneity_statistic']:.2f}",
                            f"p-value: {bd_display}",
                        )
                        if bd_p < 0.05:
                            st.warning(
                                "The OR varies significantly across strata — "
                                "possible effect modification. Consider reporting "
                                "stratum-specific estimates instead of the pooled OR."
                            )
                    else:
                        st.info("Homogeneity test requires at least 2 strata.")

                    # Interpretation
                    with st.expander("Mantel-Haenszel Interpretation", expanded=True):
                        st.markdown(mh["interpretation"])

        # --- E-Value: Sensitivity to Unmeasured Confounding ---
        if or_result["value"] is not None:
            try:
                crude_e = calculate_e_value_for_or(
                    or_result["value"], or_result["ci_lower"], or_result["ci_upper"]
                )
            except Exception as e:
                st.warning(f"Could not compute E-value: {e}")
                crude_e = None

            if crude_e is not None and crude_e["e_value"] is not None:
                st.divider()
                st.markdown("### E-Value: Sensitivity to Unmeasured Confounding")
                st.caption(
                    "How strong would unmeasured confounding need to be "
                    "to explain away this association?"
                )

                adjusted_e = None
                if mh_result is not None:
                    try:
                        adjusted_e = calculate_e_value_for_or(
                            mh_result["or_value"],
                            mh_result["or_ci_lower"],
                            mh_result["or_ci_upper"],
                        )
                    except Exception:
                        adjusted_e = None
                    if adjusted_e and adjusted_e["e_value"] is None:
                        adjusted_e = None

                if adjusted_e is not None:
                    col_crude_e, col_adj_e = st.columns(2)
                    with col_crude_e:
                        st.markdown("**Crude OR**")
                        st.metric("E-value (point)", f"{crude_e['e_value']:.2f}")
                        if crude_e["e_value_ci"] is not None:
                            st.metric("E-value (CI)", f"{crude_e['e_value_ci']:.2f}")
                    with col_adj_e:
                        st.markdown("**MH-Adjusted OR**")
                        st.metric("E-value (point)", f"{adjusted_e['e_value']:.2f}")
                        if adjusted_e["e_value_ci"] is not None:
                            st.metric("E-value (CI)", f"{adjusted_e['e_value_ci']:.2f}")
                else:
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        st.metric(
                            "E-value (point estimate)",
                            f"{crude_e['e_value']:.2f}",
                        )
                    with col_e2:
                        if crude_e["e_value_ci"] is not None:
                            st.metric(
                                "E-value (confidence interval)",
                                f"{crude_e['e_value_ci']:.2f}",
                            )

                # Robustness badge
                e_val = crude_e["e_value"]
                if e_val >= 5:
                    robustness_color = "#d4edda"
                    robustness_text = "Quite Robust"
                elif e_val >= 3:
                    robustness_color = "#fff3cd"
                    robustness_text = "Moderately Robust"
                else:
                    robustness_color = "#f8d7da"
                    robustness_text = "Vulnerable"

                st.markdown(
                    f'<div style="background-color: {robustness_color}; '
                    f'padding: 15px; border-radius: 8px; text-align: center; '
                    f'margin-top: 10px;">'
                    f"<strong>Robustness to Unmeasured Confounding: "
                    f"{robustness_text}</strong></div>",
                    unsafe_allow_html=True,
                )

                with st.expander("E-Value Interpretation"):
                    st.markdown(crude_e["interpretation"])
                    if adjusted_e is not None:
                        st.markdown("---")
                        st.markdown(
                            f"**Adjusted OR E-value:** "
                            f"{adjusted_e['interpretation']}"
                        )
