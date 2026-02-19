"""Meta-Analysis page for pooling effect estimates and generating forest/funnel plots."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.meta_analysis import run_meta_analysis
from utils.constants import (
    META_MEASURE_LABELS,
    Z_SCORE_95,
)
from utils.ui_helpers import plot_download_button

st.set_page_config(page_title="Meta-Analysis - EpiAssist", layout="wide")

st.title("Meta-Analysis")
st.markdown("""
Pool effect estimates from multiple studies using inverse-variance fixed-effect
or DerSimonian-Laird random-effects models. Generate forest plots and funnel plots.
""")

st.divider()

# --- Settings ---
col_measure, col_model = st.columns(2)
with col_measure:
    measure_type = st.selectbox(
        "Effect measure",
        list(META_MEASURE_LABELS.keys()),
        format_func=lambda x: f"{x} — {META_MEASURE_LABELS[x]}",
        index=0,
        key="meta_measure_type_select",
    )
with col_model:
    model = st.radio(
        "Model",
        ["Both", "Fixed-effect", "Random-effects"],
        horizontal=True,
        key="meta_model_select",
    )
model_map = {"Both": "both", "Fixed-effect": "fixed", "Random-effects": "random"}
model_key = model_map[model]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Study Data", "Results", "Funnel Plot"])

# --- Tab 1: Study Data ---
with tab1:
    # Import from Paper Analyzer
    if "paper_results" in st.session_state:
        with st.expander("Import from Paper Analyzer"):
            paper_results = st.session_state.paper_results
            all_ems = paper_results.get("effect_measures", [])
            # Also check standalone CIs that could be matched
            standalone_cis = paper_results.get("confidence_intervals", [])

            # Try to enrich effect measures missing CIs with standalone CIs from the same page
            enriched_ems = []
            for em in all_ems:
                if em.get("ci_lower") is not None and em.get("ci_upper") is not None:
                    enriched_ems.append(em)
                else:
                    # Try matching a standalone CI from the same page
                    matched_ci = None
                    for ci in standalone_cis:
                        if ci.get("page") == em.get("page") and ci.get("lower") is not None and ci.get("upper") is not None:
                            matched_ci = ci
                            break
                    if matched_ci:
                        enriched = {**em, "ci_lower": matched_ci["lower"], "ci_upper": matched_ci["upper"]}
                        enriched_ems.append(enriched)
                    else:
                        enriched_ems.append(em)

            importable = [
                em for em in enriched_ems
                if em.get("ci_lower") is not None and em.get("ci_upper") is not None
            ]

            if importable:
                # Group by measure type
                types_found = sorted(set(em.get("type", "unknown") for em in importable))
                st.info(f"Found {len(importable)} effect measure(s) with complete CIs: {', '.join(types_found)}")

                import_type = st.selectbox(
                    "Import measure type",
                    types_found,
                    key="meta_import_type",
                )

                if st.button("Import Studies", key="meta_import_btn"):
                    imported = []
                    skipped_import = 0
                    for i, em in enumerate(importable):
                        if em.get("type") == import_type:
                            try:
                                imported.append({
                                    "Study Name": f"Study p.{em.get('page', '?')} #{i + 1}",
                                    "Effect": float(em["value"]),
                                    "CI Lower": float(em["ci_lower"]),
                                    "CI Upper": float(em["ci_upper"]),
                                })
                            except (ValueError, TypeError):
                                skipped_import += 1

                    if imported:
                        st.session_state.meta_studies_df = pd.DataFrame(imported)
                        msg = f"Imported {len(imported)} {import_type} studies."
                        if skipped_import > 0:
                            msg += f" ({skipped_import} skipped due to invalid values.)"
                        st.success(msg)
                    else:
                        st.warning(f"No {import_type} studies with complete CIs found.")
            elif all_ems:
                n_total = len(all_ems)
                n_no_ci = sum(1 for em in all_ems if em.get("ci_lower") is None or em.get("ci_upper") is None)
                st.warning(
                    f"Found {n_total} effect measure(s) but {n_no_ci} lack confidence intervals. "
                    "You can manually enter CIs in the study data table below."
                )
            else:
                st.warning("No effect measures found in Paper Analyzer results.")

    # Default data
    if "meta_studies_df" not in st.session_state:
        st.session_state.meta_studies_df = pd.DataFrame({
            "Study Name": ["Study 1", "Study 2", "Study 3"],
            "Effect": [1.50, 2.00, 1.80],
            "CI Lower": [1.10, 1.30, 1.20],
            "CI Upper": [2.10, 3.10, 2.70],
        })

    st.markdown("### Enter Study Data")
    st.markdown("Enter effect estimates and 95% confidence intervals for each study.")

    edited_df = st.data_editor(
        st.session_state.meta_studies_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Study Name": st.column_config.TextColumn("Study Name", required=True),
            "Effect": st.column_config.NumberColumn("Effect", required=True, format="%.3f"),
            "CI Lower": st.column_config.NumberColumn("CI Lower", required=True, format="%.3f"),
            "CI Upper": st.column_config.NumberColumn("CI Upper", required=True, format="%.3f"),
        },
        key="meta_data_editor",
    )

    # Persist edits
    st.session_state.meta_studies_df = edited_df

    run_btn = st.button("Run Meta-Analysis", type="primary", key="meta_run_btn")

    if run_btn:
        # Convert dataframe to study dicts
        studies = []
        skipped_rows = 0
        for _, row in edited_df.iterrows():
            if pd.notna(row.get("Effect")) and pd.notna(row.get("CI Lower")) and pd.notna(row.get("CI Upper")):
                try:
                    studies.append({
                        "name": str(row.get("Study Name", "")),
                        "effect": float(row["Effect"]),
                        "ci_lower": float(row["CI Lower"]),
                        "ci_upper": float(row["CI Upper"]),
                    })
                except (ValueError, TypeError):
                    skipped_rows += 1
            else:
                skipped_rows += 1

        if skipped_rows > 0:
            st.warning(f"{skipped_rows} row(s) with missing or invalid data were excluded.")

        try:
            result = run_meta_analysis(studies, measure_type, model_key)

            if "errors" in result:
                for err in result["errors"]:
                    st.error(err)
            else:
                st.session_state.meta_results = result
                st.session_state.meta_measure_type = measure_type
                st.session_state.meta_model = model_key
                st.success("Meta-analysis complete! Switch to the Results tab.")
        except Exception as e:
            st.error(f"Error running meta-analysis: {e}")


# --- Tab 2: Results ---
with tab2:
    if "meta_results" not in st.session_state:
        st.info("Run a meta-analysis from the Study Data tab to see results here.")
    else:
        result = st.session_state.meta_results
        mtype = st.session_state.meta_measure_type
        mmodel = st.session_state.meta_model
        if mtype != measure_type or mmodel != model_key:
            st.warning("Settings have changed since the last run. Re-run the analysis to update results.")
        is_ratio = result["is_ratio"]
        null_value = result["null_value"]
        label = META_MEASURE_LABELS.get(mtype, mtype)

        # --- Pooled Estimates ---
        st.markdown("### Pooled Estimates")

        cols = st.columns(3 if result["fixed"] and result["random"] else 2)

        if result["fixed"]:
            with cols[0]:
                st.metric(
                    f"Fixed-Effect {label}",
                    f"{result['fixed']['value']:.3f}",
                )
                st.markdown(
                    f"95% CI: {result['fixed']['ci_lower']:.3f} – {result['fixed']['ci_upper']:.3f}"
                )
                st.markdown(f"z = {result['fixed']['z_value']:.3f}, p = {result['fixed']['p_value']:.4f}")

        if result["random"]:
            idx = 1 if result["fixed"] else 0
            with cols[idx]:
                st.metric(
                    f"Random-Effects {label}",
                    f"{result['random']['value']:.3f}",
                )
                st.markdown(
                    f"95% CI: {result['random']['ci_lower']:.3f} – {result['random']['ci_upper']:.3f}"
                )
                st.markdown(f"z = {result['random']['z_value']:.3f}, p = {result['random']['p_value']:.4f}")
                pi = result["random"]["prediction_interval"]
                pi_note = result["random"].get("prediction_interval_note")
                st.markdown(f"Prediction interval: {pi[0]:.3f} – {pi[1]:.3f}")
                if pi_note:
                    st.markdown(f"Note: {pi_note}")

        het_idx = -1
        with cols[het_idx]:
            het = result["heterogeneity"]
            st.metric("I²", f"{het['i_squared']:.1f}%")
            st.markdown(f"Q = {het['q_statistic']:.2f}, p = {het['q_p_value']:.4f}")
            st.markdown(f"τ² = {het['tau_squared']:.4f}")

        # --- Interpretations ---
        st.markdown("### Interpretation")

        if result["fixed"]:
            st.info(result["fixed"]["interpretation"])
        if result["random"]:
            st.info(result["random"]["interpretation"])
        st.info(het["interpretation"])

        # --- Forest Plot ---
        st.markdown("### Forest Plot")

        # Decide which model's weights to show primarily
        primary = result["random"] if result["random"] else result["fixed"]
        studies = result["studies"]
        weights = primary["weights"]

        fig = go.Figure()

        # Individual studies
        for i, (s, w) in enumerate(zip(studies, weights)):
            fig.add_trace(go.Scatter(
                x=[s["effect"]],
                y=[s["name"]],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[s["ci_upper"] - s["effect"]],
                    arrayminus=[s["effect"] - s["ci_lower"]],
                    thickness=1.5,
                ),
                mode="markers",
                marker=dict(
                    size=max(6, w * 0.8),
                    color="#4ECDC4",
                    symbol="square",
                ),
                name=f"{s['name']} ({w:.1f}%)",
                showlegend=False,
                hovertemplate=(
                    f"<b>{s['name']}</b><br>"
                    f"{label}: {s['effect']:.3f}<br>"
                    f"95% CI: {s['ci_lower']:.3f} – {s['ci_upper']:.3f}<br>"
                    f"Weight: {w:.1f}%<extra></extra>"
                ),
            ))

        # Pooled diamond
        pooled_val = primary["value"]
        pooled_lo = primary["ci_lower"]
        pooled_hi = primary["ci_upper"]
        pool_label = "Random-effects" if result["random"] else "Fixed-effect"

        fig.add_trace(go.Scatter(
            x=[pooled_lo, pooled_val, pooled_hi, pooled_val, pooled_lo],
            y=["Pooled", "Pooled", "Pooled", "Pooled", "Pooled"],
            mode="lines",
            fill="toself",
            fillcolor="rgba(255, 107, 107, 0.5)",
            line=dict(color="#FF6B6B", width=2),
            name=f"{pool_label} pooled",
            showlegend=False,
            hovertemplate=(
                f"<b>{pool_label} Pooled</b><br>"
                f"{label}: {pooled_val:.3f}<br>"
                f"95% CI: {pooled_lo:.3f} – {pooled_hi:.3f}<extra></extra>"
            ),
        ))

        # Null reference line
        fig.add_vline(
            x=null_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"No effect ({null_value})",
        )

        # Weight legend annotation
        fig.add_annotation(
            text="Node size = study weight",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.02,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )

        # Order y-axis: studies at top, pooled at bottom
        study_names = [s["name"] for s in studies]
        y_order = ["Pooled"] + list(reversed(study_names))

        fig.update_layout(
            height=max(300, 80 * (len(studies) + 1)),
            yaxis=dict(
                categoryorder="array",
                categoryarray=y_order,
            ),
            xaxis_title=label,
            margin=dict(l=10, r=10, t=30, b=40),
        )

        if is_ratio:
            fig.update_xaxes(type="log")

        st.plotly_chart(fig, use_container_width=True)
        plot_download_button(fig, filename="forest_plot")

        # --- Study Weights Table ---
        st.markdown("### Study Weights")

        weight_data = []
        for s, w_f, w_r in zip(
            studies,
            result["fixed"]["weights"] if result["fixed"] else [None] * len(studies),
            result["random"]["weights"] if result["random"] else [None] * len(studies),
        ):
            row = {
                "Study": s["name"],
                label: f"{s['effect']:.3f}",
                "95% CI": f"{s['ci_lower']:.3f} – {s['ci_upper']:.3f}",
            }
            if w_f is not None:
                row["Fixed Weight (%)"] = f"{w_f:.1f}"
            if w_r is not None:
                row["Random Weight (%)"] = f"{w_r:.1f}"
            weight_data.append(row)

        st.dataframe(pd.DataFrame(weight_data), use_container_width=True, hide_index=True)


# --- Tab 3: Funnel Plot ---
with tab3:
    if "meta_results" not in st.session_state:
        st.info("Run a meta-analysis from the Study Data tab to see the funnel plot here.")
    else:
        result = st.session_state.meta_results
        mtype = st.session_state.meta_measure_type
        is_ratio = result["is_ratio"]
        label = META_MEASURE_LABELS.get(mtype, mtype)
        studies = result["studies"]

        # Use random-effects pooled if available, else fixed
        primary = result["random"] if result["random"] else result["fixed"]

        st.markdown("### Funnel Plot")
        st.markdown("Asymmetry in the funnel plot may indicate publication bias.")

        # Get pooled estimate on analysis scale
        if is_ratio:
            pooled_theta = math.log(primary["value"])
            effects = [s["theta"] for s in studies]
        else:
            pooled_theta = primary["value"]
            effects = [s["theta"] for s in studies]

        se_values = [s["se"] for s in studies]

        fig = go.Figure()

        # Study points
        fig.add_trace(go.Scatter(
            x=effects,
            y=se_values,
            mode="markers",
            marker=dict(size=10, color="#4ECDC4"),
            text=[s["name"] for s in studies],
            hovertemplate="<b>%{text}</b><br>Effect: %{x:.3f}<br>SE: %{y:.3f}<extra></extra>",
            showlegend=False,
        ))

        # Pooled estimate vertical line
        fig.add_vline(
            x=pooled_theta,
            line_dash="solid",
            line_color="#FF6B6B",
            annotation_text="Pooled",
        )

        # Pseudo 95% CI funnel
        se_max = max(se_values) * 1.1
        se_range = [0, se_max]
        funnel_left = [pooled_theta - Z_SCORE_95 * se for se in se_range]
        funnel_right = [pooled_theta + Z_SCORE_95 * se for se in se_range]

        fig.add_trace(go.Scatter(
            x=funnel_left + funnel_right[::-1],
            y=se_range + se_range[::-1],
            fill="toself",
            fillcolor="rgba(200, 200, 200, 0.2)",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.update_layout(
            xaxis_title=f"ln({label})" if is_ratio else label,
            yaxis_title="Standard Error",
            yaxis=dict(autorange="reversed"),
            height=450,
            margin=dict(l=10, r=10, t=30, b=40),
        )

        st.plotly_chart(fig, use_container_width=True)
        plot_download_button(fig, filename="funnel_plot")

        st.markdown("#### Interpreting the Funnel Plot")
        st.markdown("""
        - **Symmetric funnel**: No evidence of publication bias
        - **Asymmetric funnel**: May indicate publication bias, small-study effects, or heterogeneity
        - **Missing bottom-left/right**: Suggests small studies with non-significant results may be unpublished
        """)

# --- Footer ---
st.divider()

st.markdown("### How It Works")
with st.expander("Methods"):
    st.markdown("""
    **Fixed-Effect Model** (Inverse Variance):
    - Assumes one true effect across all studies
    - Weights studies by 1/variance (larger studies weighted more)

    **Random-Effects Model** (DerSimonian-Laird):
    - Assumes study effects vary around a mean
    - Adds between-study variance (τ²) to weights
    - Prediction interval shows where future studies might fall

    **Heterogeneity:**
    - **Q statistic**: Tests whether studies share a common effect
    - **I²**: Percentage of variability due to heterogeneity (not chance)
    - **τ²**: Estimated between-study variance

    **Ratio measures** (OR, RR, HR, PR, IRR) are analyzed on the log scale
    and back-transformed for display.
    """)
