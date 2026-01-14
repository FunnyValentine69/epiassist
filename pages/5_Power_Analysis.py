"""Power Analysis page for sample size and E-value calculations."""

import streamlit as st
import plotly.graph_objects as go

from core.power_calculator import (
    calculate_sample_size,
    calculate_power,
    generate_power_curve,
    calculate_sample_size_for_or,
    classify_effect_size,
)
from core.e_value import calculate_e_value_for_or
from utils.constants import ALPHA_DEFAULT, POWER_DEFAULT
from utils.interpretations import interpret_power

st.set_page_config(page_title="Power Analysis - EpiAssist", layout="wide")

st.title("Power Analysis")
st.markdown("""
Calculate required sample sizes for your study and assess the robustness
of findings using E-value sensitivity analysis.
""")

st.divider()

tab1, tab2 = st.tabs(["Sample Size Calculator", "E-Value Calculator"])

with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Parameters")

        calc_mode = st.radio(
            "Calculation mode",
            ["Effect Size (Cohen's h)", "Detect Specific OR"],
            horizontal=True,
        )

        if calc_mode == "Effect Size (Cohen's h)":
            effect_size = st.slider(
                "Expected effect size (Cohen's h)",
                min_value=0.1,
                max_value=1.5,
                value=0.5,
                step=0.05,
                help="Small=0.2, Medium=0.5, Large=0.8",
            )
            st.caption(f"Effect size classification: **{classify_effect_size(effect_size)}**")
        else:
            st.markdown("#### Expected Odds Ratio")
            p0 = st.number_input(
                "Baseline probability (unexposed group)",
                min_value=0.01,
                max_value=0.99,
                value=0.15,
                format="%.2f",
                help="Probability of outcome in unexposed group",
            )
            target_or = st.number_input(
                "Expected Odds Ratio to detect",
                min_value=1.1,
                max_value=20.0,
                value=3.0,
                format="%.1f",
                help="The OR you want to be able to detect",
            )

        alpha = st.selectbox(
            "Significance level (alpha)",
            [0.05, 0.01, 0.10],
            index=0,
        )

        power = st.slider(
            "Desired power (1 - beta)",
            min_value=0.70,
            max_value=0.95,
            value=0.80,
            step=0.05,
        )

        calculate_btn = st.button("Calculate Sample Size", type="primary")

    with col2:
        st.markdown("### Results")

        if calculate_btn:
            if calc_mode == "Effect Size (Cohen's h)":
                n = calculate_sample_size(effect_size, alpha, power)
                st.session_state.power_n = n
                st.session_state.power_effect = effect_size
                st.session_state.power_alpha = alpha
                st.session_state.power_power = power

                st.metric("Required Sample Size (per group)", f"{n:,}")
                st.metric("Total Sample Size", f"{n * 2:,}")

                st.markdown("#### Interpretation")
                st.info(
                    f"To detect an effect size of {effect_size} ({classify_effect_size(effect_size)}) "
                    f"with {power*100:.0f}% power at α = {alpha}, "
                    f"you need **{n:,} participants per group** ({n * 2:,} total)."
                )
            else:
                result = calculate_sample_size_for_or(p0, target_or, alpha, power)
                st.session_state.power_or_result = result

                if result["n_total"]:
                    col_exp, col_unexp = st.columns(2)
                    with col_exp:
                        st.metric("Exposed Group", f"{result['n_exposed']:,}")
                    with col_unexp:
                        st.metric("Unexposed Group", f"{result['n_unexposed']:,}")

                    st.metric("Total Sample Size", f"{result['n_total']:,}")

                    st.markdown("#### Interpretation")
                    st.info(result["interpretation"])
                else:
                    st.error(result["interpretation"])

        elif "power_n" in st.session_state:
            n = st.session_state.power_n
            st.metric("Required Sample Size (per group)", f"{n:,}")
            st.metric("Total Sample Size", f"{n * 2:,}")
        else:
            st.info("Set parameters and click Calculate to see required sample size.")

        # Power curve
        st.markdown("### Power Curve")

        if calculate_btn or "power_effect" in st.session_state:
            if calc_mode == "Effect Size (Cohen's h)" or "power_effect" in st.session_state:
                eff = effect_size if calc_mode == "Effect Size (Cohen's h)" else st.session_state.get("power_effect", 0.5)
                alp = alpha if calculate_btn else st.session_state.get("power_alpha", 0.05)

                curve_data = generate_power_curve(eff, alp, (10, 500))

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=curve_data["n"],
                        y=curve_data["power"],
                        mode="lines",
                        name="Power",
                        line=dict(color="#4ECDC4", width=3),
                    )
                )

                # Add reference line at 80% power
                fig.add_hline(y=0.8, line_dash="dash", line_color="gray", annotation_text="80% power")

                # Mark the calculated point
                if "power_n" in st.session_state:
                    fig.add_trace(
                        go.Scatter(
                            x=[st.session_state.power_n],
                            y=[st.session_state.power_power],
                            mode="markers",
                            marker=dict(size=12, color="#FF6B6B"),
                            name="Your study",
                        )
                    )

                fig.update_layout(
                    title=f"Power vs Sample Size (effect size = {eff})",
                    xaxis_title="Sample Size (per group)",
                    yaxis_title="Statistical Power",
                    yaxis=dict(range=[0, 1.05]),
                    height=350,
                )

                st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Calculate sample size to see power curve.")

with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### E-Value Calculator")
        st.markdown("""
        The E-value quantifies how strong unmeasured confounding would need
        to be to explain away an observed association.
        """)

        point_estimate = st.number_input(
            "Point estimate (OR or RR)",
            min_value=0.1,
            max_value=20.0,
            value=3.05,
            format="%.2f",
            help="Your observed odds ratio or risk ratio",
        )

        use_ci = st.checkbox("Include confidence interval", value=True)

        if use_ci:
            ci_col1, ci_col2 = st.columns(2)
            with ci_col1:
                ci_lower = st.number_input(
                    "Lower CI bound",
                    min_value=0.01,
                    max_value=20.0,
                    value=2.10,
                    format="%.2f",
                )
            with ci_col2:
                ci_upper = st.number_input(
                    "Upper CI bound",
                    min_value=0.01,
                    max_value=50.0,
                    value=4.42,
                    format="%.2f",
                )
        else:
            ci_lower = None
            ci_upper = None

        calculate_e_btn = st.button("Calculate E-Value", type="primary")

    with col2:
        st.markdown("### Results")

        if calculate_e_btn:
            result = calculate_e_value_for_or(point_estimate, ci_lower, ci_upper)
            st.session_state.e_value_result = result

        if "e_value_result" in st.session_state:
            result = st.session_state.e_value_result

            if result["e_value"]:
                st.metric("E-value (point estimate)", f"{result['e_value']:.2f}")

                if result["e_value_ci"]:
                    st.metric("E-value (confidence interval)", f"{result['e_value_ci']:.2f}")
                    st.caption("E-value for CI represents the minimum confounding strength to shift the CI to include 1.0")

                st.markdown("#### Interpretation")
                st.info(result["interpretation"])

                # Visual indicator
                e_val = result["e_value"]
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
                    f"""
                    <div style="
                        background-color: {robustness_color};
                        padding: 15px;
                        border-radius: 8px;
                        text-align: center;
                        margin-top: 10px;
                    ">
                        <strong>Robustness: {robustness_text}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error(result["interpretation"])
        else:
            st.info("Enter your estimates to calculate the E-value.")

        st.markdown("### Reference Guide")
        with st.expander("How to interpret E-values"):
            st.markdown("""
            | E-value | Interpretation |
            |---------|----------------|
            | < 2 | Vulnerable - weak confounders could explain the effect |
            | 2 - 3 | Somewhat vulnerable |
            | 3 - 5 | Moderately robust |
            | 5 - 10 | Quite robust |
            | > 10 | Very robust - strong confounding needed |

            The E-value represents the **minimum strength of association** that
            an unmeasured confounder would need with **both** the exposure and
            outcome to fully explain away the observed effect.

            **Example**: E-value = 5.5 means a confounder would need to have
            risk ratios of at least 5.5 with both hearing loss AND unemployment
            (above and beyond measured confounders) to explain away the OR of 3.05.
            """)

st.divider()

st.markdown("### Demo: Hearing Loss Study")
st.markdown("""
For the hearing loss and unemployment study with **OR = 3.05 (95% CI: 2.10-4.42)**:

| Measure | Value | Interpretation |
|---------|-------|----------------|
| E-value (point estimate) | 5.55 | Quite robust to unmeasured confounding |
| E-value (CI bound) | 3.58 | At minimum, moderate unmeasured confounding needed |

**Sample size needed to replicate**:
- At 80% power, α = 0.05, to detect OR = 3.0: approximately 75 per group (150 total)
- The original study had 700 participants, providing very high power (>99%)
""")
