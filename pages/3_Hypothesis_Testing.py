"""Hypothesis Testing page for formal statistical inference."""

import streamlit as st

from core.llm_providers import detect_provider, get_provider_functions
from utils.constants import ALPHA_DEFAULT
from utils.ui_helpers import styled_banner

st.set_page_config(page_title="Hypothesis Testing - EpiAssist", layout="wide")

st.title("Hypothesis Testing")
st.markdown("""
Formulate null and alternative hypotheses in plain English.
Determine whether to reject or fail to reject the null hypothesis.
""")

st.divider()


def call_llm(question: str) -> str | None:
    """Call the best available LLM to generate hypothesis framework.

    Args:
        question: The research question to analyze.

    Returns:
        Generated text or None if no provider available or on failure.
    """
    provider = detect_provider()
    if provider is None:
        return None

    prompt = f"""You are an epidemiology research assistant. Given this research question: {question}

Generate:
1. PICO Framework:
   - P (Population):
   - I (Intervention/Exposure):
   - C (Comparison):
   - O (Outcome):

2. Potential Biases to Consider:
   - Selection Bias concerns:
   - Information Bias concerns:
   - Confounding concerns:

3. Suggested Null Hypothesis (H0):
4. Suggested Alternative Hypothesis (H1):

Be concise and specific to epidemiological research."""

    try:
        funcs = get_provider_functions(provider)
        return funcs["chat"](prompt)
    except Exception:
        return None


def parse_llm_response(response: str) -> dict:
    """Parse LLM response to extract key fields.

    Args:
        response: Raw response from LLM.

    Returns:
        Dictionary with parsed fields.
    """
    parsed = {
        "pico_p": "",
        "pico_i": "",
        "pico_c": "",
        "pico_o": "",
        "bias_selection": "",
        "bias_information": "",
        "bias_confounding": "",
        "h0": "",
        "h1": "",
        "full_response": response,
    }

    lines = response.split("\n")
    current_section = None

    for line in lines:
        line_lower = line.lower().strip()

        # PICO parsing
        if "p (population)" in line_lower or "- p:" in line_lower:
            parsed["pico_p"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "i (intervention" in line_lower or "- i:" in line_lower:
            parsed["pico_i"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "c (comparison)" in line_lower or "- c:" in line_lower:
            parsed["pico_c"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "o (outcome)" in line_lower or "- o:" in line_lower:
            parsed["pico_o"] = line.split(":", 1)[-1].strip() if ":" in line else ""

        # Bias parsing
        elif "selection bias" in line_lower:
            parsed["bias_selection"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "information bias" in line_lower:
            parsed["bias_information"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "confounding" in line_lower and "bias" not in parsed["bias_confounding"]:
            parsed["bias_confounding"] = line.split(":", 1)[-1].strip() if ":" in line else ""

        # Hypothesis parsing
        elif "null hypothesis" in line_lower or "h0:" in line_lower or "(h0)" in line_lower:
            parsed["h0"] = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "alternative hypothesis" in line_lower or "h1:" in line_lower or "(h1)" in line_lower:
            parsed["h1"] = line.split(":", 1)[-1].strip() if ":" in line else ""

    return parsed


# AI-Assisted Research Question Section
st.markdown("### Research Question")
research_question = st.text_input(
    "Enter your research question",
    value="Is hearing loss associated with unemployment?",
    placeholder="e.g., Is smoking associated with lung cancer?",
)

col_btn, col_status = st.columns([1, 2])
with col_btn:
    if detect_provider():
        generate_btn = st.button("Auto-Generate with AI", type="secondary")
    else:
        generate_btn = False

if generate_btn and research_question:
    with st.spinner("Generating with AI..."):
        response = call_llm(research_question)

    if response:
        parsed = parse_llm_response(response)
        st.session_state.hypo_ai_response = parsed
        st.session_state.hypo_h0 = parsed["h0"] if parsed["h0"] else st.session_state.get("hypo_h0", "")
        st.session_state.hypo_h1 = parsed["h1"] if parsed["h1"] else st.session_state.get("hypo_h1", "")
        st.success("Generated! Fields auto-filled below.")
    else:
        st.warning("AI generation failed. The provider may be temporarily unavailable.")

# Show AI response if available
if "hypo_ai_response" in st.session_state:
    with st.expander("View AI-Generated Analysis", expanded=True):
        parsed = st.session_state.hypo_ai_response

        st.markdown("**PICO Framework:**")
        pico_cols = st.columns(4)
        with pico_cols[0]:
            st.markdown(f"**P:** {parsed['pico_p']}")
        with pico_cols[1]:
            st.markdown(f"**I:** {parsed['pico_i']}")
        with pico_cols[2]:
            st.markdown(f"**C:** {parsed['pico_c']}")
        with pico_cols[3]:
            st.markdown(f"**O:** {parsed['pico_o']}")

        st.markdown("**Potential Biases:**")
        if parsed["bias_selection"]:
            st.markdown(f"- Selection: {parsed['bias_selection']}")
        if parsed["bias_information"]:
            st.markdown(f"- Information: {parsed['bias_information']}")
        if parsed["bias_confounding"]:
            st.markdown(f"- Confounding: {parsed['bias_confounding']}")

        with st.expander("Full AI Response"):
            st.text(parsed["full_response"])

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Define Hypotheses")

    # Use session state values if available from AI generation
    default_h0 = st.session_state.get(
        "hypo_h0", "There is no association between hearing loss and unemployment."
    )
    default_h1 = st.session_state.get(
        "hypo_h1", "There is an association between hearing loss and unemployment."
    )

    h0 = st.text_area(
        "Null Hypothesis (H0)",
        value=default_h0,
        help="The hypothesis of no effect or no difference",
        key="h0_input",
    )

    h1 = st.text_area(
        "Alternative Hypothesis (H1)",
        value=default_h1,
        help="The hypothesis you're trying to support",
        key="h1_input",
    )

    st.markdown("### Test Parameters")

    alpha = st.selectbox(
        "Significance level (alpha)",
        [0.05, 0.01, 0.10],
        index=0,
        help="The threshold for statistical significance",
    )

    p_value = st.number_input(
        "P-value from your analysis",
        min_value=0.0,
        max_value=1.0,
        value=0.001,
        format="%.6f",
        help="The p-value from your statistical test",
    )

    evaluate_btn = st.button("Evaluate Hypothesis", type="primary")

with col2:
    st.markdown("### Decision")

    if evaluate_btn:
        # Make decision
        if p_value < alpha:
            decision = "REJECT H0"
            decision_color = "green"
            conclusion = (
                f"At significance level α = {alpha}, "
                f"we **reject the null hypothesis** (p = {p_value:.6f} < {alpha})."
            )
            meaning = (
                "There is sufficient statistical evidence to conclude that "
                f"the alternative hypothesis is supported. "
                f"The observed association is unlikely to be due to chance alone."
            )
        else:
            decision = "FAIL TO REJECT H0"
            decision_color = "orange"
            conclusion = (
                f"At significance level α = {alpha}, "
                f"we **fail to reject the null hypothesis** (p = {p_value:.6f} ≥ {alpha})."
            )
            meaning = (
                "There is insufficient statistical evidence to conclude that "
                f"the alternative hypothesis is supported. "
                f"However, this does NOT mean the null hypothesis is true."
            )

        # Display decision prominently
        styled_banner(decision, "success" if decision_color == "green" else "warning")

        st.markdown("### Conclusion")
        st.markdown(conclusion)

        st.markdown("### What This Means")
        st.info(meaning)

        # Show hypotheses summary
        st.markdown("### Hypotheses Summary")
        st.markdown(f"""
        - **H0**: {h0}
        - **H1**: {h1}
        - **α**: {alpha}
        - **p-value**: {p_value:.6f}
        """)
    else:
        st.info("Enter your hypotheses and p-value, then click Evaluate.")

    st.markdown("### Study Design Considerations")

    with st.expander("Bias Checklist"):
        st.markdown("**Selection Bias:**")
        st.checkbox(
            "Cases and controls selected from the same source population",
            key="bias_selection_1",
        )
        st.checkbox(
            "No differential participation by exposure status",
            key="bias_selection_2",
        )
        st.markdown("**Information Bias:**")
        st.checkbox(
            "Outcome assessment blinded to exposure status",
            key="bias_information_1",
        )
        st.checkbox(
            "Measurement error equal across groups",
            key="bias_information_2",
        )
        st.markdown("**Confounding:**")
        st.checkbox(
            "All known confounders measured",
            key="bias_confounding_1",
        )
        st.checkbox(
            "Unmeasured confounding considered",
            key="bias_confounding_2",
        )
        st.markdown("**Temporal Relationship:**")
        st.checkbox(
            "Exposure precedes outcome in time",
            key="bias_temporal_1",
        )
        st.checkbox(
            "Reverse causation ruled out",
            key="bias_temporal_2",
        )

    with st.expander("PICO Framework"):
        st.markdown("Structure your research question using PICO:")
        ai = st.session_state.get("hypo_ai_response", {})
        # Initialize PICO fields from AI response if not already in session state
        for field in ("pico_p", "pico_i", "pico_c", "pico_o"):
            if field not in st.session_state:
                st.session_state[field] = ai.get(field, "")
        st.text_input(
            "P (Population)",
            key="pico_p",
            placeholder="Who is being studied?",
        )
        st.text_input(
            "I (Intervention/Exposure)",
            key="pico_i",
            placeholder="What is the exposure of interest?",
        )
        st.text_input(
            "C (Comparison)",
            key="pico_c",
            placeholder="What is the reference group?",
        )
        st.text_input(
            "O (Outcome)",
            key="pico_o",
            placeholder="What outcome is being measured?",
        )

st.divider()

st.markdown("### Example: Hearing Loss Study")
st.markdown("""
**Research Question:** Is hearing loss associated with unemployment in working-age adults?

| Parameter | Value |
|---|---|
| H0 | No association between hearing loss and unemployment |
| H1 | Hearing loss is associated with unemployment |
| α | 0.05 |
| p-value | < 0.001 |
| **Decision** | **Reject H0** |

**Interpretation:** There is strong statistical evidence of an association between
hearing loss and unemployment. The odds of unemployment are approximately 3 times
higher among individuals with hearing loss compared to those without.
""")
