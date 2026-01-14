"""Hypothesis Testing page for formal statistical inference."""

import subprocess

import streamlit as st

from utils.constants import ALPHA_DEFAULT

st.set_page_config(page_title="Hypothesis Testing - EpiAssist", layout="wide")

st.title("Hypothesis Testing")
st.markdown("""
Formulate null and alternative hypotheses in plain English.
Determine whether to reject or fail to reject the null hypothesis.
""")

st.divider()


def call_ollama(question: str) -> str | None:
    """Call Ollama to generate hypothesis framework from research question.

    Args:
        question: The research question to analyze.

    Returns:
        Generated text or None if failed.
    """
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
        result = subprocess.run(
            ["ollama", "run", "llama3.2:latest", prompt],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def parse_ollama_response(response: str) -> dict:
    """Parse Ollama response to extract key fields.

    Args:
        response: Raw response from Ollama.

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
    generate_btn = st.button("Auto-Generate with AI", type="secondary")

if generate_btn and research_question:
    with st.spinner("Generating with Ollama (llama3.2:latest)..."):
        response = call_ollama(research_question)

    if response:
        parsed = parse_ollama_response(response)
        st.session_state.hypo_ai_response = parsed
        st.session_state.hypo_h0 = parsed["h0"] if parsed["h0"] else st.session_state.get("hypo_h0", "")
        st.session_state.hypo_h1 = parsed["h1"] if parsed["h1"] else st.session_state.get("hypo_h1", "")
        st.success("Generated! Fields auto-filled below.")
    else:
        st.error("Ollama not available. Please enter hypotheses manually.")

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
        st.markdown(
            f"""
        <div style="
            background-color: {'#d4edda' if decision_color == 'green' else '#fff3cd'};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        ">
            <h2 style="color: {'#155724' if decision_color == 'green' else '#856404'}; margin: 0;">
                {decision}
            </h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

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
        st.markdown("""
        **Selection Bias:**
        - Are cases and controls selected from the same source population?
        - Is there differential participation by exposure status?

        **Information Bias:**
        - Is outcome assessment blinded to exposure status?
        - Is measurement error equal across groups?

        **Confounding:**
        - Have all known confounders been measured?
        - Is unmeasured confounding plausible?

        **Temporal Relationship:**
        - Does exposure precede outcome in time?
        - Could reverse causation explain the association?
        """)

    with st.expander("PICO Framework"):
        st.markdown("""
        Structure your research question using PICO:

        - **P**opulation: Who is being studied?
        - **I**ntervention/Exposure: What is the exposure of interest?
        - **C**omparison: What is the reference group?
        - **O**utcome: What outcome is being measured?

        **Example (Hearing Loss Study):**
        - P: Working-age adults
        - I: Hearing loss
        - C: No hearing loss
        - O: Unemployment status
        """)

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
