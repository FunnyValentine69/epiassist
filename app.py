"""EpiAssist - Epidemiological Research Assistant.

Main entry point for the Streamlit application.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="EpiAssist",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "dag_graph" not in st.session_state:
    st.session_state.dag_graph = None
if "dag_exposure" not in st.session_state:
    st.session_state.dag_exposure = None
if "dag_outcome" not in st.session_state:
    st.session_state.dag_outcome = None
if "dag_confounders" not in st.session_state:
    st.session_state.dag_confounders = []

# Main page content
st.title("EpiAssist")
st.subheader("Epidemiological Research Assistant")

st.markdown("""
**EpiAssist** helps epidemiology researchers save hundreds of hours by automating
common tasks in causal inference, statistical analysis, and literature review.
""")

# Workflow diagram
st.markdown("### Workflow")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    **1. Build DAG**

    Define variables and
    causal relationships
    """)

with col2:
    st.markdown("""
    **2. Identify Confounders**

    Automatic detection
    using graph theory
    """)

with col3:
    st.markdown("""
    **3. Calculate Statistics**

    OR, RR, RD with
    95% CI and p-values
    """)

with col4:
    st.markdown("""
    **4. Test Hypotheses**

    Formal hypothesis
    testing framework
    """)

with col5:
    st.markdown("""
    **5. Analyze Papers**

    Extract statistics
    from PDF papers
    """)

st.divider()

# Features section
st.markdown("### Features")

features = {
    "DAG Builder": "Create and visualize Directed Acyclic Graphs for causal inference. Automatically detect confounders using the backdoor criterion.",
    "Statistics Calculator": "Calculate Odds Ratios, Risk Ratios, Risk Differences, and Chi-square tests with confidence intervals and plain English interpretations.",
    "Hypothesis Testing": "Structure research questions into formal hypotheses. Get guidance on study design and bias assessment.",
    "Paper Analyzer": "Upload epidemiological papers (PDF) and automatically extract reported statistics including ORs, CIs, and p-values.",
    "Power Analysis": "Calculate required sample sizes, generate power curves, and perform E-value sensitivity analysis for unmeasured confounding.",
}

for feature, description in features.items():
    with st.expander(f"**{feature}**"):
        st.write(description)

st.divider()

# Demo use case
st.markdown("### Demo Use Case: Hearing Loss and Unemployment")

st.info("""
**Research Question:** Is hearing loss associated with unemployment?

This demo uses simulated NHANES-like data to investigate the relationship between
hearing loss (exposure) and unemployment (outcome), while accounting for potential
confounders like Age, Education, and Depression.

**Sample Data:**
- Exposed (Hearing Loss): 80 unemployed, 150 employed (n=230)
- Unexposed: 70 unemployed, 400 employed (n=470)
- **Crude OR = 3.05** (hearing loss associated with 3x higher unemployment odds)
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    EpiAssist v0.1.0 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
