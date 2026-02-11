"""DAG Builder page for creating and visualizing causal diagrams."""

import re

import streamlit as st

from core.dag_engine import DAGEngine
from core.confounder_detector import (
    compare_adjustment_sets,
    suggest_adjustment_set,
)
from utils.constants import DEMO_DAG_NODES, DEMO_DAG_EDGES

st.set_page_config(page_title="DAG Builder - EpiAssist", layout="wide")

# Initialize session state
if "dag_engine" not in st.session_state:
    st.session_state.dag_engine = DAGEngine()
if "dag_exposure" not in st.session_state:
    st.session_state.dag_exposure = None
if "dag_outcome" not in st.session_state:
    st.session_state.dag_outcome = None


def load_demo_dag() -> None:
    """Load the demo DAG (Hearing Loss -> Unemployment)."""
    engine = DAGEngine()
    for node in DEMO_DAG_NODES:
        engine.add_node(node["name"], node["type"])
    for source, target in DEMO_DAG_EDGES:
        engine.add_edge(source, target)

    st.session_state.dag_engine = engine
    st.session_state.dag_exposure = "Hearing Loss"
    st.session_state.dag_outcome = "Unemployment"


st.title("DAG Builder")
st.markdown("""
Build Directed Acyclic Graphs (DAGs) to represent causal relationships between variables.
Automatically detect confounders and identify adjustment sets.
""")

st.divider()

# Get current engine
engine: DAGEngine = st.session_state.dag_engine
nodes = engine.nodes

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Add Variables")

    with st.expander("What are variable types?", expanded=False):
        st.markdown("""
        - **Exposure**: The main variable of interest (risk factor)
        - **Outcome**: The result you're measuring
        - **Confounder**: Variables that affect both exposure and outcome
        - **Mediator**: Variables on the causal pathway between exposure and outcome
        """)

    with st.form("add_node_form"):
        node_name = st.text_input("Variable name", placeholder="e.g., Hearing Loss")
        node_type = st.selectbox(
            "Variable type",
            ["exposure", "outcome", "confounder", "mediator"]
        )
        add_node_btn = st.form_submit_button("Add Variable")

        if add_node_btn and node_name:
            if node_name in nodes:
                st.error(f"Variable '{node_name}' already exists!")
            else:
                engine.add_node(node_name, node_type)
                st.success(f"Added '{node_name}' as {node_type}")
                st.rerun()

    st.markdown("### Add Relationships")

    if len(nodes) >= 2:
        with st.form("add_edge_form"):
            source = st.selectbox("From variable", nodes, key="edge_source")
            target = st.selectbox("To variable", nodes, key="edge_target")
            add_edge_btn = st.form_submit_button("Add Edge")

            if add_edge_btn:
                if source == target:
                    st.error("Cannot create self-loop!")
                elif (source, target) in engine.edges:
                    st.error("Edge already exists!")
                else:
                    try:
                        engine.add_edge(source, target)
                        st.success(f"Added edge: {source} → {target}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
    else:
        st.info("Add at least 2 variables to create relationships.")

    # Remove section
    st.markdown("### Remove")

    if nodes:
        col_rm1, col_rm2 = st.columns(2)
        with col_rm1:
            node_to_remove = st.selectbox("Remove variable", [""] + nodes)
            if st.button("Remove", key="rm_node") and node_to_remove:
                engine.remove_node(node_to_remove)
                st.rerun()

        with col_rm2:
            if engine.edges:
                edge_options = [f"{s} → {t}" for s, t in engine.edges]
                edge_to_remove = st.selectbox("Remove edge", [""] + edge_options)
                if st.button("Remove", key="rm_edge") and edge_to_remove:
                    parts = edge_to_remove.split(" → ")
                    engine.remove_edge(parts[0], parts[1])
                    st.rerun()

with col2:
    st.markdown("### DAG Visualization")

    if nodes:
        # Render the DAG
        dot = engine.render_graphviz()
        st.graphviz_chart(dot.source)

        # Legend
        st.markdown("""
        **Legend:**
        <span style="color:#FF6B6B">&#9679;</span> Exposure |
        <span style="color:#4ECDC4">&#9679;</span> Outcome |
        <span style="color:#FFE66D">&#9679;</span> Confounder |
        <span style="color:#A78BFA">&#9679;</span> Mediator
        """, unsafe_allow_html=True)
    else:
        st.info("Add variables and relationships to see your DAG here.")

    st.markdown("### Confounder Analysis")

    if nodes:
        # Select exposure and outcome
        exposures = engine.get_nodes_by_type("exposure")
        outcomes = engine.get_nodes_by_type("outcome")

        col_exp, col_out = st.columns(2)

        with col_exp:
            if exposures:
                st.session_state.dag_exposure = st.selectbox(
                    "Exposure variable",
                    exposures,
                    index=exposures.index(st.session_state.dag_exposure)
                    if st.session_state.dag_exposure in exposures
                    else 0,
                )
            else:
                st.warning("No exposure variable defined")

        with col_out:
            if outcomes:
                st.session_state.dag_outcome = st.selectbox(
                    "Outcome variable",
                    outcomes,
                    index=outcomes.index(st.session_state.dag_outcome)
                    if st.session_state.dag_outcome in outcomes
                    else 0,
                )
            else:
                st.warning("No outcome variable defined")

        # Run confounder analysis
        if st.session_state.dag_exposure and st.session_state.dag_outcome:
            if st.button("Detect Confounders"):
                adjustment_set = suggest_adjustment_set(
                    engine.graph,
                    st.session_state.dag_exposure,
                    st.session_state.dag_outcome,
                )

                if adjustment_set:
                    st.success(f"**Confounders found:** {', '.join(adjustment_set)}")
                    st.info(f"**Suggested adjustment set:** {', '.join(adjustment_set)}")
                    st.markdown("""
                    To estimate the causal effect of the exposure on the outcome,
                    you should adjust for these variables in your analysis.
                    """)

                    # Paper Analyzer comparison
                    if "paper_results" in st.session_state:
                        try:
                            paper_results = st.session_state.paper_results
                            adjusted_measures = [
                                m
                                for m in paper_results.get("effect_measures", [])
                                if m.get("adjusted") is True
                                and isinstance(m.get("adjusted_for"), str)
                                and m.get("adjusted_for")
                            ]

                            with st.expander("Compare with Paper Adjustments", expanded=False):
                                if not adjusted_measures:
                                    st.info(
                                        "No adjusted effect measures with named covariates "
                                        "found in the analyzed paper."
                                    )
                                else:
                                    for measure in adjusted_measures:
                                        paper_vars = [
                                            v.strip()
                                            for v in re.split(r",|\band\b", measure["adjusted_for"])
                                            if v.strip()
                                        ]
                                        comparison = compare_adjustment_sets(
                                            adjustment_set, paper_vars
                                        )

                                        label = f"{measure['type']} = {measure['value']}"
                                        st.markdown(f"**{label}** (adjusted for: {measure['adjusted_for']})")

                                        if comparison["overlap"]:
                                            st.markdown(f"Overlap: {', '.join(comparison['overlap'])}")
                                        if comparison["dag_only"]:
                                            st.warning(
                                                f"In DAG but not in paper: {', '.join(comparison['dag_only'])}"
                                            )
                                        if comparison["paper_only"]:
                                            st.info(
                                                f"In paper but not in DAG: {', '.join(comparison['paper_only'])}"
                                            )
                        except Exception as e:
                            st.warning(f"Could not compare with paper adjustments: {e}")
                else:
                    st.info("No confounders detected between exposure and outcome.")
    else:
        st.warning("Add variables to perform confounder analysis.")

st.divider()

# Demo and export section
col_demo, col_export = st.columns(2)

with col_demo:
    st.markdown("### Demo: Hearing Loss and Unemployment")
    st.markdown("""
    Click "Load Demo" to see a pre-built DAG examining the relationship between
    hearing loss and unemployment, with common confounders.
    """)
    if st.button("Load Demo"):
        load_demo_dag()
        st.rerun()

with col_export:
    st.markdown("### Export DAG")
    if nodes:
        dag_data = engine.to_dict()
        st.download_button(
            "Download DAG (JSON)",
            data=str(dag_data),
            file_name="dag_export.json",
            mime="application/json",
        )

        if st.button("Clear DAG"):
            st.session_state.dag_engine = DAGEngine()
            st.session_state.dag_exposure = None
            st.session_state.dag_outcome = None
            st.rerun()
