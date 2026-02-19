"""Tests for core/dag_engine.py — DAGEngine class."""

import graphviz
import pytest

from core.dag_engine import DAGEngine


def _make_simple_dag():
    """Helper: create Exposure→Outcome with Age confounder."""
    dag = DAGEngine()
    dag.add_node("Exposure", "exposure")
    dag.add_node("Outcome", "outcome")
    dag.add_node("Age", "confounder")
    dag.add_edge("Age", "Exposure")
    dag.add_edge("Age", "Outcome")
    dag.add_edge("Exposure", "Outcome")
    return dag


# === Add Node ===============================================================


class TestAddNode:
    """Tests for DAGEngine.add_node."""

    def test_add_exposure(self):
        """Should add an exposure node."""
        dag = DAGEngine()
        dag.add_node("Smoking", "exposure")
        assert "Smoking" in dag.nodes

    def test_add_outcome(self):
        """Should add an outcome node."""
        dag = DAGEngine()
        dag.add_node("Cancer", "outcome")
        assert "Cancer" in dag.nodes

    def test_add_confounder(self):
        """Should add a confounder node."""
        dag = DAGEngine()
        dag.add_node("Age", "confounder")
        assert "Age" in dag.nodes

    def test_add_mediator(self):
        """Should add a mediator node."""
        dag = DAGEngine()
        dag.add_node("DNA Damage", "mediator")
        assert "DNA Damage" in dag.nodes

    def test_invalid_type_raises(self):
        """Invalid node type should raise ValueError."""
        dag = DAGEngine()
        with pytest.raises(ValueError, match="node_type must be one of"):
            dag.add_node("X", "invalid_type")

    def test_label_attribute(self):
        """Node should have label attribute matching name."""
        dag = DAGEngine()
        dag.add_node("Smoking", "exposure")
        assert dag.graph.nodes["Smoking"]["label"] == "Smoking"


# === Add Edge ===============================================================


class TestAddEdge:
    """Tests for DAGEngine.add_edge."""

    def test_valid_edge(self):
        """Should add an edge between existing nodes."""
        dag = DAGEngine()
        dag.add_node("A", "exposure")
        dag.add_node("B", "outcome")
        dag.add_edge("A", "B")
        assert ("A", "B") in dag.edges

    def test_missing_source_raises(self):
        """Missing source node should raise ValueError."""
        dag = DAGEngine()
        dag.add_node("B", "outcome")
        with pytest.raises(ValueError, match="Source node"):
            dag.add_edge("A", "B")

    def test_missing_target_raises(self):
        """Missing target node should raise ValueError."""
        dag = DAGEngine()
        dag.add_node("A", "exposure")
        with pytest.raises(ValueError, match="Target node"):
            dag.add_edge("A", "B")

    def test_edge_appears_in_list(self):
        """Added edge should appear in edges property."""
        dag = _make_simple_dag()
        assert ("Exposure", "Outcome") in dag.edges
        assert ("Age", "Exposure") in dag.edges


# === Remove Node and Edge ===================================================


class TestRemoveNodeAndEdge:
    """Tests for remove_node and remove_edge."""

    def test_remove_node(self):
        """Removing a node should remove it from the graph."""
        dag = _make_simple_dag()
        dag.remove_node("Age")
        assert "Age" not in dag.nodes

    def test_remove_node_cascades_edges(self):
        """Removing a node should also remove its edges."""
        dag = _make_simple_dag()
        dag.remove_node("Age")
        assert ("Age", "Exposure") not in dag.edges
        assert ("Age", "Outcome") not in dag.edges

    def test_remove_nonexistent_node_no_op(self):
        """Removing a nonexistent node should not raise or alter the graph."""
        dag = _make_simple_dag()
        nodes_before = set(dag.nodes)
        edges_before = set(dag.edges)
        dag.remove_node("Nonexistent")
        assert set(dag.nodes) == nodes_before
        assert set(dag.edges) == edges_before

    def test_remove_edge(self):
        """Removing an edge should remove only that edge."""
        dag = _make_simple_dag()
        dag.remove_edge("Age", "Exposure")
        assert ("Age", "Exposure") not in dag.edges
        assert ("Age", "Outcome") in dag.edges  # Other edges intact

    def test_remove_nonexistent_edge_no_op(self):
        """Removing a nonexistent edge should not raise or alter the graph."""
        dag = _make_simple_dag()
        edges_before = set(dag.edges)
        dag.remove_edge("Exposure", "Age")  # Wrong direction
        assert set(dag.edges) == edges_before


# === Get All Paths ==========================================================


class TestGetAllPaths:
    """Tests for DAGEngine.get_all_paths."""

    def test_direct_path(self):
        """Should find the direct Exposure→Outcome path."""
        dag = _make_simple_dag()
        paths = dag.get_all_paths("Exposure", "Outcome")
        assert ["Exposure", "Outcome"] in paths

    def test_multiple_paths(self):
        """DAG with mediator should have multiple paths."""
        dag = DAGEngine()
        dag.add_node("E", "exposure")
        dag.add_node("M", "mediator")
        dag.add_node("O", "outcome")
        dag.add_edge("E", "M")
        dag.add_edge("M", "O")
        dag.add_edge("E", "O")
        paths = dag.get_all_paths("E", "O")
        assert len(paths) == 2

    def test_no_path(self):
        """Disconnected nodes should have no paths."""
        dag = DAGEngine()
        dag.add_node("A", "exposure")
        dag.add_node("B", "outcome")
        assert dag.get_all_paths("A", "B") == []

    def test_nonexistent_node(self):
        """Nonexistent source should return empty list."""
        dag = _make_simple_dag()
        assert dag.get_all_paths("Ghost", "Outcome") == []


# === Render Graphviz ========================================================


class TestRenderGraphviz:
    """Tests for DAGEngine.render_graphviz."""

    def test_returns_digraph(self):
        """Should return a graphviz.Digraph object."""
        dag = _make_simple_dag()
        dot = dag.render_graphviz()
        assert isinstance(dot, graphviz.Digraph)

    def test_nodes_in_source(self):
        """All node names should appear in Graphviz source."""
        dag = _make_simple_dag()
        source = dag.render_graphviz().source
        for node in dag.nodes:
            assert node in source

    def test_edges_in_source(self):
        """Edges should appear in Graphviz source (as source -> target)."""
        dag = _make_simple_dag()
        source = dag.render_graphviz().source
        assert "Exposure -> Outcome" in source
        assert "Age -> Exposure" in source
        assert "Age -> Outcome" in source


# === Serialization ==========================================================


class TestSerialization:
    """Tests for to_dict and from_dict."""

    def test_to_dict_structure(self):
        """to_dict should return nodes and edges lists."""
        dag = _make_simple_dag()
        data = dag.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 3

    def test_roundtrip(self):
        """to_dict → from_dict should reproduce the same graph."""
        dag = _make_simple_dag()
        data = dag.to_dict()
        dag2 = DAGEngine()
        dag2.from_dict(data)
        assert set(dag.nodes) == set(dag2.nodes)
        assert set(dag.edges) == set(dag2.edges)

    def test_from_dict_clears_existing(self):
        """from_dict should clear existing nodes before loading."""
        dag = _make_simple_dag()
        dag.from_dict({"nodes": [{"name": "X", "type": "exposure"}], "edges": []})
        assert dag.nodes == ["X"]

    def test_empty_dict(self):
        """from_dict with empty data should produce empty graph."""
        dag = _make_simple_dag()
        dag.from_dict({"nodes": [], "edges": []})
        assert dag.nodes == []
        assert dag.edges == []

    def test_node_types_preserved(self):
        """Node types should survive serialization roundtrip."""
        dag = _make_simple_dag()
        data = dag.to_dict()
        dag2 = DAGEngine()
        dag2.from_dict(data)
        assert dag2.get_node_type("Exposure") == "exposure"
        assert dag2.get_node_type("Age") == "confounder"


# === Node Queries ===========================================================


class TestNodeQueries:
    """Tests for get_nodes_by_type, get_node_type, nodes/edges properties."""

    def test_get_nodes_by_type(self):
        """Should return only nodes of the requested type."""
        dag = _make_simple_dag()
        confounders = dag.get_nodes_by_type("confounder")
        assert confounders == ["Age"]

    def test_get_nodes_by_type_empty(self):
        """Should return empty list when no nodes of that type exist."""
        dag = _make_simple_dag()
        assert dag.get_nodes_by_type("mediator") == []

    def test_get_node_type_existing(self):
        """Should return the type of an existing node."""
        dag = _make_simple_dag()
        assert dag.get_node_type("Exposure") == "exposure"

    def test_get_node_type_nonexistent(self):
        """Should return None for nonexistent node."""
        dag = _make_simple_dag()
        assert dag.get_node_type("Ghost") is None

    def test_nodes_property(self):
        """nodes property should list all node names."""
        dag = _make_simple_dag()
        assert set(dag.nodes) == {"Exposure", "Outcome", "Age"}
