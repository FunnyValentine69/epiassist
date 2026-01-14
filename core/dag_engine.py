"""DAG Engine for creating and visualizing causal diagrams.

This module provides the DAGEngine class for building Directed Acyclic Graphs
representing causal relationships between variables.
"""

from typing import Any

import graphviz
import networkx as nx

from utils.constants import NODE_COLORS


class DAGEngine:
    """Engine for creating and manipulating Directed Acyclic Graphs.

    Attributes:
        graph: NetworkX DiGraph storing nodes and edges.
    """

    def __init__(self) -> None:
        """Initialize an empty DAG."""
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_node(self, name: str, node_type: str) -> None:
        """Add a node to the DAG.

        Args:
            name: The name/label of the variable.
            node_type: One of 'exposure', 'outcome', 'confounder', 'mediator'.

        Raises:
            ValueError: If node_type is invalid.
        """
        valid_types = {"exposure", "outcome", "confounder", "mediator"}
        if node_type not in valid_types:
            raise ValueError(f"node_type must be one of {valid_types}")

        self.graph.add_node(name, label=name, type=node_type)

    def add_edge(self, source: str, target: str) -> None:
        """Add a directed edge from source to target.

        Args:
            source: The name of the source node.
            target: The name of the target node.

        Raises:
            ValueError: If source or target node doesn't exist.
        """
        if source not in self.graph:
            raise ValueError(f"Source node '{source}' does not exist")
        if target not in self.graph:
            raise ValueError(f"Target node '{target}' does not exist")

        self.graph.add_edge(source, target)

    def remove_node(self, name: str) -> None:
        """Remove a node and all its edges from the DAG.

        Args:
            name: The name of the node to remove.
        """
        if name in self.graph:
            self.graph.remove_node(name)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the DAG.

        Args:
            source: The name of the source node.
            target: The name of the target node.
        """
        if self.graph.has_edge(source, target):
            self.graph.remove_edge(source, target)

    def get_all_paths(self, source: str, target: str) -> list[list[str]]:
        """Find all paths from source to target.

        Args:
            source: The starting node.
            target: The ending node.

        Returns:
            A list of paths, where each path is a list of node names.
        """
        if source not in self.graph or target not in self.graph:
            return []

        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXError:
            return []

    def render_graphviz(self) -> graphviz.Digraph:
        """Render the DAG as a Graphviz diagram.

        Returns:
            A graphviz.Digraph object for visualization.
        """
        dot = graphviz.Digraph(comment="Causal DAG")
        dot.attr(rankdir="LR")  # Left to right layout

        # Add nodes with colors based on type
        for node, attrs in self.graph.nodes(data=True):
            node_type = attrs.get("type", "confounder")
            color = NODE_COLORS.get(node_type, "#CCCCCC")
            dot.node(
                node,
                label=node,
                style="filled",
                fillcolor=color,
                fontcolor="black",
            )

        # Add edges
        for source, target in self.graph.edges():
            dot.edge(source, target)

        return dot

    def to_dict(self) -> dict[str, Any]:
        """Serialize the DAG to a dictionary.

        Returns:
            Dictionary representation of the DAG.
        """
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({
                "name": node,
                "type": attrs.get("type", "confounder"),
            })

        edges = []
        for source, target in self.graph.edges():
            edges.append({"source": source, "target": target})

        return {"nodes": nodes, "edges": edges}

    def from_dict(self, data: dict[str, Any]) -> None:
        """Load a DAG from a dictionary.

        Args:
            data: Dictionary with 'nodes' and 'edges' keys.
        """
        self.graph.clear()

        for node_data in data.get("nodes", []):
            self.add_node(node_data["name"], node_data["type"])

        for edge_data in data.get("edges", []):
            self.add_edge(edge_data["source"], edge_data["target"])

    def get_nodes_by_type(self, node_type: str) -> list[str]:
        """Get all nodes of a specific type.

        Args:
            node_type: The type to filter by.

        Returns:
            List of node names matching the type.
        """
        return [
            node
            for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == node_type
        ]

    def get_node_type(self, name: str) -> str | None:
        """Get the type of a node.

        Args:
            name: The node name.

        Returns:
            The node type, or None if node doesn't exist.
        """
        if name in self.graph:
            return self.graph.nodes[name].get("type")
        return None

    @property
    def nodes(self) -> list[str]:
        """Get list of all node names."""
        return list(self.graph.nodes())

    @property
    def edges(self) -> list[tuple[str, str]]:
        """Get list of all edges as (source, target) tuples."""
        return list(self.graph.edges())
