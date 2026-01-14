"""Confounder detection for causal DAGs.

This module provides functions to identify confounders and suggest
adjustment sets for causal inference.
"""

import networkx as nx


def find_confounders(
    dag: nx.DiGraph, exposure: str, outcome: str
) -> list[str]:
    """Find confounders between exposure and outcome.

    A confounder is a node that has a directed path to both the exposure
    and the outcome (common cause). This is a simplified implementation
    that finds nodes with outgoing edges to both exposure and outcome.

    Args:
        dag: NetworkX DiGraph representing the causal DAG.
        exposure: Name of the exposure variable.
        outcome: Name of the outcome variable.

    Returns:
        List of confounder node names.
    """
    if exposure not in dag or outcome not in dag:
        return []

    confounders = []

    for node in dag.nodes():
        # Skip exposure and outcome themselves
        if node in (exposure, outcome):
            continue

        # Check if this node has a path to both exposure and outcome
        has_path_to_exposure = nx.has_path(dag, node, exposure)
        has_path_to_outcome = nx.has_path(dag, node, outcome)

        if has_path_to_exposure and has_path_to_outcome:
            confounders.append(node)

    return confounders


def find_backdoor_paths(
    dag: nx.DiGraph, exposure: str, outcome: str
) -> list[list[str]]:
    """Find backdoor paths between exposure and outcome.

    A backdoor path is any path from exposure to outcome that has an
    arrow pointing into the exposure. This simplified implementation
    finds paths through common ancestors.

    Args:
        dag: NetworkX DiGraph representing the causal DAG.
        exposure: Name of the exposure variable.
        outcome: Name of the outcome variable.

    Returns:
        List of backdoor paths, where each path is a list of node names.
    """
    if exposure not in dag or outcome not in dag:
        return []

    backdoor_paths = []

    # Find all ancestors of the exposure (nodes that point to exposure)
    ancestors_of_exposure = set(nx.ancestors(dag, exposure))

    # For each ancestor that also connects to outcome, trace the path
    for ancestor in ancestors_of_exposure:
        # Check if this ancestor has a path to outcome
        if nx.has_path(dag, ancestor, outcome):
            # Get paths from ancestor to both exposure and outcome
            try:
                paths_to_exposure = list(
                    nx.all_simple_paths(dag, ancestor, exposure)
                )
                paths_to_outcome = list(
                    nx.all_simple_paths(dag, ancestor, outcome)
                )

                # Combine to form backdoor paths
                for path_e in paths_to_exposure:
                    for path_o in paths_to_outcome:
                        # Avoid paths that go through the direct causal path
                        if not _path_goes_through_direct_effect(
                            path_o, exposure, outcome
                        ):
                            # Create backdoor path: reverse path to exposure + path to outcome
                            backdoor = list(reversed(path_e[:-1])) + path_o
                            if backdoor not in backdoor_paths:
                                backdoor_paths.append(backdoor)
            except nx.NetworkXError:
                continue

    return backdoor_paths


def _path_goes_through_direct_effect(
    path: list[str], exposure: str, outcome: str
) -> bool:
    """Check if a path goes through the exposure->outcome edge."""
    for i in range(len(path) - 1):
        if path[i] == exposure and path[i + 1] == outcome:
            return True
    return False


def suggest_adjustment_set(
    dag: nx.DiGraph, exposure: str, outcome: str
) -> list[str]:
    """Suggest a minimal sufficient adjustment set.

    For the simplified common-cause confounding model, the adjustment
    set is simply all identified confounders.

    Args:
        dag: NetworkX DiGraph representing the causal DAG.
        exposure: Name of the exposure variable.
        outcome: Name of the outcome variable.

    Returns:
        List of variable names to adjust for.
    """
    return find_confounders(dag, exposure, outcome)


def get_direct_causes(dag: nx.DiGraph, node: str) -> list[str]:
    """Get all direct causes (parents) of a node.

    Args:
        dag: NetworkX DiGraph representing the causal DAG.
        node: Name of the node.

    Returns:
        List of parent node names.
    """
    if node not in dag:
        return []
    return list(dag.predecessors(node))


def get_direct_effects(dag: nx.DiGraph, node: str) -> list[str]:
    """Get all direct effects (children) of a node.

    Args:
        dag: NetworkX DiGraph representing the causal DAG.
        node: Name of the node.

    Returns:
        List of child node names.
    """
    if node not in dag:
        return []
    return list(dag.successors(node))
