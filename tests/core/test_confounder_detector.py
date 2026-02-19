"""Tests for core/confounder_detector.py — graph analysis functions.

The existing test_confounder_matching.py covers normalize/match/compare.
This file covers find_confounders, find_backdoor_paths,
suggest_adjustment_set, get_direct_causes, and get_direct_effects.
"""

import networkx as nx

from core.confounder_detector import (
    find_backdoor_paths,
    find_confounders,
    get_direct_causes,
    get_direct_effects,
    suggest_adjustment_set,
)


def _make_confounding_dag():
    """Helper: Age→Exposure, Age→Outcome, Exposure→Outcome."""
    g = nx.DiGraph()
    g.add_node("Exposure", type="exposure")
    g.add_node("Outcome", type="outcome")
    g.add_node("Age", type="confounder")
    g.add_edge("Age", "Exposure")
    g.add_edge("Age", "Outcome")
    g.add_edge("Exposure", "Outcome")
    return g


def _make_multi_confounder_dag():
    """Helper: Age and SES both confound Exposure→Outcome."""
    g = _make_confounding_dag()
    g.add_node("SES", type="confounder")
    g.add_edge("SES", "Exposure")
    g.add_edge("SES", "Outcome")
    return g


def _make_mediator_dag():
    """Helper: Exposure→Mediator→Outcome (no confounding)."""
    g = nx.DiGraph()
    g.add_node("Exposure", type="exposure")
    g.add_node("Outcome", type="outcome")
    g.add_node("Mediator", type="mediator")
    g.add_edge("Exposure", "Mediator")
    g.add_edge("Mediator", "Outcome")
    g.add_edge("Exposure", "Outcome")
    return g


# === Find Confounders =======================================================


class TestFindConfounders:
    """Tests for find_confounders."""

    def test_single_confounder(self):
        """Should detect Age as a confounder."""
        g = _make_confounding_dag()
        result = find_confounders(g, "Exposure", "Outcome")
        assert "Age" in result

    def test_multiple_confounders(self):
        """Should detect both Age and SES."""
        g = _make_multi_confounder_dag()
        result = find_confounders(g, "Exposure", "Outcome")
        assert set(result) == {"Age", "SES"}

    def test_mediator_not_included(self):
        """A mediator (Exposure→M→Outcome) should not be listed as confounder."""
        g = _make_mediator_dag()
        result = find_confounders(g, "Exposure", "Outcome")
        assert "Mediator" not in result

    def test_direct_only_no_confounders(self):
        """A DAG with only Exposure→Outcome has no confounders."""
        g = nx.DiGraph()
        g.add_edge("Exposure", "Outcome")
        result = find_confounders(g, "Exposure", "Outcome")
        assert result == []

    def test_missing_exposure(self):
        """Missing exposure node should return empty list."""
        g = _make_confounding_dag()
        assert find_confounders(g, "NonExistent", "Outcome") == []

    def test_missing_outcome(self):
        """Missing outcome node should return empty list."""
        g = _make_confounding_dag()
        assert find_confounders(g, "Exposure", "NonExistent") == []

    def test_exposure_outcome_not_in_result(self):
        """Exposure and Outcome themselves should never appear as confounders."""
        g = _make_confounding_dag()
        result = find_confounders(g, "Exposure", "Outcome")
        assert "Exposure" not in result
        assert "Outcome" not in result


# === Find Backdoor Paths ====================================================


class TestFindBackdoorPaths:
    """Tests for find_backdoor_paths."""

    def test_single_backdoor_path(self):
        """Age→Exposure and Age→Outcome should create a backdoor path."""
        g = _make_confounding_dag()
        paths = find_backdoor_paths(g, "Exposure", "Outcome")
        assert len(paths) >= 1
        # At least one path should go through Age
        age_in_path = any("Age" in p for p in paths)
        assert age_in_path

    def test_multiple_confounders_multiple_paths(self):
        """Multiple confounders should produce multiple backdoor paths."""
        g = _make_multi_confounder_dag()
        paths = find_backdoor_paths(g, "Exposure", "Outcome")
        assert len(paths) >= 2

    def test_no_backdoor_direct_only(self):
        """A direct-only DAG should have no backdoor paths."""
        g = nx.DiGraph()
        g.add_edge("Exposure", "Outcome")
        paths = find_backdoor_paths(g, "Exposure", "Outcome")
        assert paths == []

    def test_missing_node_returns_empty(self):
        """Missing node should return empty list."""
        g = _make_confounding_dag()
        assert find_backdoor_paths(g, "Ghost", "Outcome") == []

    def test_backdoor_contains_confounder(self):
        """Each backdoor path should contain at least one confounder."""
        g = _make_confounding_dag()
        paths = find_backdoor_paths(g, "Exposure", "Outcome")
        confounders = find_confounders(g, "Exposure", "Outcome")
        for path in paths:
            assert any(c in path for c in confounders)


# === Suggest Adjustment Set =================================================


class TestSuggestAdjustmentSet:
    """Tests for suggest_adjustment_set."""

    def test_returns_confounders(self):
        """Adjustment set should include all confounders."""
        g = _make_confounding_dag()
        adj_set = suggest_adjustment_set(g, "Exposure", "Outcome")
        assert "Age" in adj_set

    def test_empty_for_direct_only(self):
        """Direct-only DAG should have empty adjustment set."""
        g = nx.DiGraph()
        g.add_edge("Exposure", "Outcome")
        assert suggest_adjustment_set(g, "Exposure", "Outcome") == []

    def test_multiple_confounders_included(self):
        """All confounders should be in the adjustment set."""
        g = _make_multi_confounder_dag()
        adj_set = suggest_adjustment_set(g, "Exposure", "Outcome")
        assert set(adj_set) == {"Age", "SES"}


# === Get Direct Causes ======================================================


class TestGetDirectCauses:
    """Tests for get_direct_causes (parents)."""

    def test_parents_returned(self):
        """Exposure should have Age as direct cause."""
        g = _make_confounding_dag()
        parents = get_direct_causes(g, "Exposure")
        assert "Age" in parents

    def test_no_parents(self):
        """A root node (Age) should have no direct causes."""
        g = _make_confounding_dag()
        assert get_direct_causes(g, "Age") == []

    def test_nonexistent_node(self):
        """Nonexistent node should return empty list."""
        g = _make_confounding_dag()
        assert get_direct_causes(g, "Ghost") == []


# === Get Direct Effects =====================================================


class TestGetDirectEffects:
    """Tests for get_direct_effects (children)."""

    def test_children_returned(self):
        """Age should have Exposure and Outcome as direct effects."""
        g = _make_confounding_dag()
        children = get_direct_effects(g, "Age")
        assert set(children) == {"Exposure", "Outcome"}

    def test_no_children(self):
        """Outcome (leaf) should have no direct effects."""
        g = _make_confounding_dag()
        assert get_direct_effects(g, "Outcome") == []

    def test_nonexistent_node(self):
        """Nonexistent node should return empty list."""
        g = _make_confounding_dag()
        assert get_direct_effects(g, "Ghost") == []
