"""Integration smoke tests — verify all modules import and basic cross-module flows work."""


# === Module Imports ==========================================================


class TestModuleImports:
    """Verify all core modules import without error."""

    def test_import_stats_calculator(self):
        """stats_calculator should import cleanly."""
        from core.stats_calculator import calculate_odds_ratio  # noqa: F401

    def test_import_power_calculator(self):
        """power_calculator should import cleanly."""
        from core.power_calculator import calculate_sample_size  # noqa: F401

    def test_import_dag_engine(self):
        """dag_engine should import cleanly."""
        from core.dag_engine import DAGEngine  # noqa: F401

    def test_import_confounder_detector(self):
        """confounder_detector should import cleanly."""
        from core.confounder_detector import find_confounders  # noqa: F401

    def test_import_paper_parser(self):
        """paper_parser should import cleanly."""
        from core.paper_parser import find_effect_measures  # noqa: F401


# === End-to-End Flows ========================================================


class TestEndToEndFlows:
    """Cross-module integration tests."""

    def test_dag_to_confounder_detection(self):
        """Build a DAG and detect confounders from it."""
        from core.confounder_detector import find_confounders
        from core.dag_engine import DAGEngine

        dag = DAGEngine()
        dag.add_node("Smoking", "exposure")
        dag.add_node("Cancer", "outcome")
        dag.add_node("Age", "confounder")
        dag.add_edge("Age", "Smoking")
        dag.add_edge("Age", "Cancer")
        dag.add_edge("Smoking", "Cancer")

        confounders = find_confounders(dag.graph, "Smoking", "Cancer")
        assert "Age" in confounders

    def test_stats_to_nnt_chain(self):
        """Calculate RD then derive NNT from it."""
        from core.stats_calculator import calculate_nnt, calculate_risk_difference

        rd = calculate_risk_difference(40, 60, 10, 90)
        nnt = calculate_nnt(rd["value"])
        assert nnt["value"] is not None
        assert nnt["value"] > 0

    def test_power_roundtrip(self):
        """Sample size → power should roundtrip to ~0.80."""
        from core.power_calculator import calculate_power, calculate_sample_size

        n = calculate_sample_size(0.5)
        power = calculate_power(n, 0.5)
        assert abs(power - 0.80) < 0.02

    def test_dag_serialization_roundtrip(self):
        """DAG to_dict → from_dict should preserve structure."""
        from core.dag_engine import DAGEngine

        dag = DAGEngine()
        dag.add_node("E", "exposure")
        dag.add_node("O", "outcome")
        dag.add_edge("E", "O")

        data = dag.to_dict()
        dag2 = DAGEngine()
        dag2.from_dict(data)

        assert set(dag.nodes) == set(dag2.nodes)
        assert set(dag.edges) == set(dag2.edges)

    def test_parser_all_functions_on_one_block(self):
        """Run all parser extract functions on a single text block."""
        from core.paper_parser import (
            find_beta_coefficients,
            find_confidence_intervals,
            find_effect_measures,
            find_mean_differences,
            find_p_values,
            find_sample_sizes,
            find_standard_deviations,
            find_weighted_statistics,
        )

        text = (
            "In our study of n = 500 participants, the OR = 2.5 (95% CI: 1.2-3.8), "
            "p < 0.001. Mean difference = 3.2. beta = 0.45. SD = 1.5. "
            "Weighted prevalence: 25.3%."
        )

        effects = find_effect_measures(text)
        assert any(r["type"] == "OR" and r["value"] == 2.5 for r in effects)

        cis = find_confidence_intervals(text)
        assert any(r["lower"] == 1.2 and r["upper"] == 3.8 for r in cis)

        pvals = find_p_values(text)
        assert any(r["value"] == 0.001 for r in pvals)

        assert any(r["value"] == 500 for r in find_sample_sizes(text))
        assert any(r["value"] == 3.2 for r in find_mean_differences(text))
        assert any(r["value"] == 0.45 for r in find_beta_coefficients(text))
        assert any(r["value"] == 1.5 and r["type"] == "SD" for r in find_standard_deviations(text))
        assert any(r["value"] == 25.3 for r in find_weighted_statistics(text))
