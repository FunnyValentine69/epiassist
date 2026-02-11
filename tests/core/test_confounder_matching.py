"""Tests for variable name normalization and column-to-DAG matching."""

import pytest

from core.confounder_detector import (
    compare_adjustment_sets,
    match_columns_to_dag_nodes,
    normalize_variable_name,
)


class TestNormalizeVariableName:
    """Tests for normalize_variable_name."""

    def test_lowercase(self) -> None:
        assert normalize_variable_name("Age") == "age"

    def test_underscore_to_space(self) -> None:
        assert normalize_variable_name("hearing_loss") == "hearing loss"

    def test_hyphen_to_space(self) -> None:
        assert normalize_variable_name("hearing-loss") == "hearing loss"

    def test_mixed_case_underscore(self) -> None:
        assert normalize_variable_name("Hearing_Loss") == "hearing loss"

    def test_strip_whitespace(self) -> None:
        assert normalize_variable_name("  age  ") == "age"

    def test_already_normalized(self) -> None:
        assert normalize_variable_name("age") == "age"

    def test_empty_string(self) -> None:
        assert normalize_variable_name("") == ""

    def test_multi_word(self) -> None:
        assert normalize_variable_name("Body Mass Index") == "body mass index"


class TestMatchColumnsToDagNodes:
    """Tests for match_columns_to_dag_nodes."""

    def test_exact_case_insensitive(self) -> None:
        result = match_columns_to_dag_nodes(["age", "sex"], ["Age", "Sex"])
        assert result == {"Age": "age", "Sex": "sex"}

    def test_underscore_to_space(self) -> None:
        result = match_columns_to_dag_nodes(
            ["hearing_loss", "age"], ["Hearing Loss"]
        )
        assert result == {"Hearing Loss": "hearing_loss"}

    def test_no_match_returns_empty(self) -> None:
        result = match_columns_to_dag_nodes(["weight", "height"], ["Age"])
        assert result == {}

    def test_partial_match(self) -> None:
        result = match_columns_to_dag_nodes(
            ["age", "income", "bmi"], ["Age", "Smoking"]
        )
        assert result == {"Age": "age"}
        assert "Smoking" not in result

    def test_empty_columns(self) -> None:
        result = match_columns_to_dag_nodes([], ["Age"])
        assert result == {}

    def test_empty_dag_nodes(self) -> None:
        result = match_columns_to_dag_nodes(["age"], [])
        assert result == {}

    def test_preserves_original_column_names(self) -> None:
        result = match_columns_to_dag_nodes(
            ["Hearing_Loss", "AGE"], ["hearing loss", "age"]
        )
        assert result == {"hearing loss": "Hearing_Loss", "age": "AGE"}


class TestCompareAdjustmentSets:
    """Tests for compare_adjustment_sets."""

    def test_full_overlap(self) -> None:
        result = compare_adjustment_sets(["Age", "Sex"], ["age", "sex"])
        assert sorted(result["overlap"]) == ["Age", "Sex"]
        assert result["dag_only"] == []
        assert result["paper_only"] == []

    def test_no_overlap(self) -> None:
        result = compare_adjustment_sets(["Age"], ["BMI"])
        assert result["overlap"] == []
        assert result["dag_only"] == ["Age"]
        assert result["paper_only"] == ["BMI"]

    def test_partial_overlap(self) -> None:
        result = compare_adjustment_sets(
            ["Age", "Smoking"], ["age", "BMI"]
        )
        assert result["overlap"] == ["Age"]
        assert result["dag_only"] == ["Smoking"]
        assert result["paper_only"] == ["BMI"]

    def test_empty_dag_set(self) -> None:
        result = compare_adjustment_sets([], ["Age"])
        assert result["overlap"] == []
        assert result["dag_only"] == []
        assert result["paper_only"] == ["Age"]

    def test_empty_paper_set(self) -> None:
        result = compare_adjustment_sets(["Age"], [])
        assert result["overlap"] == []
        assert result["dag_only"] == ["Age"]
        assert result["paper_only"] == []

    def test_normalized_matching(self) -> None:
        result = compare_adjustment_sets(
            ["Hearing Loss"], ["hearing_loss"]
        )
        assert result["overlap"] == ["Hearing Loss"]
        assert result["dag_only"] == []
        assert result["paper_only"] == []

    def test_preserves_original_names(self) -> None:
        result = compare_adjustment_sets(
            ["Age"], ["AGE"]
        )
        # overlap uses DAG's original name
        assert result["overlap"] == ["Age"]
