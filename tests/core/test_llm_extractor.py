"""Tests for core.llm_extractor module (provider-agnostic version)."""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_extractor import (
    _dedup_key,
    extract_with_llm,
    is_llm_available,
    merge_results,
)
from core.llm_providers._parse import _safe_float, _safe_int


# --- _safe_float / _safe_int (imported from _parse) ---


class TestSafeConversions:
    def test_safe_float_valid(self):
        assert _safe_float("2.5") == 2.5
        assert _safe_float("0.001") == 0.001

    def test_safe_float_empty(self):
        assert _safe_float("") is None
        assert _safe_float(None) is None

    def test_safe_float_invalid(self):
        assert _safe_float("abc") is None

    def test_safe_int_valid(self):
        assert _safe_int("450") == 450
        assert _safe_int("100.0") == 100

    def test_safe_int_empty(self):
        assert _safe_int("") is None

    def test_safe_int_invalid(self):
        assert _safe_int("abc") is None


# --- is_llm_available ---


class TestIsLlmAvailable:
    @patch("core.llm_extractor.detect_provider", return_value="gemini")
    def test_available_gemini(self, mock_detect):
        available, provider = is_llm_available()
        assert available is True
        assert provider == "gemini"

    @patch("core.llm_extractor.detect_provider", return_value="ollama")
    def test_available_ollama(self, mock_detect):
        available, provider = is_llm_available()
        assert available is True
        assert provider == "ollama"

    @patch("core.llm_extractor.detect_provider", return_value=None)
    def test_unavailable(self, mock_detect):
        available, provider = is_llm_available()
        assert available is False
        assert provider is None


# --- extract_with_llm ---


class TestExtractWithLlm:
    @patch("core.llm_extractor.detect_provider", return_value=None)
    def test_no_provider_returns_empty(self, mock_detect):
        result = extract_with_llm("some text", page=1)
        assert all(len(v) == 0 for v in result.values())

    @patch("core.llm_extractor.get_provider_functions")
    @patch("core.llm_extractor.detect_provider", return_value="gemini")
    def test_delegates_to_provider(self, mock_detect, mock_get_funcs):
        mock_extract = MagicMock(return_value={
            "effect_measures": [{"type": "OR", "value": 2.5, "page": 1}],
            "confidence_intervals": [], "p_values": [], "sample_sizes": [],
            "beta_coefficients": [], "mean_differences": [],
            "standard_deviations": [], "weighted_statistics": [],
        })
        mock_get_funcs.return_value = {"extract_stats": mock_extract, "chat": MagicMock()}

        result = extract_with_llm("some text", page=1)
        assert len(result["effect_measures"]) == 1
        mock_extract.assert_called_once_with("some text", 1)

    @patch("core.llm_extractor.get_provider_functions")
    @patch("core.llm_extractor.detect_provider", return_value="ollama")
    def test_exception_returns_empty(self, mock_detect, mock_get_funcs):
        mock_get_funcs.return_value = {
            "extract_stats": MagicMock(side_effect=RuntimeError("boom")),
            "chat": MagicMock(),
        }
        result = extract_with_llm("some text", page=1)
        assert all(len(v) == 0 for v in result.values())


# --- merge_results ---


class TestMergeResults:
    def test_tags_source(self):
        regex = {"effect_measures": [{"type": "OR", "value": 2.5, "ci_lower": None, "ci_upper": None, "page": 1, "context": "x"}]}
        llm = {"effect_measures": [{"type": "HR", "value": 1.8, "ci_lower": None, "ci_upper": None, "page": 2, "context": "y"}]}
        for cat in ["confidence_intervals", "p_values", "sample_sizes", "beta_coefficients", "mean_differences", "standard_deviations", "weighted_statistics"]:
            regex[cat] = []
            llm[cat] = []

        merged = merge_results(regex, llm)
        assert merged["effect_measures"][0]["source"] == "regex"
        assert merged["effect_measures"][1]["source"] == "llm"

    def test_deduplication_float_equality(self):
        regex = {
            "effect_measures": [{"type": "OR", "value": 2.5, "ci_lower": 1.2, "ci_upper": 3.8, "page": 1, "context": "x"}],
            "confidence_intervals": [], "p_values": [], "sample_sizes": [],
            "beta_coefficients": [], "mean_differences": [], "standard_deviations": [], "weighted_statistics": [],
        }
        llm = {
            "effect_measures": [{"type": "OR", "value": 2.50, "ci_lower": 1.20, "ci_upper": 3.80, "page": 1, "context": "y"}],
            "confidence_intervals": [], "p_values": [], "sample_sizes": [],
            "beta_coefficients": [], "mean_differences": [], "standard_deviations": [], "weighted_statistics": [],
        }
        merged = merge_results(regex, llm)
        assert len(merged["effect_measures"]) == 1
        assert merged["effect_measures"][0]["source"] == "regex"

    def test_additive_non_duplicate(self):
        regex = {
            "p_values": [{"value": 0.001, "operator": "<", "page": 1, "context": "x"}],
            "effect_measures": [], "confidence_intervals": [], "sample_sizes": [],
            "beta_coefficients": [], "mean_differences": [], "standard_deviations": [], "weighted_statistics": [],
        }
        llm = {
            "p_values": [{"value": 0.03, "operator": "=", "page": 2, "context": "y"}],
            "effect_measures": [], "confidence_intervals": [], "sample_sizes": [],
            "beta_coefficients": [], "mean_differences": [], "standard_deviations": [], "weighted_statistics": [],
        }
        merged = merge_results(regex, llm)
        assert len(merged["p_values"]) == 2

    def test_empty_inputs(self):
        regex = {cat: [] for cat in ["effect_measures", "confidence_intervals", "p_values", "sample_sizes", "beta_coefficients", "mean_differences", "standard_deviations", "weighted_statistics"]}
        llm = {cat: [] for cat in regex}
        merged = merge_results(regex, llm)
        assert all(len(v) == 0 for v in merged.values())


# --- _dedup_key ---


class TestDedupKey:
    def test_effect_measures_key(self):
        item = {"type": "OR", "value": 2.5, "ci_lower": 1.2, "ci_upper": 3.8, "page": 1}
        key = _dedup_key(item, "effect_measures")
        assert key == ("OR", 2.5, 1.2, 3.8, 1)

    def test_sample_sizes_key(self):
        item = {"value": 450, "page": 2}
        key = _dedup_key(item, "sample_sizes")
        assert key == (450, 2)

    def test_none_ci_fields(self):
        item = {"type": "OR", "value": 2.5, "ci_lower": None, "ci_upper": None, "page": 1}
        key = _dedup_key(item, "effect_measures")
        assert key == ("OR", 2.5, None, None, 1)
