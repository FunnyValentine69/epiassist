"""Tests for core.llm_extractor module."""

from unittest.mock import MagicMock, patch

import pytest

from core.llm_extractor import (
    _dedup_key,
    _extraction_to_dict,
    _safe_float,
    _safe_int,
    extract_with_llm,
    is_llm_available,
    merge_results,
)


# --- _safe_float / _safe_int ---


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


# --- _extraction_to_dict ---


class TestExtractionToDict:
    """Test conversion of LangExtract Extraction objects to our dict schema."""

    def _make_extraction(self, cls: str, text: str, attrs: dict) -> MagicMock:
        ext = MagicMock()
        ext.extraction_class = cls
        ext.extraction_text = text
        ext.attributes = attrs
        return ext

    def test_effect_measure(self):
        ext = self._make_extraction(
            "effect_measure",
            "OR = 2.45",
            {"type": "OR", "value": "2.45", "ci_lower": "1.12", "ci_upper": "5.34"},
        )
        result = _extraction_to_dict(ext, page=3)
        assert result is not None
        cat, d = result
        assert cat == "effect_measures"
        assert d["value"] == 2.45
        assert d["ci_lower"] == 1.12
        assert d["ci_upper"] == 5.34
        assert d["page"] == 3
        assert d["type"] == "OR"

    def test_confidence_interval(self):
        ext = self._make_extraction(
            "confidence_interval",
            "95% CI: 1.2-3.8",
            {"lower": "1.2", "upper": "3.8", "level": "95"},
        )
        result = _extraction_to_dict(ext, page=1)
        assert result is not None
        cat, d = result
        assert cat == "confidence_intervals"
        assert d["lower"] == 1.2
        assert d["upper"] == 3.8
        assert d["level"] == 95

    def test_p_value(self):
        ext = self._make_extraction(
            "p_value", "p<0.001", {"value": "0.001", "operator": "<"}
        )
        result = _extraction_to_dict(ext, page=2)
        assert result is not None
        cat, d = result
        assert cat == "p_values"
        assert d["value"] == 0.001
        assert d["operator"] == "<"

    def test_sample_size(self):
        ext = self._make_extraction(
            "sample_size", "n=450", {"value": "450"}
        )
        result = _extraction_to_dict(ext, page=1)
        assert result is not None
        cat, d = result
        assert cat == "sample_sizes"
        assert d["value"] == 450

    def test_beta_coefficient(self):
        ext = self._make_extraction(
            "beta_coefficient",
            "β = 0.34",
            {"value": "0.34", "ci_lower": "0.12", "ci_upper": "0.56", "se": "0.11"},
        )
        result = _extraction_to_dict(ext, page=4)
        assert result is not None
        cat, d = result
        assert cat == "beta_coefficients"
        assert d["value"] == 0.34
        assert d["se"] == 0.11

    def test_mean_difference(self):
        ext = self._make_extraction(
            "mean_difference",
            "MD = 3.2",
            {"value": "3.2", "ci_lower": "1.0", "ci_upper": "5.4"},
        )
        result = _extraction_to_dict(ext, page=1)
        assert result is not None
        cat, d = result
        assert cat == "mean_differences"
        assert d["value"] == 3.2

    def test_standard_deviation(self):
        ext = self._make_extraction(
            "standard_deviation",
            "SD 1.5",
            {"value": "1.5", "mean": "3.2", "sd_type": "SD"},
        )
        result = _extraction_to_dict(ext, page=1)
        assert result is not None
        cat, d = result
        assert cat == "standard_deviations"
        assert d["value"] == 1.5
        assert d["mean"] == 3.2
        assert d["type"] == "SD"

    def test_weighted_statistic(self):
        ext = self._make_extraction(
            "weighted_statistic",
            "weighted prevalence: 25.3%",
            {"stat_type": "prevalence", "value": "25.3", "weight_method": "IPW"},
        )
        result = _extraction_to_dict(ext, page=2)
        assert result is not None
        cat, d = result
        assert cat == "weighted_statistics"
        assert d["value"] == 25.3

    def test_invalid_value_returns_none(self):
        ext = self._make_extraction(
            "effect_measure",
            "OR = ???",
            {"type": "OR", "value": "not_a_number", "ci_lower": "", "ci_upper": ""},
        )
        assert _extraction_to_dict(ext, page=1) is None

    def test_unknown_class_returns_none(self):
        ext = self._make_extraction(
            "unknown_class", "something", {"value": "1.0"}
        )
        assert _extraction_to_dict(ext, page=1) is None

    def test_ci_missing_lower_returns_none(self):
        ext = self._make_extraction(
            "confidence_interval",
            "CI: ?-3.8",
            {"lower": "", "upper": "3.8", "level": "95"},
        )
        assert _extraction_to_dict(ext, page=1) is None


# --- merge_results ---


class TestMergeResults:
    def test_tags_source(self):
        regex = {"effect_measures": [{"type": "OR", "value": 2.5, "ci_lower": None, "ci_upper": None, "page": 1, "context": "x"}]}
        llm = {"effect_measures": [{"type": "HR", "value": 1.8, "ci_lower": None, "ci_upper": None, "page": 2, "context": "y"}]}
        # Fill missing categories
        for cat in ["confidence_intervals", "p_values", "sample_sizes", "beta_coefficients", "mean_differences", "standard_deviations", "weighted_statistics"]:
            regex[cat] = []
            llm[cat] = []

        merged = merge_results(regex, llm)
        assert merged["effect_measures"][0]["source"] == "regex"
        assert merged["effect_measures"][1]["source"] == "llm"

    def test_deduplication_float_equality(self):
        """2.5 and 2.50 should be treated as duplicates."""
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
        # Duplicate should be dropped — only the regex one remains
        assert len(merged["effect_measures"]) == 1
        assert merged["effect_measures"][0]["source"] == "regex"

    def test_additive_non_duplicate(self):
        """Different values should both appear."""
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


# --- is_llm_available ---


class TestIsLlmAvailable:
    @patch("core.llm_extractor.requests.get")
    def test_available(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        # langextract may or may not be importable in test env
        # We test the requests portion
        with patch.dict("sys.modules", {"langextract": MagicMock()}):
            assert is_llm_available() is True

    def test_import_error(self):
        with patch.dict("sys.modules", {"langextract": None}):
            # When module is None in sys.modules, import raises ImportError
            import sys
            saved = sys.modules.get("langextract")
            sys.modules["langextract"] = None
            try:
                # Force re-evaluation by calling the function
                # The function does `import langextract` which will raise ImportError
                # because sys.modules["langextract"] is None
                result = is_llm_available()
                assert result is False
            finally:
                if saved is not None:
                    sys.modules["langextract"] = saved
                else:
                    sys.modules.pop("langextract", None)

    @patch("core.llm_extractor.requests.get", side_effect=ConnectionError("Connection refused"))
    def test_unreachable_server(self, mock_get):
        with patch.dict("sys.modules", {"langextract": MagicMock()}):
            assert is_llm_available() is False


# --- extract_with_llm ---


class TestExtractWithLlm:
    def test_exception_returns_empty(self):
        """Any exception during extraction should return empty results."""
        with patch.dict("sys.modules", {"langextract": MagicMock()}) as mock_modules:
            mock_lx = mock_modules["langextract"]
            mock_lx.extract.side_effect = RuntimeError("Ollama down")
            result = extract_with_llm("some text", page=1)
            assert all(len(v) == 0 for v in result.values())


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
