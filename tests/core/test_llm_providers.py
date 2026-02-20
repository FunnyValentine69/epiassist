"""Tests for core.llm_providers package."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from core.llm_providers._parse import (
    CATEGORIES,
    _empty_results,
    _safe_float,
    _safe_int,
    parse_extraction_response,
)


# --- _safe_float / _safe_int ---


class TestSafeFloat:
    def test_valid_string(self):
        assert _safe_float("2.5") == 2.5
        assert _safe_float("0.001") == 0.001

    def test_valid_numeric(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(42) == 42.0

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid(self):
        assert _safe_float("abc") is None

    def test_whitespace(self):
        assert _safe_float("  2.5  ") == 2.5
        assert _safe_float("  ") is None


class TestSafeInt:
    def test_valid_string(self):
        assert _safe_int("450") == 450
        assert _safe_int("100.0") == 100

    def test_valid_numeric(self):
        assert _safe_int(42) == 42

    def test_empty_string(self):
        assert _safe_int("") is None

    def test_none(self):
        assert _safe_int(None) is None

    def test_invalid(self):
        assert _safe_int("abc") is None

    def test_bool_not_int(self):
        # bool is subclass of int, but we treat it separately
        assert _safe_int(True) is not None  # True -> 1 via float conversion


# --- parse_extraction_response ---


class TestParseExtractionResponse:
    def test_valid_full_json(self):
        data = {
            "effect_measures": [
                {"type": "OR", "value": 2.45, "ci_lower": 1.12, "ci_upper": 5.34, "context": "OR=2.45"}
            ],
            "confidence_intervals": [],
            "p_values": [{"value": 0.024, "operator": "=", "context": "p=0.024"}],
            "sample_sizes": [{"value": 450}],
            "beta_coefficients": [],
            "mean_differences": [],
            "standard_deviations": [],
            "weighted_statistics": [],
        }
        result = parse_extraction_response(json.dumps(data), page=3)
        assert len(result["effect_measures"]) == 1
        assert result["effect_measures"][0]["value"] == 2.45
        assert result["effect_measures"][0]["page"] == 3
        assert len(result["p_values"]) == 1
        assert result["sample_sizes"][0]["value"] == 450

    def test_none_input(self):
        result = parse_extraction_response(None, page=1)
        assert all(len(v) == 0 for v in result.values())
        assert len(result) == 8

    def test_empty_string_input(self):
        result = parse_extraction_response("", page=1)
        assert all(len(v) == 0 for v in result.values())

    def test_invalid_json(self):
        result = parse_extraction_response("not json at all", page=1)
        assert all(len(v) == 0 for v in result.values())

    def test_non_dict_json(self):
        result = parse_extraction_response("[1, 2, 3]", page=1)
        assert all(len(v) == 0 for v in result.values())

    def test_partial_data(self):
        data = {
            "effect_measures": [
                {"type": "HR", "value": 1.78, "ci_lower": 1.23, "ci_upper": 2.56, "context": "HR"}
            ],
            # Missing other keys — should still work
        }
        result = parse_extraction_response(json.dumps(data), page=1)
        assert len(result["effect_measures"]) == 1
        assert result["effect_measures"][0]["type"] == "HR"
        # Other categories should be empty
        assert len(result["p_values"]) == 0

    def test_bad_values_filtered(self):
        data = {
            "effect_measures": [
                {"type": "OR", "value": "not_a_number", "ci_lower": None, "ci_upper": None, "context": "x"}
            ],
            "sample_sizes": [{"value": "abc"}],
            "p_values": [],
            "confidence_intervals": [],
            "beta_coefficients": [],
            "mean_differences": [],
            "standard_deviations": [],
            "weighted_statistics": [],
        }
        result = parse_extraction_response(json.dumps(data), page=1)
        assert len(result["effect_measures"]) == 0
        assert len(result["sample_sizes"]) == 0

    def test_non_list_category_skipped(self):
        data = {"effect_measures": "not a list", "p_values": []}
        result = parse_extraction_response(json.dumps(data), page=1)
        assert len(result["effect_measures"]) == 0

    def test_non_dict_item_skipped(self):
        data = {"effect_measures": ["not a dict", 42]}
        result = parse_extraction_response(json.dumps(data), page=1)
        assert len(result["effect_measures"]) == 0

    def test_all_categories_parsed(self):
        data = {
            "effect_measures": [{"type": "RR", "value": 1.5, "ci_lower": None, "ci_upper": None, "context": ""}],
            "confidence_intervals": [{"level": 95, "lower": 1.2, "upper": 3.8, "context": ""}],
            "p_values": [{"value": 0.03, "operator": "=", "context": ""}],
            "sample_sizes": [{"value": 100}],
            "beta_coefficients": [{"value": 0.5, "ci_lower": None, "ci_upper": None, "se": 0.1, "context": ""}],
            "mean_differences": [{"value": 2.0, "ci_lower": None, "ci_upper": None, "context": ""}],
            "standard_deviations": [{"value": 1.5, "mean": 5.0, "type": "SD", "context": ""}],
            "weighted_statistics": [{"stat_type": "prevalence", "value": 25.3, "weight_method": "IPW", "context": ""}],
        }
        result = parse_extraction_response(json.dumps(data), page=2)
        for cat in CATEGORIES:
            assert len(result[cat]) == 1, f"Expected 1 item in {cat}"

    def test_ci_requires_both_bounds(self):
        data = {
            "confidence_intervals": [
                {"level": 95, "lower": 1.2, "upper": None, "context": ""},
            ]
        }
        result = parse_extraction_response(json.dumps(data), page=1)
        assert len(result["confidence_intervals"]) == 0


# --- detect_provider ---


class TestDetectProvider:
    @patch("core.llm_providers.requests.get")
    @patch("core.llm_providers.get_api_key", return_value="fake-key")
    def test_gemini_when_key_present(self, mock_key, mock_get):
        with patch.dict("sys.modules", {"google.genai": MagicMock()}):
            from core.llm_providers import detect_provider

            result = detect_provider()
            assert result == "gemini"

    @patch("core.llm_providers.requests.get")
    @patch("core.llm_providers.get_api_key", return_value=None)
    def test_ollama_when_reachable(self, mock_key, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        from core.llm_providers import detect_provider

        result = detect_provider()
        assert result == "ollama"

    @patch("core.llm_providers.requests.get", side_effect=ConnectionError)
    @patch("core.llm_providers.get_api_key", return_value=None)
    def test_none_when_nothing_available(self, mock_key, mock_get):
        from core.llm_providers import detect_provider

        result = detect_provider()
        assert result is None

    @patch("core.llm_providers.requests.get")
    @patch("core.llm_providers.get_api_key", return_value="fake-key")
    def test_gemini_wins_over_ollama(self, mock_key, mock_get):
        """When both are available, Gemini takes priority."""
        mock_get.return_value = MagicMock(status_code=200)
        with patch.dict("sys.modules", {"google.genai": MagicMock()}):
            from core.llm_providers import detect_provider

            result = detect_provider()
            assert result == "gemini"


# --- Ollama provider ---


class TestOllamaProvider:
    @patch("core.llm_providers.ollama.requests.post")
    def test_extract_stats_success(self, mock_post):
        response_data = {
            "message": {
                "content": json.dumps({
                    "effect_measures": [{"type": "OR", "value": 2.5, "ci_lower": 1.2, "ci_upper": 3.8, "context": "x"}],
                    "confidence_intervals": [],
                    "p_values": [],
                    "sample_sizes": [],
                    "beta_coefficients": [],
                    "mean_differences": [],
                    "standard_deviations": [],
                    "weighted_statistics": [],
                })
            }
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from core.llm_providers.ollama import extract_stats

        result = extract_stats("some text", page=5)
        assert len(result["effect_measures"]) == 1
        assert result["effect_measures"][0]["value"] == 2.5
        assert result["effect_measures"][0]["page"] == 5

    @patch("core.llm_providers.ollama.requests.post", side_effect=ConnectionError)
    def test_extract_stats_failure(self, mock_post):
        from core.llm_providers.ollama import extract_stats

        result = extract_stats("some text", page=1)
        assert all(len(v) == 0 for v in result.values())

    @patch("core.llm_providers.ollama.requests.post")
    def test_chat_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "Hello world"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from core.llm_providers.ollama import chat

        result = chat("Say hello")
        assert result == "Hello world"

    @patch("core.llm_providers.ollama.requests.post", side_effect=ConnectionError)
    def test_chat_failure(self, mock_post):
        from core.llm_providers.ollama import chat

        result = chat("Say hello")
        assert result is None


# --- Gemini provider ---


class TestGeminiProvider:
    """Tests for Gemini provider with mocked google-genai SDK.

    Since google-genai may not be installed locally, we patch _get_client
    directly and inject fake google.genai.types via patch.dict(sys.modules)
    which auto-restores on exit.
    """

    def _genai_modules(self):
        """Create mock google.genai modules for patch.dict injection."""
        mock_genai = MagicMock()
        mock_types = MagicMock()
        mock_types.Content = MagicMock()
        mock_types.Part = MagicMock()
        mock_types.GenerateContentConfig = MagicMock()
        mock_genai.types = mock_types
        return {"google.genai": mock_genai, "google.genai.types": mock_types}

    def test_extract_stats_success(self):
        json_str = json.dumps({
            "effect_measures": [{"type": "HR", "value": 1.78, "ci_lower": 1.23, "ci_upper": 2.56, "context": "HR"}],
            "confidence_intervals": [],
            "p_values": [],
            "sample_sizes": [],
            "beta_coefficients": [],
            "mean_differences": [],
            "standard_deviations": [],
            "weighted_statistics": [],
        })
        mock_response = MagicMock()
        mock_response.text = json_str
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, self._genai_modules()):
            with patch("core.llm_providers.gemini._get_client", return_value=mock_client):
                from core.llm_providers.gemini import extract_stats

                result = extract_stats("some text", page=2)
                assert len(result["effect_measures"]) == 1
                assert result["effect_measures"][0]["value"] == 1.78
                assert result["effect_measures"][0]["page"] == 2

    def test_extract_stats_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, self._genai_modules()):
            with patch("core.llm_providers.gemini._get_client", return_value=mock_client):
                from core.llm_providers.gemini import extract_stats

                result = extract_stats("some text", page=1)
                assert all(len(v) == 0 for v in result.values())

    def test_chat_success(self):
        mock_response = MagicMock()
        mock_response.text = "Generated analysis"
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(sys.modules, self._genai_modules()):
            with patch("core.llm_providers.gemini._get_client", return_value=mock_client):
                from core.llm_providers.gemini import chat

                result = chat("Some question")
                assert result == "Generated analysis"

    def test_chat_failure(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        with patch.dict(sys.modules, self._genai_modules()):
            with patch("core.llm_providers.gemini._get_client", return_value=mock_client):
                from core.llm_providers.gemini import chat

                result = chat("Some question")
                assert result is None


# --- empty_results ---


class TestEmptyResults:
    def test_has_all_categories(self):
        result = _empty_results()
        assert len(result) == 8
        for cat in CATEGORIES:
            assert cat in result
            assert result[cat] == []

    def test_returns_new_dict(self):
        r1 = _empty_results()
        r2 = _empty_results()
        r1["p_values"].append({"value": 0.05})
        assert len(r2["p_values"]) == 0
