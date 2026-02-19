"""Tests for utils/ui_helpers.py shared UI functions."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# styled_banner
# ---------------------------------------------------------------------------

class TestStyledBanner:
    """Tests for the styled_banner function."""

    @patch("utils.ui_helpers.st")
    def test_success_banner(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Test message", "success")
        mock_st.markdown.assert_called_once()
        html = mock_st.markdown.call_args[0][0]
        assert "Test message" in html
        assert "#1a472a" in html  # success bg color

    @patch("utils.ui_helpers.st")
    def test_warning_banner(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Warning text", "warning")
        html = mock_st.markdown.call_args[0][0]
        assert "#4a3800" in html  # warning bg color

    @patch("utils.ui_helpers.st")
    def test_error_banner(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Error text", "error")
        html = mock_st.markdown.call_args[0][0]
        assert "#4a1a1a" in html  # error bg color

    @patch("utils.ui_helpers.st")
    def test_info_banner(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Info text", "info")
        html = mock_st.markdown.call_args[0][0]
        assert "#1a3a4a" in html  # info bg color

    @patch("utils.ui_helpers.st")
    def test_unknown_level_defaults_to_info(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Text", "nonexistent")
        html = mock_st.markdown.call_args[0][0]
        assert "#1a3a4a" in html  # info (default) bg color

    @patch("utils.ui_helpers.st")
    def test_unsafe_allow_html_enabled(self, mock_st):
        from utils.ui_helpers import styled_banner
        styled_banner("Text", "success")
        _, kwargs = mock_st.markdown.call_args
        assert kwargs.get("unsafe_allow_html") is True


# ---------------------------------------------------------------------------
# robustness_badge
# ---------------------------------------------------------------------------

class TestRobustnessBadge:
    """Tests for the robustness_badge function."""

    @patch("utils.ui_helpers.styled_banner")
    def test_high_e_value_success(self, mock_banner):
        from utils.ui_helpers import robustness_badge
        robustness_badge(6.0)
        mock_banner.assert_called_once()
        text = mock_banner.call_args[0][0]
        level = mock_banner.call_args[0][1]
        assert "Quite Robust" in text
        assert level == "success"

    @patch("utils.ui_helpers.styled_banner")
    def test_moderate_e_value_warning(self, mock_banner):
        from utils.ui_helpers import robustness_badge
        robustness_badge(4.0)
        text = mock_banner.call_args[0][0]
        level = mock_banner.call_args[0][1]
        assert "Moderately Robust" in text
        assert level == "warning"

    @patch("utils.ui_helpers.styled_banner")
    def test_low_e_value_error(self, mock_banner):
        from utils.ui_helpers import robustness_badge
        robustness_badge(1.5)
        text = mock_banner.call_args[0][0]
        level = mock_banner.call_args[0][1]
        assert "Vulnerable" in text
        assert level == "error"

    @patch("utils.ui_helpers.styled_banner")
    def test_boundary_e_value_5(self, mock_banner):
        from utils.ui_helpers import robustness_badge
        robustness_badge(5.0)
        text = mock_banner.call_args[0][0]
        assert "Quite Robust" in text

    @patch("utils.ui_helpers.styled_banner")
    def test_boundary_e_value_3(self, mock_banner):
        from utils.ui_helpers import robustness_badge
        robustness_badge(3.0)
        text = mock_banner.call_args[0][0]
        assert "Moderately Robust" in text


# ---------------------------------------------------------------------------
# plot_download_button
# ---------------------------------------------------------------------------

class TestPlotDownloadButton:
    """Tests for the plot_download_button function."""

    @patch("utils.ui_helpers.st")
    def test_creates_download_button(self, mock_st):
        from utils.ui_helpers import plot_download_button

        mock_fig = MagicMock()
        mock_fig.to_html.return_value = "<html>plot</html>"

        plot_download_button(mock_fig, filename="test_plot")

        mock_fig.to_html.assert_called_once_with(include_plotlyjs="cdn")
        mock_st.download_button.assert_called_once()
        kwargs = mock_st.download_button.call_args[1]
        assert kwargs["file_name"] == "test_plot.html"
        assert kwargs["mime"] == "text/html"

    @patch("utils.ui_helpers.st")
    def test_custom_label(self, mock_st):
        from utils.ui_helpers import plot_download_button

        mock_fig = MagicMock()
        mock_fig.to_html.return_value = "<html></html>"

        plot_download_button(mock_fig, label="Custom Label")

        kwargs = mock_st.download_button.call_args[1]
        assert kwargs["label"] == "Custom Label"

    @patch("utils.ui_helpers.st")
    def test_html_bytes_encoding(self, mock_st):
        from utils.ui_helpers import plot_download_button

        mock_fig = MagicMock()
        mock_fig.to_html.return_value = "<html>α β γ</html>"

        plot_download_button(mock_fig)

        kwargs = mock_st.download_button.call_args[1]
        data = kwargs["data"]
        assert isinstance(data, bytes)
        assert "α β γ".encode("utf-8") in data
