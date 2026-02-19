"""Shared UI helper functions for consistent styling across EpiAssist pages.

Provides dark-mode-compatible banners, plot download wrappers,
and robustness badge display.
"""

from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Styled banner — dark-mode safe replacement for Bootstrap-style HTML divs
# ---------------------------------------------------------------------------

_BANNER_STYLES = {
    "success": {"bg": "#1a472a", "border": "#2d6a4f", "text": "#a3d9a5"},
    "warning": {"bg": "#4a3800", "border": "#6b5200", "text": "#ffd966"},
    "error": {"bg": "#4a1a1a", "border": "#6b2d2d", "text": "#f4a0a0"},
    "info": {"bg": "#1a3a4a", "border": "#2d5a6b", "text": "#a0d4e4"},
}


def styled_banner(text: str, level: str = "success") -> None:
    """Render a dark-mode-compatible colored banner.

    Args:
        text: The message to display (supports basic markdown).
        level: One of 'success', 'warning', 'error', 'info'.
    """
    style = _BANNER_STYLES.get(level, _BANNER_STYLES["info"])
    st.markdown(
        f'<div style="'
        f"background-color: {style['bg']}; "
        f"border: 1px solid {style['border']}; "
        f"color: {style['text']}; "
        f"padding: 16px 20px; "
        f"border-radius: 8px; "
        f"text-align: center; "
        f"margin: 8px 0; "
        f"font-size: 1rem; "
        f"font-weight: 600;"
        f'">'
        f"{text}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Robustness badge for E-value display
# ---------------------------------------------------------------------------

def robustness_badge(e_value: float) -> None:
    """Display a dark-mode-safe robustness badge based on E-value.

    Args:
        e_value: The computed E-value.
    """
    if e_value >= 5:
        styled_banner(f"Robustness: Quite Robust (E = {e_value:.2f})", "success")
    elif e_value >= 3:
        styled_banner(f"Robustness: Moderately Robust (E = {e_value:.2f})", "warning")
    else:
        styled_banner(f"Robustness: Vulnerable (E = {e_value:.2f})", "error")


# ---------------------------------------------------------------------------
# Plot download — interactive HTML export for Plotly figures
# ---------------------------------------------------------------------------

def plot_download_button(
    fig: object,
    filename: str = "plot",
    label: str = "Download Plot (HTML)",
) -> None:
    """Add a download button for a Plotly figure as interactive HTML.

    Uses fig.to_html() — no kaleido dependency needed.

    Args:
        fig: A plotly.graph_objects.Figure.
        filename: Base filename (without extension).
        label: Button label text.
    """
    html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        label=label,
        data=html_bytes,
        file_name=f"{filename}.html",
        mime="text/html",
    )
