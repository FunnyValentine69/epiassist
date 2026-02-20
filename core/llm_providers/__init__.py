"""LLM provider registry and auto-detection.

Supports Gemini (cloud) and Ollama (local). Provider selection is automatic:
1. GEMINI_API_KEY present + google-genai installed -> Gemini
2. Ollama reachable at localhost:11434 -> Ollama
3. Neither -> None (LLM features hidden)
"""

import os
from typing import Optional

import requests


def get_api_key(key_name: str) -> Optional[str]:
    """Get an API key from st.secrets or environment variables.

    Args:
        key_name: Name of the key (e.g., "GEMINI_API_KEY").

    Returns:
        Key value or None if not found.
    """
    try:
        import streamlit as st

        val = st.secrets.get(key_name)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key_name) or None


def detect_provider() -> Optional[str]:
    """Auto-detect the best available LLM provider.

    Returns:
        "gemini", "ollama", or None.
    """
    # Priority 1: Gemini (cloud)
    if get_api_key("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401

            return "gemini"
        except ImportError:
            pass

    # Priority 2: Ollama (local)
    try:
        resp = requests.get("http://localhost:11434", timeout=2)
        if resp.status_code == 200:
            return "ollama"
    except Exception:
        pass

    return None


def get_provider_functions(name: str) -> dict:
    """Lazy-import a provider module and return its functions.

    Args:
        name: Provider name ("gemini" or "ollama").

    Returns:
        Dict with "extract_stats" and "chat" callables.
    """
    if name == "gemini":
        from core.llm_providers.gemini import chat, extract_stats

        return {"extract_stats": extract_stats, "chat": chat}
    elif name == "ollama":
        from core.llm_providers.ollama import chat, extract_stats

        return {"extract_stats": extract_stats, "chat": chat}
    else:
        raise ValueError(f"Unknown provider: {name}")


def provider_display_name(name: Optional[str]) -> str:
    """Human-readable display name for a provider.

    Args:
        name: Provider name or None.

    Returns:
        Display string like "Gemini (cloud)" or "Ollama (local)".
    """
    names = {
        "gemini": "Gemini (cloud)",
        "ollama": "Ollama (local)",
    }
    return names.get(name, "None")
