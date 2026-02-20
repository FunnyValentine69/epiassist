"""Ollama LLM provider — local inference via native HTTP API."""

import requests

from core.llm_providers._parse import parse_extraction_response
from core.llm_providers.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FEW_SHOT_ASSISTANT,
    FEW_SHOT_USER,
)

EXTRACTION_MODEL = "llama3.1:8b"
CHAT_MODEL = "llama3.2:latest"
BASE_URL = "http://localhost:11434"


def extract_stats(text: str, page: int = 1) -> dict[str, list[dict]]:
    """Extract epidemiological statistics from text via Ollama.

    Args:
        text: Page text to extract from.
        page: Page number for attribution.

    Returns:
        Dict with 8 category keys, each containing a list of extracted dicts.
    """
    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT},
        {"role": "user", "content": text},
    ]
    try:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": EXTRACTION_MODEL,
                "messages": messages,
                "format": "json",
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw_json = resp.json()["message"]["content"]
        return parse_extraction_response(raw_json, page)
    except Exception:
        from core.llm_providers._parse import _empty_results

        return _empty_results()


def chat(prompt: str) -> str | None:
    """General chat completion via Ollama.

    Args:
        prompt: The prompt to send.

    Returns:
        Response text or None on failure.
    """
    try:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception:
        return None
