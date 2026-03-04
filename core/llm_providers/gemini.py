"""Gemini LLM provider — cloud inference via google-genai SDK.

The google-genai import is deferred to function calls so this module
can be imported (and tested) even when the package is not installed.
"""

import logging

from core.llm_providers import get_api_key
from core.llm_providers._parse import _empty_results, parse_extraction_response
from core.llm_providers.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    FEW_SHOT_ASSISTANT,
    FEW_SHOT_USER,
)

logger = logging.getLogger(__name__)

EXTRACTION_MODEL = "gemini-2.0-flash"
CHAT_MODEL = "gemini-2.0-flash"


def _get_client():
    """Create a Gemini client with the configured API key."""
    from google import genai

    api_key = get_api_key("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


def extract_stats(text: str, page: int = 1) -> dict[str, list[dict]]:
    """Extract epidemiological statistics from text via Gemini.

    Args:
        text: Page text to extract from.
        page: Page number for attribution.

    Returns:
        Dict with 8 category keys, each containing a list of extracted dicts.
    """
    try:
        from google.genai import types

        client = _get_client()
        response = client.models.generate_content(
            model=EXTRACTION_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=FEW_SHOT_USER)],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part(text=FEW_SHOT_ASSISTANT)],
                ),
                types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=EXTRACTION_SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return parse_extraction_response(response.text, page)
    except Exception as e:
        logger.warning("Gemini extraction failed: %s", e)
        return _empty_results()


def chat(prompt: str) -> str | None:
    """General chat completion via Gemini.

    Args:
        prompt: The prompt to send.

    Returns:
        Response text or None on failure.
    """
    try:
        from google.genai import types

        client = _get_client()
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        logger.warning("Gemini chat failed: %s", e)
        return None
