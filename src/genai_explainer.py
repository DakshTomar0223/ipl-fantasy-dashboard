"""Gemini-powered player explanation helper for the IPL Fantasy Dashboard."""

import json
import os
from typing import Any

import streamlit as st
from google import genai


# Model used for concise player-form and fantasy-selection explanations.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _build_prompt_context(player_data: dict, match_context: dict) -> str:
    """Serialize player and match data safely for the model prompt."""
    return json.dumps(
        {"player": player_data or {}, "match": match_context or {}},
        indent=2,
        default=str,
        ensure_ascii=False,
    )


def _get_api_key() -> str:
    """Read the Gemini key from Streamlit Secrets, then environment."""
    try:
        api_key = st.secrets.get("HF_TOKEN", "")
    except Exception:
        api_key = ""

    # Keep HF_TOKEN as the configured secret name for compatibility with
    # the user's existing Streamlit setup. Its VALUE must be a Gemini API key.
    api_key = api_key or os.getenv("HF_TOKEN", "")
    return api_key.strip() if api_key else ""


def generate_player_text_explanation(
    player_data: dict,
    match_context: dict = None,
    hf_token: str = None,
) -> str:
    """Generate a short, evidence-grounded explanation using Gemini.

    The hf_token parameter remains for compatibility with the existing UI,
    but the key is read from HF_TOKEN Secrets/environment to avoid passing
    credentials through UI calls.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "Gemini API key is missing. Add HF_TOKEN to Streamlit Cloud "
            "Secrets, using a Gemini API key as its value."
        )

    context = _build_prompt_context(player_data, match_context or {})
    prompt = f"""You are an IPL fantasy-cricket analyst.

Explain the player's fantasy-selection evidence for the specified match.
Use only facts present in the supplied data. Do not invent statistics,
injuries, pitch conditions, player availability, or match facts. If a useful
field is missing, say that it is unavailable. In 3-5 concise sentences,
discuss recent form, role, opportunity, upside, and risk only where supported
by the data. End with a brief selection takeaway. Do not guarantee future
performance or fantasy points.

PLAYER AND MATCH DATA:
{context}
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if not text:
            return "Gemini returned an empty explanation. Please try again."
        return text

    except Exception as exc:
        message = str(exc)
        lowered = message.lower()

        if any(term in lowered for term in ("api key not valid", "unauthenticated", "401", "invalid api key")):
            return (
                "Gemini authentication failed. Check that HF_TOKEN in "
                "Streamlit Secrets contains a valid Gemini API key from Google AI Studio."
            )
        if "429" in lowered or "resource_exhausted" in lowered or "quota" in lowered:
            return (
                "Gemini API quota or rate limit reached. Wait and try again, "
                "or check the limits for your Google AI Studio project."
            )
        if "not found" in lowered and "model" in lowered:
            return (
                f"Gemini model '{DEFAULT_MODEL}' is unavailable for this API key. "
                "Check the current model list in Google AI Studio."
            )
        # Do not include the API key in the returned error message.
        return f"Could not generate the Gemini explanation: {message[:500]}"
