
import os
import streamlit as st
from huggingface_hub import InferenceClient

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _build_prompt_context(
    player_data: dict,
    match_context: dict,
) -> str:
    """Format player evidence into readable context."""

    sensitivities = player_data.get("sensitivities", [])
    sensitivity_str = ""

    if sensitivities:
        top_features = [
            (
                f"- {item['feature']}: {item['value']} "
                f"(Impact on score: "
                f"{item['prediction_difference']:+.2f} pts vs median)"
            )
            for item in sensitivities[:3]
        ]
        sensitivity_str = "\n".join(top_features)

    match_str = ""

    if match_context:
        teams = " vs ".join(match_context.get("teams", []))
        match_str = (
            f"Match: {teams} | "
            f"Venue: {match_context.get('venue', 'N/A')} | "
            f"Date: {match_context.get('date', 'N/A')}"
        )

    return f"""
Match Info:
{match_str}

Player Details:
- Name: {player_data.get('player', 'Unknown')}
- Role: {player_data.get('role', 'N/A')}
- Team: {player_data.get('team', 'N/A')}
- Model Predicted Fantasy Points: {player_data.get('predicted_points', 'N/A')}
- Last match points: {player_data.get('last_fantasy_points', 'N/A')}
- Previous 5-match average: {player_data.get('last_5_average', 'N/A')}
- Previous 10-match average: {player_data.get('last_10_average', 'N/A')}
- Points against this opponent: {player_data.get('previous_opponent_points', 'N/A')}

Key Model Drivers:
{sensitivity_str if sensitivity_str else 'N/A'}
"""


def generate_player_text_explanation(
    player_data: dict,
    match_context: dict = None,
    hf_token: str = None,
) -> str:
    """Generate a player explanation using Hugging Face."""

    token = hf_token

    if not token:
        token = os.getenv("HF_TOKEN")

    if not token:
        try:
            token = st.secrets.get("HF_TOKEN", "")
        except Exception:
            token = ""

    token = token.strip() if token else ""

    if not token:
        return (
            "HF_TOKEN is missing. Add it to "
            "Streamlit Cloud → App settings → Secrets."
        )

    if not token.startswith("hf_"):
        return (
            "HF_TOKEN does not look like a Hugging Face access token. "
            "Check that you entered the actual token value."
        )

    try:
        client = InferenceClient(
            model=DEFAULT_MODEL,
            token=token,
        )

        formatted_context = _build_prompt_context(
            player_data,
            match_context or {},
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cricket fantasy analyst. Explain in "
                    "2 to 3 concise sentences why the player was "
                    "selected. Focus on recent form, role and opponent "
                    "history. Do not use technical jargon. Do not invent "
                    "facts not present in the supplied data."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Explain this player's selection using the data:\n"
                    f"{formatted_context}"
                ),
            },
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.6,
        )

        explanation = response.choices[0].message.content

        if not explanation:
            return "The model returned an empty explanation."

        return explanation.strip()

    except Exception as e:
        error = str(e)

        if "401" in error or "Unauthorized" in error:
            return (
                "Hugging Face authentication failed (401). Check that "
                "Streamlit Cloud Secrets contains the current, valid "
                "HF_TOKEN with permission to call Inference Providers. "
                "If the token is correct, verify that it has not expired "
                "or been revoked."
            )

        return f"Error generating explanation: {error}"