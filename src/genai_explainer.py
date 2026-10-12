import os
import streamlit as st
from huggingface_hub import InferenceClient

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
def _build_prompt_context(player_data: dict, match_context: dict) -> str:
    """Formats player evidence metrics and model sensitivities into readable context."""
    sensitivities = player_data.get("sensitivities", [])
    sensitivity_str = ""
    if sensitivities:
        top_features = [
            f"- {item['feature']}: {item['value']} (Impact on score: {item['prediction_difference']:+.2f} pts vs median)"
            for item in sensitivities[:3]
        ]
        sensitivity_str = "\n".join(top_features)

    match_str = ""
    if match_context:
        teams = " vs ".join(match_context.get("teams", []))
        match_str = f"Match: {teams} | Venue: {match_context.get('venue', 'N/A')} | Date: {match_context.get('date', 'N/A')}"

    return f"""
Match Info:
{match_str}

Player Details:
- Name: {player_data.get('player', 'Unknown')}
- Role: {player_data.get('role', 'N/A')} ({player_data.get('team', 'N/A')})
- Model Predicted Fantasy Points: {player_data.get('predicted_points', 'N/A')}
- Recent Form / History:
  * Last match points: {player_data.get('last_fantasy_points', 'N/A')}
  * Previous 5-match avg: {player_data.get('last_5_average', 'N/A')}
  * Previous 10-match avg: {player_data.get('last_10_average', 'N/A')}
  * Past points against this opponent: {player_data.get('previous_opponent_points', 'N/A')}

Key Model Sensitivity Drivers (Feature Perturbations):
{sensitivity_str if sensitivity_str else 'N/A'}
"""


@st.cache_data(ttl=3600, show_spinner=False)
def generate_player_text_explanation(player_data: dict, match_context: dict = None, hf_token: str = None) -> str:
    """
    Calls Hugging Face Serverless API to explain why a player was selected by the model.
    """
    token = hf_token or os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
    
    if not token:
        return (
            "⚠️ **Hugging Face Token Missing:** Please add `HF_TOKEN` to your environment variables "
            "or `.streamlit/secrets.toml` to enable GenAI explanations."
        )

    try:
        client = InferenceClient(api_key=token)
        formatted_context = _build_prompt_context(player_data, match_context)

        system_instruction = (
            "You are a cricket fantasy analyst. "
            "In 2 to 3 concise, clear sentences, explain why this player was selected by the predictive model. "
            "Focus on their recent form, key feature drivers, role advantage, and opponent history. "
            "Do NOT use technical jargon like 'sensitivity', 'dataframe', or 'median imputation'."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Explain player selection based on this data:\n{formatted_context}"}
        ]

        response = client.chat_completion(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=150,
            temperature=0.6,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error generating explanation: {str(e)}"