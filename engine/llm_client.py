import os
import requests


def call_llm(messages: list) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENROUTER_API_KEY.")

    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        "messages": messages,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise ValueError(f"Agent error: {body['error'].get('message', body['error'])}")
    if "choices" not in body or not body["choices"]:
        raise ValueError(f"Unexpected response from model: {body}")
    content = body["choices"][0]["message"]["content"]
    if not content:
        raise ValueError("Agent returned an empty response. The model may be rate-limited or unavailable.")
    return {"choices": [{"message": {"content": content}}]}
