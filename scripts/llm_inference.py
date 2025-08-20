# src/llm_inference.py
import requests


def call_local_llm(model_name: str, user_text: str, temperature: float = 0.7) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": temperature,
        "max_tokens": -1,
        "stream": False,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    if "choices" in data and len(data["choices"]) > 0:
        return data["choices"][0]["message"]["content"]
    else:
        # fallback
        return ""
