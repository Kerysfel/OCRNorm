# pyright: reportMissingTypeStubs=false
# scripts/llm_inference.py
import os
import time
import requests  # type: ignore
import logging


def call_local_llm(
    model_name: str,
    user_text: str,
    temperature: float = 0.7,
    url: str | None = None,
    timeout_seconds: float | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> str:
    # Endpoint and timeout can be provided via params or environment variables.
    # Fallbacks ensure backward compatibility with local Ollama/llama.cpp style servers.
    endpoint_url = (
        url
        or os.getenv("LLM_ENDPOINT_URL")
        or os.getenv("LLM_API_BASE")
        or "http://localhost:1234/v1/chat/completions"
    )
    # Allow a tuple timeout (connect, read) or a single float; default to (5, 60)
    default_timeout = os.getenv("LLM_TIMEOUT_SECONDS")
    timeout: tuple[float, float]
    if timeout_seconds is None:
        if default_timeout is not None:
            try:
                t = float(default_timeout)
                timeout = (t, t)
            except Exception:
                timeout = (5.0, 60.0)
        else:
            timeout = (5.0, 60.0)
    else:
        if isinstance(timeout_seconds, (int, float)):
            t = float(timeout_seconds)
            timeout = (t, t)
        elif isinstance(timeout_seconds, (list, tuple)) and len(timeout_seconds) == 2:
            timeout = (float(timeout_seconds[0]), float(timeout_seconds[1]))
        else:
            timeout = (5.0, 60.0)

    payload: dict[str, object] = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": temperature,
        "stream": False,
    }

    # Optional generation controls
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)
    else:
        payload["max_tokens"] = -1
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None:
        payload["top_k"] = int(top_k)
    if seed is not None:
        payload["seed"] = int(seed)

    # Retry policy
    max_retries = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
    backoff_base = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(endpoint_url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                logging.error(
                    "LLM returned no choices. Payload accepted but empty response body structure."
                )
                return ""
        except Exception as e:
            if attempt < max_retries:
                sleep_s = backoff_base ** (attempt - 1)
                logging.warning(
                    "LLM request failed (attempt %d/%d): %s. Retrying in %.2fs...",
                    attempt,
                    max_retries,
                    e,
                    sleep_s,
                )
                try:
                    time.sleep(sleep_s)
                except Exception:
                    pass
            else:
                logging.error(
                    "LLM request failed after %d attempts: %s", max_retries, e
                )
                return ""

    # Should not reach here
    return ""
