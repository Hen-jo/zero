"""
Text-only LLM client for LLMjudge.
Supports GPT (Azure), Claude (Anthropic), Gemini (REST).
No image support — judge only needs text I/O.
"""
import threading
import requests
from openai import AzureOpenAI
import anthropic

from config import (
    HUB_API_KEYS, HUB_PROJECT_ID,
    HUB_GPT_BASE, HUB_CLAUDE_BASE,
    GPT_MODELS, GPT_RESPONSES_MODELS, CLAUDE_MODELS, GEMINI_MODELS, ALL_MODELS,
)

_ROOT = "https://internal-apigw-kr.hmg-corp.io/hchat-in/api/v3"


# ── Thread-safe round-robin key pool ─────────────────────────────────────────
class KeyPool:
    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("No API keys available")
        self._keys = keys
        self._idx = 0
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
        return key


_key_pool = KeyPool(HUB_API_KEYS)
_extra_headers = {"X-Project-Id": HUB_PROJECT_ID} if HUB_PROJECT_ID else {}


# ── GPT Chat Completions ──────────────────────────────────────────────────────
def _gpt_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=HUB_GPT_BASE,
        api_key=_key_pool.get(),
        api_version="2024-10-21",
        default_headers=_extra_headers or None,
    )


def _call_gpt(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = _gpt_client().chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content


# ── GPT Responses API (gpt-5+) ────────────────────────────────────────────────
def _call_gpt_responses(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    combined = f"{system}\n\n{user}" if system else user
    payload = {
        "model": model_id,
        "input": [{"role": "user", "content": combined}],
        "max_output_tokens": max_tokens,
    }
    api_key = _key_pool.get()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        **_extra_headers,
    }
    r = requests.post(f"{_ROOT}/openai/responses", json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    data = r.json()

    if data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason", "unknown")
        raise RuntimeError(f"Responses API incomplete: {reason}")

    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part["text"]

    raise ValueError(f"Responses API: no output_text. status={data.get('status')}")


# ── Claude ────────────────────────────────────────────────────────────────────
def _claude_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=HUB_CLAUDE_BASE,
        api_key=_key_pool.get(),
        default_headers=_extra_headers,
    )


def _call_claude(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    kwargs = {}
    if system:
        kwargs["system"] = system

    message = _claude_client().messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    return message.content[0].text


# ── Gemini ────────────────────────────────────────────────────────────────────
def _call_gemini(model_id: str, system: str, user: str, max_tokens: int = 512) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.0,
            "responseMimeType": "application/json",  # force JSON output
        },
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}

    headers = {"Content-Type": "application/json"}
    if HUB_PROJECT_ID:
        headers["X-Project-Id"] = HUB_PROJECT_ID

    api_key = _key_pool.get()
    url = f"{_ROOT}/models/{model_id}:generateContent?key={api_key}"
    r = requests.post(url, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


# ── Public interface ──────────────────────────────────────────────────────────
def call_llm(model_key: str, system: str, user: str, max_tokens: int = 512) -> str:
    """
    Call any supported model by its key (e.g. 'gpt-4.1-mini', 'claude-haiku-4-5').
    Returns the text response.
    """
    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(ALL_MODELS)}")

    model_id = ALL_MODELS[model_key]

    if model_key in GPT_RESPONSES_MODELS:
        return _call_gpt_responses(model_id, system, user, max_tokens)
    elif model_key in GPT_MODELS:
        return _call_gpt(model_id, system, user, max_tokens)
    elif model_key in CLAUDE_MODELS:
        return _call_claude(model_id, system, user, max_tokens)
    elif model_key in GEMINI_MODELS:
        return _call_gemini(model_id, system, user, max_tokens)
    else:
        raise ValueError(f"No backend registered for model: {model_key}")
