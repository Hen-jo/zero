"""
LLMjudge config — reuses D:/paper/regen/.env for API keys and endpoints.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# regen .env 에서 API 키 로드
REGEN_DIR = Path(__file__).parent.parent.parent / "regen"
load_dotenv(REGEN_DIR / ".env")

# ── API Keys ─────────────────────────────────────────────────────────────────
HUB_PROJECT_ID: str | None = os.getenv("HUB_PROJECT_ID")
HUB_API_KEYS: list[str] = [
    k for i in range(1, 6)
    if (k := os.getenv(f"HUB_API_KEY_{i}"))
]
if not HUB_API_KEYS:
    single = os.getenv("HUB_API_KEY")
    if single:
        HUB_API_KEYS = [single]

if not HUB_API_KEYS:
    print("[LLMjudge] WARNING: No HUB_API_KEY_* found. Set them in D:/paper/regen/.env", file=sys.stderr)

# ── Endpoints ─────────────────────────────────────────────────────────────────
_ROOT = "https://internal-apigw-kr.hmg-corp.io/hchat-in/api/v3"
HUB_GPT_BASE    = _ROOT                       # AzureOpenAI base
HUB_CLAUDE_BASE = f"{_ROOT}/claude"           # Anthropic base
# Gemini: {_ROOT}/models/{model}:generateContent?key=...

# ── Models ────────────────────────────────────────────────────────────────────
# 기본 judge 앙상블 (3개 모델 평균)
DEFAULT_JUDGE_MODELS = ["claude-sonnet-4-6", "gpt-5.4", "gemini-3.1-pro-preview"]

# GPT Chat Completions API
GPT_MODELS = {
    "gpt-4o":       "gpt-4o",
    "gpt-4.1":      "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
}
# GPT Responses API (gpt-5+)
GPT_RESPONSES_MODELS = {
    "gpt-5":        "gpt-5",
    "gpt-5-mini":   "gpt-5-mini",
    "gpt-5.1":      "gpt-5.1",
    "gpt-5.2":      "gpt-5.2",
    "gpt-5.4":      "gpt-5.4",
    "gpt-5.4-mini": "gpt-5.4-mini",
}
CLAUDE_MODELS = {
    "claude-haiku-4-5":  "claude-haiku-4-5",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
}
GEMINI_MODELS = {
    "gemini-2.5-flash":       "gemini-2.5-flash",
    "gemini-2.5-pro":         "gemini-2.5-pro",
    "gemini-3-flash-preview":  "gemini-3-flash-preview",
    "gemini-3.1-pro-preview":  "gemini-3.1-pro-preview",
}
ALL_MODELS = {**GPT_MODELS, **GPT_RESPONSES_MODELS, **CLAUDE_MODELS, **GEMINI_MODELS}
