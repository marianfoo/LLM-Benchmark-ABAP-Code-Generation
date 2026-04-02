import os
from typing import Dict, List, NotRequired, TypedDict

from dotenv import load_dotenv


load_dotenv()


class RunnableModel(TypedDict):
    name: str
    provider: str
    max_tokens: int
    temperature: float
    # Optional: only used by OPENAI_RESPONSES provider (gpt-5.3-codex etc.)
    reasoning_effort: NotRequired[str]


class ModelProvider(TypedDict, total=False):
    base_url: str | None
    api_key: str
    api_key_env: str


def _env(var_name: str) -> str:
    return os.getenv(var_name, "").strip()


def get_provider_api_key(provider_name: str) -> str:
    provider = API_PROVIDERS.get(provider_name)
    if provider is None:
        raise RuntimeError(f"Unknown provider '{provider_name}'")

    api_key = provider.get("api_key", "").strip()
    env_name = provider.get("api_key_env", "API_KEY")
    if api_key:
        return api_key

    raise RuntimeError(
        f"Missing API key for provider '{provider_name}'. "
        f"Set {env_name} in your .env file or shell environment."
    )


API_PROVIDERS: Dict[str, ModelProvider] = {
    "GROQ": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "api_key": _env("GROQ_API_KEY"),
    },
    "MISTRAL": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "api_key": _env("MISTRAL_API_KEY"),
    },
    "ANTHROPIC": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key_env": "ANTHROPIC_API_KEY",
        "api_key": _env("ANTHROPIC_API_KEY"),
    },
    "OPENAI": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api_key": _env("OPENAI_API_KEY"),
    },
    "OPENAI_DIRECT": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api_key": _env("OPENAI_API_KEY"),
    },
    # Responses API (/v1/responses) — for codex/reasoning models that don't support
    # /v1/chat/completions. Uses client.responses.create() instead of
    # client.chat.completions.create(). Same OPENAI_API_KEY, no batch support.
    "OPENAI_RESPONSES": {
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
        "api_key": _env("OPENAI_API_KEY"),
    },
    "SAP_AICORE": {
        "base_url": None,
        "api_key_env": "AICORE_CLIENT_SECRET",
        "api_key": _env("AICORE_CLIENT_SECRET"),
    },
    # DeepSeek: OpenAI-compatible API. deepseek-chat = V3.2 (general), deepseek-reasoner = R1 (reasoning)
    "DEEPSEEK": {
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "api_key": _env("DEEPSEEK_API_KEY"),
    },
    # Google Gemini: OpenAI-compatible endpoint. Supports batch API (50% discount, 24h window).
    # Use GEMINI_API_KEY from Google AI Studio (aistudio.google.com).
    "GOOGLE": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "api_key": _env("GEMINI_API_KEY"),
    },
    # Google Vertex AI: uses Application Default Credentials (gcloud auth application-default login).
    # Requires GOOGLE_CLOUD_PROJECT and GCS_BUCKET_NAME env vars.
    # Batch API with 50% cost reduction, no predefined rate limits.
    "GOOGLE_VERTEX": {
        "base_url": None,
        "api_key_env": "GOOGLE_CLOUD_PROJECT",
        "api_key": _env("GOOGLE_CLOUD_PROJECT"),
    },
    # Z.ai (Zhipu AI): OpenAI-compatible API. GLM-5.1 frontier model.
    # $0.72/$2.30 per 1M tokens (GLM-5 pricing; GLM-5.1 similar).
    "ZAI": {
        "base_url": "https://api.z.ai/api/paas/v4/",
        "api_key_env": "ZAI_API_KEY",
        "api_key": _env("ZAI_API_KEY"),
    },
}

MODELS_TO_RUN: List[RunnableModel] = [
    # --- Groq Models ---
    {
        "name": "llama-3.3-70b-versatile",  # Groq: Llama 3.3 70B Versatile 128k - $0.59/$0.79 per 1M tokens
        "provider": "GROQ",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "qwen/qwen3-32b",  # Groq: Qwen3 32B 131k - $0.29/$0.59 per 1M tokens
        "provider": "GROQ",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "openai/gpt-oss-20b",  # Groq: GPT OSS 20B 128k - $0.075/$0.30 per 1M tokens
        "provider": "GROQ",
        "temperature": 0.2,
        "max_tokens": 10000,
    },
    {
        "name": "openai/gpt-oss-120b",  # Groq: GPT OSS 120B 128k - $0.15/$0.60 per 1M tokens
        "provider": "GROQ",
        "temperature": 0.2,
        "max_tokens": 10000,
    },
    # --- Mistral Models ---
    {
        "name": "codestral-latest",  # Mistral: Codestral - $0.30/$0.90 per 1M tokens
        "provider": "MISTRAL",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "mistral-large-2512",  # Mistral: Large 3 (675B MoE, 41B active) - $0.50/$1.50 per 1M tokens
        "provider": "MISTRAL",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    # --- Closed-Source Models ---
    {
        "name": "gpt-5.2",  # OpenAI: GPT-5.2 - $1.25/$10.00 per 1M tokens (batch)
        "provider": "OPENAI",
        "temperature": 1,  # Note: GPT-5 temperature cannot be changed per OpenAI restrictions
        "max_tokens": 5000,
    },
    {
        "name": "gpt-5.2-codex",  # OpenAI: GPT-5.2 Codex (non-batch, direct API calls)
        "provider": "OPENAI_DIRECT",
        "temperature": 1,  # Note: GPT-5 temperature cannot be changed per OpenAI restrictions
        "max_tokens": 5000,
    },
    {
        # OpenAI: GPT-5.3 Codex - $1.75/$14.00 per 1M tokens
        # Uses /v1/responses (Responses API), not /v1/chat/completions
        "name": "gpt-5.3-codex",
        "provider": "OPENAI_RESPONSES",
        "temperature": 1,
        "max_tokens": 5000,
        "reasoning_effort": "medium",
    },
    {
        "name": "claude-opus-4-6",  # Anthropic: Claude Opus 4.6 - $5.00/$25.00 per 1M tokens, 1M context window
        "provider": "ANTHROPIC",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "claude-opus-4-5-20251101",  # Anthropic: Claude Opus 4.5 - $5.00/$25.00 per 1M tokens
        "provider": "ANTHROPIC",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "claude-haiku-4-5-20251001",  # Anthropic: Claude Haiku 4.5 - fastest near-frontier model, $0.80/$4.00 per 1M tokens
        "provider": "ANTHROPIC",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    # --- DeepSeek Models ---
    {
        # DeepSeek: R1 reasoning model (comparable to gpt-5.3-codex) - $0.55/$2.19 per 1M tokens
        # OpenAI-compatible API, no batch support — uses parallel runner.
        # temperature is silently ignored by deepseek-reasoner (per API docs).
        # max_tokens covers both CoT reasoning + final answer (API default 32K, max 64K).
        # Multi-turn: only response.choices[0].message.content is saved (reasoning_content is dropped),
        # which matches DeepSeek's requirement to not replay reasoning_content in history.
        "name": "deepseek-reasoner",
        "provider": "DEEPSEEK",
        "temperature": 0.2,
        "max_tokens": 16000,
    },
    {
        "name": "sap-abap-1",  # SAP: ABAP-1 via SAP AI Core orchestration deployment
        "provider": "SAP_AICORE",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    # --- Google Gemini Models ---
    {
        # Google: Gemini 3.1 Pro Preview — latest frontier reasoning model (Feb 2026).
        # Uses Vertex AI Batch API (global endpoint) — 50% cost reduction, no rate limits.
        # Requires: gcloud auth application-default login, GOOGLE_CLOUD_PROJECT, GCS_BUCKET_NAME.
        "name": "gemini-3.1-pro-preview",
        "provider": "GOOGLE_VERTEX",
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    {
        # Google: Gemini 3 Flash Preview — fast frontier-class model (Dec 2025).
        # OpenAI-compatible batch API supported at 50% cost reduction.
        "name": "gemini-3-flash-preview",
        "provider": "GOOGLE",
        "temperature": 0.2,
        "max_tokens": 8192,
    },
    # --- Z.ai (Zhipu AI) Models ---
    {
        # Z.ai: GLM-5 — 745B MoE (44B active), 205K context. $0.72/$2.30 per 1M tokens.
        "name": "glm-5",
        "provider": "ZAI",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
]
