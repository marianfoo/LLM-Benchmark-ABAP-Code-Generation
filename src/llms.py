import os
from typing import Dict, List, TypedDict

from dotenv import load_dotenv


load_dotenv()


class RunnableModel(TypedDict):
    name: str
    provider: str
    max_tokens: int
    temperature: float


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
    "SAP_AICORE": {
        "base_url": None,
        "api_key_env": "AICORE_CLIENT_SECRET",
        "api_key": _env("AICORE_CLIENT_SECRET"),
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
    # --- Closed-Source Models ---
    {
        "name": "gpt-5.2",  # OpenAI: GPT-5.1 - $1.25/$10.00 per 1M tokens (Standard)
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
        "name": "claude-opus-4-5-20251101",  # Anthropic: Claude Opus 4.5 - $5.00/$25.00 per 1M tokens
        "provider": "ANTHROPIC",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "sap-abap-1",  # SAP: ABAP-1 via SAP AI Core orchestration deployment
        "provider": "SAP_AICORE",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
]
