import os
from typing import Dict, List, TypedDict

from dotenv import load_dotenv


load_dotenv()


class RunnableModel(TypedDict):
    name: str
    provider: str
    max_tokens: int
    temperature: float


class ModelProvider(TypedDict):
    base_url: str | None
    api_key: str


API_PROVIDERS: Dict[str, ModelProvider] = {
    "KISKI": {
        "base_url": "https://chat-ai.academiccloud.de/v1",
        "api_key": os.getenv("KISKI_API_KEY", ""),
    },
    "OLLAMA": {"base_url": "http://localhost:11434/v1", "api_key": ""},
    "ANTHROPIC": {
        "base_url": "https://api.anthropic.com/v1/",
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
    },
    "OPENAI": {"base_url": None, "api_key": os.getenv("OPENAI_API_KEY", "")},
}

MODELS_TO_RUN: List[RunnableModel] = [
    {
        "name": "llama-3.3-70b-instruct",
        "provider": "KISKI",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "qwen2.5-coder-32b-instruct",
        "provider": "KISKI",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "codestral-22b",
        "provider": "KISKI",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
    {
        "name": "gpt-oss:20b",
        "provider": "OLLAMA",
        "temperature": 0.2,
        "max_tokens": 10000,
    },
    {
        "name": "gpt-oss:120b",
        "provider": "OLLAMA",
        "temperature": 0.2,
        "max_tokens": 10000,
    },
    {
        "name": "qwen3-coder",
        "provider": "OLLAMA",
        "temperature": 0.2,
        "max_tokens": 10000,
    },
    {
        "name": "gpt-5-2025-08-07",
        "provider": "OPENAI",
        "temperature": 1,
        "max_tokens": 5000,
    },
    {
        "name": "claude-sonnet-4-20250514",
        "provider": "ANTHROPIC",
        "temperature": 0.2,
        "max_tokens": 5000,
    },
]
