"""SAP ABAP-1 client using SAP Cloud SDK for AI (Python) v6 Orchestration V2 API.

Reference:
    SAP Cloud SDK for AI (Python) - generative v6.1.2
    Orchestration Service V2 API documentation
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping

from dotenv import load_dotenv


load_dotenv()

REQUIRED_AICORE_ENV_VARS = (
    "AICORE_AUTH_URL",
    "AICORE_CLIENT_ID",
    "AICORE_CLIENT_SECRET",
    "AICORE_BASE_URL",
    "AICORE_RESOURCE_GROUP",
)


def missing_aicore_env_vars() -> list[str]:
    return [name for name in REQUIRED_AICORE_ENV_VARS if not os.getenv(name, "").strip()]


def _extract_text(raw_content: Any) -> str:
    """Extract plain text from various response content formats."""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(raw_content)


@dataclass
class ABAP1OrchestrationClient:
    """Client for SAP ABAP-1 via SAP AI Core Orchestration Service V2.

    Uses the SAP Cloud SDK for AI (Python) ``gen_ai_hub.orchestration_v2``
    package (``sap-ai-sdk-gen``).  The orchestration service routes
    requests to the model specified by *model_name* through the
    orchestration deployment discovered automatically from the
    AICORE_* environment variables.

    SDK reference: SAP Cloud SDK for AI (Python) - generative v6.1.2
    """

    model_name: str
    model_version: str = "latest"
    temperature: float = 0.2
    max_tokens: int = 5000

    # Private fields (not part of the constructor)
    _service: Any = field(init=False, repr=False)
    _config: Any = field(init=False, repr=False)
    # Message classes stored for use in complete / complete_sync
    _SystemMessage: Any = field(init=False, repr=False)
    _UserMessage: Any = field(init=False, repr=False)
    _AssistantMessage: Any = field(init=False, repr=False)

    def __post_init__(self):
        try:
            # SAP Cloud SDK for AI (Python) v6 – Orchestration V2 imports
            from gen_ai_hub.orchestration_v2.models.config import (
                ModuleConfig,
                OrchestrationConfig,
            )
            from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
            from gen_ai_hub.orchestration_v2.models.message import (
                AssistantMessage,
                SystemMessage,
                UserMessage,
            )
            from gen_ai_hub.orchestration_v2.models.template import (
                PromptTemplatingModuleConfig,
                Template,
            )
            from gen_ai_hub.orchestration_v2.service import OrchestrationService
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'sap-ai-sdk-gen'. Install it with:\n"
                '  uv add "sap-ai-sdk-gen[all]>=5.11.1"'
            ) from exc

        # Store message classes for building history in complete_sync
        self._SystemMessage = SystemMessage
        self._UserMessage = UserMessage
        self._AssistantMessage = AssistantMessage

        # Step 1: Define the Template with a placeholder for the user query
        template = Template(
            template=[
                SystemMessage(content="{{?system_prompt}}"),
                UserMessage(content="{{?user_query}}"),
            ],
            defaults={"system_prompt": "You are a helpful assistant."},
        )

        # Step 2: Define the LLM
        llm = LLMModelDetails(
            name=self.model_name,
            version=self.model_version,
            params={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )

        # Step 3: Create the Orchestration Configuration
        prompt_template = PromptTemplatingModuleConfig(prompt=template, model=llm)
        module_config = ModuleConfig(prompt_templating=prompt_template)
        config = OrchestrationConfig(modules=module_config)

        # Step 4: Create the OrchestrationService (deployment URL is
        # resolved automatically from AICORE_* env vars)
        self._service = OrchestrationService(config=config)
        self._config = config

    @classmethod
    def from_env(
        cls,
        *,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 5000,
    ) -> "ABAP1OrchestrationClient":
        missing = missing_aicore_env_vars()
        if missing:
            missing_csv = ", ".join(missing)
            raise RuntimeError(
                "Missing SAP AI Core environment variables: "
                f"{missing_csv}. Set them in your .env file."
            )

        # Default model name: sap-abap-1 (single dash, as required by AI Core API)
        resolved_model_name = (model_name or os.getenv("AICORE_MODEL_NAME", "sap-abap-1")).strip()
        resolved_model_version = os.getenv("AICORE_MODEL_VERSION", "latest").strip() or "latest"
        return cls(
            model_name=resolved_model_name,
            model_version=resolved_model_version,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _build_history_and_placeholders(
        self, chat_history: Iterable[Mapping[str, Any]]
    ) -> tuple[dict[str, str], List[Any]]:
        """Convert a generic chat history into V2 API placeholder values + history.

        The V2 orchestration API uses:
        - ``placeholder_values`` for template variables (system_prompt, user_query)
        - ``history`` for prior conversation turns (prepended to template messages)

        This method:
        1. Extracts the first system message as ``system_prompt``
        2. Takes the last user message as ``user_query``
        3. All messages in between become ``history``
        """
        messages = list(chat_history)
        system_prompt = "You are a helpful assistant."
        history: List[Any] = []

        # Extract system message (if present, use the first one)
        non_system: list[Mapping[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if role == "system" and content:
                system_prompt = content
            else:
                non_system.append(msg)

        # Last user message becomes the placeholder, everything else is history
        user_query = ""
        history_messages = non_system

        # Find and pop the last user message
        for i in range(len(history_messages) - 1, -1, -1):
            role = str(history_messages[i].get("role", "")).strip().lower()
            if role == "user":
                user_query = str(history_messages[i].get("content", "")).strip()
                history_messages = history_messages[:i]
                break

        # Convert remaining messages to V2 message objects for history
        for msg in history_messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                history.append(self._UserMessage(content=content))
            elif role == "assistant":
                history.append(self._AssistantMessage(content=content))

        placeholder_values = {
            "system_prompt": system_prompt,
            "user_query": user_query,
        }
        return placeholder_values, history

    def complete_sync(self, chat_history: Iterable[Mapping[str, Any]]) -> str:
        """Send a synchronous chat-completion request through the Orchestration Service V2.

        Prefer the async ``complete()`` method for parallel workloads –
        it uses the SDK's native ``arun()`` and avoids thread-pool overhead.

        *chat_history* is a list of ``{"role": ..., "content": ...}``
        dicts.  The system message becomes the template system prompt,
        the last user message becomes the template user query, and any
        intermediate messages are passed as conversation history.
        """
        placeholder_values, history = self._build_history_and_placeholders(chat_history)

        # Synchronous orchestration request (Step 4 from V2 docs)
        response = self._service.run(
            placeholder_values=placeholder_values,
            history=history if history else None,
        )

        try:
            # V2 API uses final_result (not orchestration_result)
            content = response.final_result.choices[0].message.content
        except Exception:
            content = str(response)
        return _extract_text(content).strip()

    async def complete(self, chat_history: Iterable[Mapping[str, Any]]) -> str:
        """Send an async chat-completion request using the SDK's native ``arun()``.

        This avoids the serialization bottleneck of wrapping the synchronous
        ``run()`` in ``asyncio.to_thread()`` and allows true concurrent requests
        when multiple tasks call ``complete()`` simultaneously.
        """
        placeholder_values, history = self._build_history_and_placeholders(
            list(chat_history)
        )

        # Use SDK-native async method for true concurrency
        response = await self._service.arun(
            placeholder_values=placeholder_values,
            history=history if history else None,
        )

        try:
            content = response.final_result.choices[0].message.content
        except Exception:
            content = str(response)
        return _extract_text(content).strip()
