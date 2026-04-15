from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings, get_settings
from .ollama_client import OllamaClient
from .tools import ToolExecution, ToolRegistry, build_default_registry


@dataclass(slots=True)
class AgentTurn:
    reply: str
    tool_events: list[ToolExecution]


@dataclass(slots=True)
class TeachingAgent:
    settings: Settings = field(default_factory=get_settings)
    registry: ToolRegistry = field(default_factory=build_default_registry)
    client: OllamaClient | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 6

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OllamaClient(self.settings.ollama_host)
        if not self.messages:
            self.messages.append({"role": "system", "content": self.settings.system_prompt})

    def run_turn(self, user_input: str) -> AgentTurn:
        self.messages.append({"role": "user", "content": user_input})
        tool_events: list[ToolExecution] = []

        for _ in range(self.max_tool_rounds):
            response = self.client.chat(
                model=self.settings.ollama_model,
                messages=self.messages,
                tools=self.registry.schemas(),
            )
            message = response.get("message", {})
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                reply = (message.get("content") or "").strip()
                return AgentTurn(reply=reply or "No response returned.", tool_events=tool_events)

            for tool_call in tool_calls:
                event = self.registry.execute_tool_call(tool_call)
                tool_events.append(event)
                self.messages.append(
                    {
                        "role": "tool",
                        "name": event.name,
                        "content": event.result,
                    }
                )

        raise RuntimeError("Tool loop exceeded the configured max_tool_rounds.")
