from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import Settings, get_settings
from .ollama_client import OllamaClient
from .rag import MarkdownRagStore, RagSearchHit, format_rag_context
from .tools import ToolExecution, ToolRegistry, build_default_registry


@dataclass(slots=True)
class AgentTurn:
    reply: str
    tool_events: list[ToolExecution]
    rag_hits: list[RagSearchHit] = field(default_factory=list)


@dataclass(slots=True)
class TeachingAgent:
    settings: Settings = field(default_factory=get_settings)
    registry: ToolRegistry = field(default_factory=build_default_registry)
    client: OllamaClient | None = None
    rag_store: MarkdownRagStore | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    max_tool_rounds: int = 6
    max_hallucination_retries: int = 2

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = OllamaClient(self.settings.ollama_host)
        if not self.messages:
            self.messages.append({"role": "system", "content": self.settings.system_prompt})
        if self.rag_store is None and self.settings.rag_auto_enabled and self._client_supports_embeddings():
            self.rag_store = self._build_default_rag_store()

    def run_turn(
        self,
        user_input: str,
        *,
        rag_hits: list[RagSearchHit] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> AgentTurn:
        turn_messages = list(self.messages)
        turn_messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "user", "content": user_input})

        if rag_hits is None:
            rag_hits = self._search_rag(user_input)
        rag_context = self._build_rag_context(rag_hits)
        if rag_context is not None:
            turn_messages.append(rag_context)

        self._write_debug_event(
            "user_input",
            {
                "user_input": user_input,
                "messages": self.messages,
            },
        )
        tool_events: list[ToolExecution] = []
        hallucination_retries = 0

        for _ in range(self.max_tool_rounds):
            request_body = {
                "model": self.settings.ollama_model,
                "messages": turn_messages,
                "tools": self.registry.schemas(),
            }
            self._write_debug_event("ollama_request", request_body)
            response = self._chat_response(turn_messages, on_text_chunk=on_text_chunk)
            self._write_debug_event("ollama_response", response)
            message = response.get("message", {})

            if self._looks_like_fake_tool_call(message):
                hallucination_retries += 1
                self._write_debug_event(
                    "hallucinated_tool_call",
                    {
                        "message": message,
                        "retry_count": hallucination_retries,
                    },
                )

                if hallucination_retries > self.max_hallucination_retries:
                    raise RuntimeError(
                        "Model produced tool-call-shaped content instead of real tool_calls too many times."
                    )

                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous assistant message looked like a fake tool call. "
                            "Do not invent tool names or put tool JSON in content. "
                            "Use only real tool_calls if you need a tool, otherwise answer normally."
                        ),
                    }
                )
                turn_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous assistant message looked like a fake tool call. "
                            "Do not invent tool names or put tool JSON in content. "
                            "Use only real tool_calls if you need a tool, otherwise answer normally."
                        ),
                    }
                )
                continue

            hallucination_retries = 0

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                self.messages.append(message)
                turn_messages.append(message)
                reply = (message.get("content") or "").strip()
                if reply:
                    return AgentTurn(reply=reply, tool_events=tool_events, rag_hits=rag_hits)

                self._write_debug_event(
                    "empty_assistant_response",
                    {
                        "user_input": user_input,
                        "rag_hit_count": len(rag_hits),
                    },
                )

                retry_reply = self._retry_empty_response(
                    user_input,
                    rag_hits,
                    on_text_chunk=on_text_chunk,
                )
                if retry_reply:
                    self.messages[-1] = {"role": "assistant", "content": retry_reply}
                    return AgentTurn(reply=retry_reply, tool_events=tool_events, rag_hits=rag_hits)

                if rag_hits:
                    fallback_reply = self._build_rag_fallback_reply(rag_hits)
                    if on_text_chunk is not None:
                        on_text_chunk(fallback_reply)
                    self.messages[-1] = {"role": "assistant", "content": fallback_reply}
                    return AgentTurn(reply=fallback_reply, tool_events=tool_events, rag_hits=rag_hits)

                fallback_reply = "The model returned an empty response."
                if on_text_chunk is not None:
                    on_text_chunk(fallback_reply)
                self.messages[-1] = {"role": "assistant", "content": fallback_reply}
                return AgentTurn(reply=fallback_reply, tool_events=tool_events, rag_hits=rag_hits)

            unknown_tool_name = self._unknown_tool_name(tool_calls)
            if unknown_tool_name is not None:
                hallucination_retries += 1
                self._write_debug_event(
                    "unknown_tool_call",
                    {
                        "message": message,
                        "unknown_tool_name": unknown_tool_name,
                        "retry_count": hallucination_retries,
                    },
                )

                if hallucination_retries > self.max_hallucination_retries:
                    raise RuntimeError(
                        f"Model requested unknown tool {unknown_tool_name!r} too many times."
                    )

                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous assistant message used an unknown tool name {unknown_tool_name!r}. "
                            "Use only real tool_calls with these tool names: "
                            f"{', '.join(self.registry.tool_names())}. "
                            "If none apply, answer normally."
                        ),
                    }
                )
                turn_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous assistant message used an unknown tool name {unknown_tool_name!r}. "
                            "Use only real tool_calls with these tool names: "
                            f"{', '.join(self.registry.tool_names())}. "
                            "If none apply, answer normally."
                        ),
                    }
                )
                continue

            self.messages.append(message)
            turn_messages.append(message)

            for tool_call in tool_calls:
                event = self.registry.execute_tool_call(tool_call)
                tool_events.append(event)
                tool_message = {"role": "tool", "name": event.name, "content": event.result}
                self.messages.append(tool_message)
                turn_messages.append(tool_message)

        raise RuntimeError("Tool loop exceeded the configured max_tool_rounds.")

    def _write_debug_event(self, event: str, payload: dict[str, Any]) -> None:
        if not self.settings.debug_log_path:
            return

        log_path = Path(self.settings.debug_log_path).expanduser()
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str))
            handle.write("\n")

    @staticmethod
    def _looks_like_fake_tool_call(message: dict[str, Any]) -> bool:
        if message.get("tool_calls"):
            return False

        content = message.get("content")
        if not isinstance(content, str):
            return False

        stripped = content.strip()
        candidate = TeachingAgent._unwrap_code_fence(stripped)
        if candidate is None:
            candidate = stripped

        if not candidate.startswith("{") or not candidate.endswith("}"):
            return False

        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return False

        return isinstance(payload, dict) and "name" in payload and "arguments" in payload

    @staticmethod
    def _unwrap_code_fence(content: str) -> str | None:
        if not content.startswith("```") or not content.endswith("```"):
            return None

        lines = content.splitlines()
        if len(lines) < 3:
            return None

        first_line = lines[0].strip()
        last_line = lines[-1].strip()
        if not first_line.startswith("```") or last_line != "```":
            return None

        inner = "\n".join(lines[1:-1]).strip()
        if not inner:
            return None

        if inner.startswith("json"):
            inner_lines = inner.splitlines()
            if len(inner_lines) == 1:
                return None
            inner = "\n".join(inner_lines[1:]).strip()

        return inner or None

    def _unknown_tool_name(self, tool_calls: list[dict[str, Any]]) -> str | None:
        for tool_call in tool_calls:
            function_payload = tool_call.get("function", tool_call)
            name = function_payload.get("name")
            if not isinstance(name, str) or not self.registry.has_tool(name):
                return str(name) if name is not None else "<missing>"
        return None

    def _build_default_rag_store(self) -> MarkdownRagStore | None:
        index_path = Path(self.settings.rag_index_path)
        if not index_path.exists() or not self._client_supports_embeddings():
            return None

        assert self.client is not None

        return MarkdownRagStore(
            workspace_root=Path.cwd(),
            index_path=index_path,
            client=self.client,
            embedding_model=self.settings.rag_embedding_model,
            chunk_size=self.settings.rag_chunk_size,
            chunk_overlap=self.settings.rag_chunk_overlap,
        )

    def _client_supports_embeddings(self) -> bool:
        return self.client is not None and hasattr(self.client, "embeddings")

    def _search_rag(self, user_input: str) -> list[RagSearchHit]:
        if self.rag_store is None or not self.settings.rag_auto_enabled:
            return []

        return self.rag_store.search(user_input, top_k=self.settings.rag_top_k)

    def _build_rag_context(self, hits: list[RagSearchHit]) -> dict[str, Any] | None:
        if not hits:
            return None

        return {
            "role": "system",
            "content": format_rag_context(hits),
        }

    def _retry_empty_response(
        self,
        user_input: str,
        rag_hits: list[RagSearchHit],
        *,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> str | None:
        retry_messages = list(self.messages[:-1])
        retry_messages.append(
            {
                "role": "system",
                "content": (
                    "The previous assistant response was empty. "
                    "Answer the user directly in plain text. "
                    "Do not call tools, do not emit tool JSON, and do not use code fences."
                ),
            }
        )
        retry_messages.append({"role": "user", "content": user_input})

        if rag_hits:
            retry_messages.append({"role": "system", "content": format_rag_context(rag_hits)})

        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "The previous assistant response was empty. Reply directly in one short paragraph. "
                    "Do not use tools or code fences. If relevant, cite the source path(s)."
                ),
            }
        )

        self._write_debug_event(
            "empty_response_retry",
            {
                "user_input": user_input,
                "rag_hit_count": len(rag_hits),
            },
        )
        if on_text_chunk is None:
            response = self.client.chat(
                model=self.settings.ollama_model,
                messages=retry_messages,
                tools=None,
            )
        else:
            response = self._chat_response(retry_messages, tools=None, on_text_chunk=on_text_chunk)
        self._write_debug_event("ollama_response", response)
        message = response.get("message", {})
        reply = (message.get("content") or "").strip()
        if reply:
            return reply
        return None

    def _chat_response(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        assert self.client is not None

        if on_text_chunk is None:
            return self.client.chat(
                model=self.settings.ollama_model,
                messages=messages,
                tools=tools if tools is not None else self.registry.schemas(),
            )

        aggregated_message: dict[str, Any] = {"role": "assistant", "content": ""}
        final_response: dict[str, Any] | None = None

        for chunk in self.client.stream_chat(
            model=self.settings.ollama_model,
            messages=messages,
            tools=tools if tools is not None else self.registry.schemas(),
        ):
            final_response = chunk
            message = chunk.get("message") or {}
            if not isinstance(message, dict):
                continue

            content = message.get("content")
            if isinstance(content, str) and content:
                aggregated_message["content"] = aggregated_message.get("content", "") + content
                on_text_chunk(content)

            if isinstance(message.get("role"), str):
                aggregated_message["role"] = message["role"]

            if "tool_calls" in message:
                aggregated_message["tool_calls"] = message["tool_calls"]

        if final_response is None:
            return {"message": aggregated_message}

        final_message = dict(final_response.get("message") or {})
        if aggregated_message.get("content"):
            final_message["content"] = aggregated_message["content"]
        elif "content" not in final_message:
            final_message["content"] = ""

        if aggregated_message.get("role"):
            final_message["role"] = aggregated_message["role"]

        if aggregated_message.get("tool_calls") is not None:
            final_message["tool_calls"] = aggregated_message["tool_calls"]

        final_response["message"] = final_message
        return final_response

    def _build_rag_fallback_reply(self, hits: list[RagSearchHit]) -> str:
        top_hit = hits[0]
        lines = [
            f"{top_hit.excerpt}",
            f"Source: {top_hit.citation}",
        ]
        if len(hits) > 1:
            extra_citations = ", ".join(hit.citation for hit in hits[1:3])
            lines.append(f"Related references: {extra_citations}")
        return "\n".join(lines)
