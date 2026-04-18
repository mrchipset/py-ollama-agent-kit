from __future__ import annotations

from dataclasses import dataclass, field

from ollama_agent_kit.agent import TeachingAgent
from ollama_agent_kit.config import Settings


@dataclass
class _SequencedClient:
    responses: list[dict]
    calls: list[dict] = field(default_factory=list)

    def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


def test_fake_tool_json_is_rejected_and_retried() -> None:
    client = _SequencedClient(
        responses=[
            {
                "message": {
                    "role": "assistant",
                    "content": '{"name": "greeting", "arguments": {"message": "hi"}}',
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                }
            },
        ]
    )
    agent = TeachingAgent(settings=Settings(), client=client)

    turn = agent.run_turn("你好")

    assert turn.reply == "Hello!"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "user"
    assert "fake tool call" in client.calls[1]["messages"][-1]["content"]


def test_fenced_fake_tool_json_is_rejected_and_retried() -> None:
    client = _SequencedClient(
        responses=[
            {
                "message": {
                    "role": "assistant",
                    "content": "```json\n{\"name\": \"greet\", \"arguments\": {\"message\": \"hi\"}}\n```",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello again!",
                }
            },
        ]
    )
    agent = TeachingAgent(settings=Settings(), client=client)

    turn = agent.run_turn("你好")

    assert turn.reply == "Hello again!"
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "user"
    assert "fake tool call" in client.calls[1]["messages"][-1]["content"]


def test_unknown_real_tool_call_is_rejected_and_retried() -> None:
    client = _SequencedClient(
        responses=[
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "greet",
                                "arguments": {},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I can answer without tools.",
                }
            },
        ]
    )
    agent = TeachingAgent(settings=Settings(), client=client)

    turn = agent.run_turn("你好")

    assert turn.reply == "I can answer without tools."
    assert len(client.calls) == 2
    assert client.calls[1]["messages"][-1]["role"] == "user"
    assert "unknown tool name" in client.calls[1]["messages"][-1]["content"]