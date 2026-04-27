from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path

from ollama_agent_kit.agent import TeachingAgent
from ollama_agent_kit.config import Settings


@dataclass
class _EchoClient:
    calls: list[dict] = field(default_factory=list)

    def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        turn_number = len(self.calls)
        return {
            "message": {
                "role": "assistant",
                "content": f"reply {turn_number}",
            }
        }


def test_agent_trims_history_to_configured_window() -> None:
    client = _EchoClient()
    settings = Settings(context_max_messages=2)
    agent = TeachingAgent(settings=settings, client=client)

    first_turn = agent.run_turn("Question 1")
    second_turn = agent.run_turn("Question 2")
    third_turn = agent.run_turn("Question 3")

    assert first_turn.reply == "reply 1"
    assert second_turn.reply == "reply 2"
    assert third_turn.reply == "reply 3"
    assert len(agent.messages) == 3
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[1]["content"] == "Question 3"
    assert agent.messages[2]["content"] == "reply 3"
    assert agent.conversation_summary is not None
    assert "Question 1" in agent.conversation_summary
    assert "reply 1" in agent.conversation_summary

    third_call_messages = client.calls[2]["messages"]
    assert any(
        message["role"] == "system" and "Conversation summary of earlier context" in message.get("content", "")
        for message in third_call_messages
    )
    assert any("Question 1" in message.get("content", "") for message in third_call_messages)
    assert any(message.get("content") == "Question 3" for message in third_call_messages)
    assert all(message.get("content") != "Question 1" for message in third_call_messages)
    assert all(message.get("content") != "reply 1" for message in third_call_messages)


def test_agent_preserves_image_messages_across_turns(tmp_path: Path) -> None:
    client = _EchoClient()
    agent = TeachingAgent(settings=Settings(), client=client)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image-bytes")

    first_turn = agent.run_turn("Look at this image", images=[image_path])
    second_turn = agent.run_turn("What about the previous image?")

    encoded_image = base64.b64encode(b"image-bytes").decode("utf-8")

    assert first_turn.reply == "reply 1"
    assert second_turn.reply == "reply 2"
    assert client.calls[0]["messages"][-1]["images"] == [encoded_image]
    assert any(
        message.get("role") == "user" and message.get("images") == [encoded_image]
        for message in client.calls[1]["messages"]
    )
    assert client.calls[1]["messages"][-1]["content"] == "What about the previous image?"