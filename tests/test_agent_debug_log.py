from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ollama_agent_kit.agent import TeachingAgent
from ollama_agent_kit.config import Settings


@dataclass
class _FakeClient:
    response: dict

    def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        return self.response


def test_run_turn_writes_debug_log(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "ollama-debug.jsonl"
    client = _FakeClient(
        response={
            "message": {
                "role": "assistant",
                "content": "hello",
            }
        }
    )
    agent = TeachingAgent(
        settings=Settings(debug_log_path=str(log_path)),
        client=client,
    )

    turn = agent.run_turn("Say hello")

    assert turn.reply == "hello"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4

    first = json.loads(lines[0])
    second = json.loads(lines[1])
    third = json.loads(lines[2])
    fourth = json.loads(lines[3])

    assert first["event"] == "user_input"
    assert first["payload"]["user_input"] == "Say hello"
    assert isinstance(first["payload"]["session_id"], str)
    assert first["payload"]["turn_id"] == 1
    assert second["event"] == "ollama_request"
    assert second["payload"]["model"] == agent.settings.ollama_model
    assert third["event"] == "ollama_response"
    assert third["payload"]["message"]["content"] == "hello"
    assert fourth["event"] == "turn_complete"
    assert fourth["payload"]["reply"] == "hello"
