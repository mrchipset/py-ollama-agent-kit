from __future__ import annotations

import json
from pathlib import Path

from ollama_agent_kit.session_export import export_session_log, load_session_exports


def _write_log(path: Path) -> None:
    entries = [
        {
            "timestamp": "2026-04-26T10:00:00+00:00",
            "event": "user_input",
            "payload": {
                "session_id": "session-a",
                "turn_id": 1,
                "user_input": "Say hello",
                "rag_hit_count": 0,
                "rag_hits": [],
            },
        },
        {
            "timestamp": "2026-04-26T10:00:01+00:00",
            "event": "ollama_request",
            "payload": {
                "session_id": "session-a",
                "turn_id": 1,
                "model": "llama3.2",
                "messages": [],
                "tools": [],
            },
        },
        {
            "timestamp": "2026-04-26T10:00:02+00:00",
            "event": "ollama_response",
            "payload": {
                "session_id": "session-a",
                "turn_id": 1,
                "message": {"role": "assistant", "content": "hello"},
            },
        },
        {
            "timestamp": "2026-04-26T10:00:03+00:00",
            "event": "turn_complete",
            "payload": {
                "session_id": "session-a",
                "turn_id": 1,
                "status": "ok",
                "reply": "hello",
                "tool_events": [],
                "rag_hits": [],
            },
        },
        {
            "timestamp": "2026-04-26T10:05:00+00:00",
            "event": "user_input",
            "payload": {
                "session_id": "session-a",
                "turn_id": 2,
                "user_input": "Use a tool",
                "rag_hit_count": 1,
                "rag_hits": [
                    {
                        "score": 0.9,
                        "citation": "docs/demo.md#L1-L4",
                        "source_path": "docs/demo.md",
                        "heading": "Intro",
                        "heading_line": 1,
                        "line_start": 1,
                        "line_end": 4,
                        "excerpt": "A demo chunk.",
                        "text": "A demo chunk.",
                    }
                ],
            },
        },
        {
            "timestamp": "2026-04-26T10:05:01+00:00",
            "event": "tool_execution",
            "payload": {
                "session_id": "session-a",
                "turn_id": 2,
                "tool": {
                    "name": "add_numbers",
                    "arguments": {"a": 1, "b": 2},
                    "result": "3",
                },
            },
        },
        {
            "timestamp": "2026-04-26T10:05:02+00:00",
            "event": "turn_complete",
            "payload": {
                "session_id": "session-a",
                "turn_id": 2,
                "status": "ok",
                "reply": "3",
                "tool_events": [
                    {
                        "name": "add_numbers",
                        "arguments": {"a": 1, "b": 2},
                        "result": "3",
                    }
                ],
                "rag_hits": [
                    {
                        "score": 0.9,
                        "citation": "docs/demo.md#L1-L4",
                        "source_path": "docs/demo.md",
                        "heading": "Intro",
                        "heading_line": 1,
                        "line_start": 1,
                        "line_end": 4,
                        "excerpt": "A demo chunk.",
                        "text": "A demo chunk.",
                    }
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def test_load_session_exports_groups_turns(tmp_path: Path) -> None:
    log_path = tmp_path / "debug.jsonl"
    _write_log(log_path)

    sessions = load_session_exports(log_path)

    assert len(sessions) == 1
    assert sessions[0].session_id == "session-a"
    assert [turn.turn_id for turn in sessions[0].turns] == [1, 2]
    assert sessions[0].turns[1].tool_events[0]["name"] == "add_numbers"


def test_export_session_log_writes_markdown_and_json(tmp_path: Path) -> None:
    log_path = tmp_path / "debug.jsonl"
    markdown_path = tmp_path / "session.md"
    json_path = tmp_path / "session.json"
    _write_log(log_path)

    export_session_log(log_path, markdown_path)
    export_session_log(log_path, json_path, output_format="json")

    markdown = markdown_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert "# Session Export" in markdown
    assert "## Turn 2" in markdown
    assert "add_numbers" in markdown
    assert data["session_id"] == "session-a"
    assert data["turns"][1]["reply"] == "3"
    assert data["turns"][1]["tool_events"][0]["name"] == "add_numbers"
