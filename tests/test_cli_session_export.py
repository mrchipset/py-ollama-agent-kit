from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ollama_agent_kit.cli import app


def test_session_export_command_exports_markdown(tmp_path: Path) -> None:
    log_path = tmp_path / "debug.jsonl"
    output_path = tmp_path / "exports" / "session.md"
    entry = {
        "timestamp": "2026-04-26T10:00:00+00:00",
        "event": "turn_complete",
        "payload": {
            "session_id": "session-x",
            "turn_id": 1,
            "status": "ok",
            "reply": "hello",
            "tool_events": [],
            "rag_hits": [],
        },
    }
    log_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["session", "export", str(log_path), str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    assert "Exported session" in result.output
    assert "hello" in output_path.read_text(encoding="utf-8")
