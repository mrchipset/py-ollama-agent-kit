from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SessionTurnExport:
    turn_id: int
    user_input: str = ""
    request: dict[str, Any] | None = None
    responses: list[dict[str, Any]] = field(default_factory=list)
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    rag_hits: list[dict[str, Any]] = field(default_factory=list)
    reply: str = ""
    status: str = "unknown"
    recovery_note: str | None = None


@dataclass(slots=True)
class SessionExport:
    session_id: str
    turns: list[SessionTurnExport]


def load_session_exports(log_path: Path, session_id: str | None = None) -> list[SessionExport]:
    entries = _load_jsonl_entries(log_path)
    if not entries:
        raise ValueError(f"No debug log entries found in {log_path}.")

    grouped: dict[str, dict[int, SessionTurnExport]] = defaultdict(dict)
    session_order: list[str] = []

    for entry in entries:
        payload = entry.get("payload") or {}
        current_session_id = payload.get("session_id") or "legacy"
        if current_session_id not in grouped:
            session_order.append(current_session_id)
        turn_id = payload.get("turn_id")
        if not isinstance(turn_id, int):
            continue

        turn = grouped[current_session_id].setdefault(turn_id, SessionTurnExport(turn_id=turn_id))
        event = entry.get("event")
        if event == "user_input":
            turn.user_input = str(payload.get("user_input", ""))
            turn.rag_hits = list(payload.get("rag_hits") or [])
        elif event == "ollama_request":
            turn.request = payload
        elif event == "ollama_response":
            turn.responses.append(payload)
        elif event == "tool_execution":
            tool = payload.get("tool")
            if isinstance(tool, dict):
                turn.tool_events.append(tool)
        elif event == "turn_complete":
            turn.status = str(payload.get("status", "unknown"))
            turn.reply = str(payload.get("reply", ""))
            turn.tool_events = list(payload.get("tool_events") or turn.tool_events)
            turn.rag_hits = list(payload.get("rag_hits") or turn.rag_hits)
            recovered_from = payload.get("recovered_from")
            if isinstance(recovered_from, str):
                turn.recovery_note = recovered_from

    if session_id is None:
        selected_session_id = session_order[-1]
        return [_build_export(selected_session_id, grouped[selected_session_id])]

    if session_id not in grouped:
        raise ValueError(f"Session id {session_id!r} was not found in {log_path}.")

    return [_build_export(session_id, grouped[session_id])]


def export_session_log(
    log_path: Path,
    output_path: Path,
    *,
    session_id: str | None = None,
    output_format: str = "markdown",
) -> SessionExport:
    session_exports = load_session_exports(log_path, session_id=session_id)
    if len(session_exports) != 1:
        raise ValueError("Exactly one session export is expected.")

    session_export = session_exports[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_format = output_format.casefold()
    if normalized_format == "markdown":
        output_path.write_text(render_markdown(session_export), encoding="utf-8")
    elif normalized_format == "json":
        output_path.write_text(json.dumps(render_json(session_export), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError("Output format must be 'markdown' or 'json'.")

    return session_export


def render_json(session_export: SessionExport) -> dict[str, Any]:
    return {
        "session_id": session_export.session_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "user_input": turn.user_input,
                "status": turn.status,
                "reply": turn.reply,
                "recovery_note": turn.recovery_note,
                "rag_hits": turn.rag_hits,
                "tool_events": turn.tool_events,
                "request": turn.request,
                "responses": turn.responses,
            }
            for turn in session_export.turns
        ],
    }


def render_markdown(session_export: SessionExport) -> str:
    lines: list[str] = [
        "# Session Export",
        "",
        f"- session id: {session_export.session_id}",
        f"- exported at: {datetime.now().astimezone().isoformat()}",
        f"- turns: {len(session_export.turns)}",
        "",
    ]

    for turn in session_export.turns:
        lines.extend(
            [
                f"## Turn {turn.turn_id}",
                "",
                "### User Input",
                turn.user_input or "",
                "",
                f"- status: {turn.status}",
            ]
        )
        if turn.recovery_note:
            lines.append(f"- recovery: {turn.recovery_note}")
        if turn.reply:
            lines.extend(["", "### Final Reply", turn.reply, ""])
        if turn.rag_hits:
            lines.extend(["### RAG Hits", ""])
            for hit in turn.rag_hits:
                citation = hit.get("citation", "unknown")
                score = hit.get("score", "")
                excerpt = hit.get("excerpt", "")
                lines.append(f"- {citation} (score {score})")
                if excerpt:
                    lines.append(f"  - {excerpt}")
            lines.append("")
        if turn.tool_events:
            lines.extend(["### Tool Calls", ""])
            for event in turn.tool_events:
                lines.append(f"- {event.get('name', 'unknown')}")
                arguments = event.get("arguments")
                result = event.get("result")
                if arguments is not None:
                    lines.append(f"  - arguments: {json.dumps(arguments, ensure_ascii=False)}")
                if result is not None:
                    lines.append(f"  - result: {result}")
            lines.append("")
        if turn.request is not None:
            lines.extend(["### Request Snapshot", ""]) 
            lines.append("```json")
            lines.append(json.dumps(turn.request, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
        if turn.responses:
            lines.extend(["### Raw Responses", ""])
            for response in turn.responses:
                lines.append("```json")
                lines.append(json.dumps(response, ensure_ascii=False, indent=2))
                lines.append("```")
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _load_jsonl_entries(log_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        entries.append(json.loads(stripped))
    return entries


def _build_export(session_id: str, turns_by_id: dict[int, SessionTurnExport]) -> SessionExport:
    ordered_turns = [turns_by_id[turn_id] for turn_id in sorted(turns_by_id)]
    return SessionExport(session_id=session_id, turns=ordered_turns)