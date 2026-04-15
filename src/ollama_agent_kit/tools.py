from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import ROOT_DIR

ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: str


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.as_ollama_tool() for tool in self._tools.values()]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> ToolExecution:
        function_payload = tool_call.get("function", tool_call)
        name = function_payload["name"]
        arguments = self._coerce_arguments(function_payload.get("arguments", {}))
        result = self.execute(name, arguments)
        return ToolExecution(name=name, arguments=arguments, result=result)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(arguments)

    @staticmethod
    def _coerce_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            if not raw_arguments.strip():
                return {}
            return json.loads(raw_arguments)
        raise TypeError(f"Unsupported tool arguments: {type(raw_arguments)!r}")


def build_default_registry(workspace_root: Path | None = None) -> ToolRegistry:
    root = (workspace_root or ROOT_DIR).resolve()

    def get_current_time(_: dict[str, Any]) -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_numbers(arguments: dict[str, Any]) -> str:
        left = float(arguments["a"])
        right = float(arguments["b"])
        return json.dumps({"sum": left + right})

    def list_workspace(arguments: dict[str, Any]) -> str:
        relative_path = arguments.get("path", ".")
        limit = int(arguments.get("limit", 20))
        target = _resolve_workspace_path(root, relative_path)

        if target.is_file():
            return json.dumps({"path": str(target.relative_to(root)), "type": "file"}, indent=2)

        entries = []
        for child in sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))[:limit]:
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(root)),
                    "type": "dir" if child.is_dir() else "file",
                }
            )
        return json.dumps({"path": str(target.relative_to(root)) or ".", "entries": entries}, indent=2)

    def read_workspace_file(arguments: dict[str, Any]) -> str:
        relative_path = arguments["path"]
        max_chars = int(arguments.get("max_chars", 2000))
        target = _resolve_workspace_path(root, relative_path)

        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {relative_path}")

        content = target.read_text(encoding="utf-8")
        return content[:max_chars]

    tools = [
        ToolDefinition(
            name="get_current_time",
            description="Return the current UTC time in ISO 8601 format.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=get_current_time,
        ),
        ToolDefinition(
            name="add_numbers",
            description="Add two numbers and return the sum.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number."},
                    "b": {"type": "number", "description": "Second number."},
                },
                "required": ["a", "b"],
            },
            handler=add_numbers,
        ),
        ToolDefinition(
            name="list_workspace",
            description="List files or directories inside the current workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace."},
                    "limit": {"type": "integer", "description": "Maximum number of entries to return."},
                },
                "required": [],
            },
            handler=list_workspace,
        ),
        ToolDefinition(
            name="read_workspace_file",
            description="Read a UTF-8 text file from the current workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path inside the workspace."},
                    "max_chars": {"type": "integer", "description": "Maximum number of characters to return."},
                },
                "required": ["path"],
            },
            handler=read_workspace_file,
        ),
    ]
    return ToolRegistry(tools)


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = (workspace_root / raw_path).resolve()
    candidate.relative_to(workspace_root)
    return candidate
