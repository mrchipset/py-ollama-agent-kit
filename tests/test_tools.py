from pathlib import Path

import pytest

from ollama_agent_kit.tools import build_default_registry


def test_add_numbers_tool_parses_json_arguments() -> None:
    registry = build_default_registry(Path.cwd())
    execution = registry.execute_tool_call(
        {
            "function": {
                "name": "add_numbers",
                "arguments": '{"a": 2, "b": 5}',
            }
        }
    )

    assert execution.result == '{"sum": 7.0}'


def test_workspace_tools_cannot_escape_root(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)

    with pytest.raises(ValueError):
        registry.execute("list_workspace", {"path": "../outside"})
