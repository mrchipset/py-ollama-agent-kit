from pathlib import Path
import json

import pytest

from ollama_agent_kit.config import Settings
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


def test_python_tool_returns_structured_output_and_result() -> None:
    registry = build_default_registry(Path.cwd())
    execution = registry.execute_tool_call(
        {
            "function": {
                "name": "run_python_code",
                "arguments": json.dumps({
                    "code": "print('hello')\nresult = 2 + 3",
                }),
            }
        }
    )

    payload = json.loads(execution.result)

    assert payload["ok"] is True
    assert payload["stdout"] == "hello\n"
    assert payload["result"] == "5"
    assert payload["result_type"] == "int"
    assert payload["timed_out"] is False


def test_python_tool_rejects_disallowed_imports() -> None:
    registry = build_default_registry(Path.cwd())
    execution = registry.execute_tool_call(
        {
            "function": {
                "name": "run_python_code",
                "arguments": json.dumps({
                    "code": "import os\nresult = os.getcwd()",
                }),
            }
        }
    )

    payload = json.loads(execution.result)

    assert payload["ok"] is False
    assert payload["error_type"] == "ValidationError"
    assert "Import not allowed: os" in payload["validation_error"]


def test_python_tool_times_out() -> None:
    settings = Settings(python_exec_timeout_seconds=0.5)
    registry = build_default_registry(Path.cwd(), settings=settings)
    execution = registry.execute_tool_call(
        {
            "function": {
                "name": "run_python_code",
                "arguments": json.dumps({
                    "code": "while True:\n    pass",
                }),
            }
        }
    )

    payload = json.loads(execution.result)

    assert payload["ok"] is False
    assert payload["timed_out"] is True
    assert payload["error_type"] == "TimeoutError"


def test_workspace_tools_cannot_escape_root(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)

    with pytest.raises(ValueError):
        registry.execute("list_workspace", {"path": "../outside"})
