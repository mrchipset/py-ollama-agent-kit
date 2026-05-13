from pathlib import Path
import json
import sys
import textwrap

import pytest

from ollama_agent_kit.config import Settings
from ollama_agent_kit.tools import build_default_registry, build_tool_registry, load_mcp_tools


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


def test_custom_tool_module_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "custom_tools.py"
    module_path.write_text(
        textwrap.dedent(
            '''
            from ollama_agent_kit.tools import ToolDefinition


            def build_tools():
                def greet(arguments):
                    return f"hello {arguments['name']}"

                return [
                    ToolDefinition(
                        name="greet",
                        description="Greet a person.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        handler=greet,
                    )
                ]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    settings = Settings(tool_modules="custom_tools", tool_mode="builtin+custom")
    registry = build_tool_registry(Path.cwd(), settings=settings)

    assert registry.has_tool("greet")
    assert registry.execute("greet", {"name": "Ada"}) == "hello Ada"
    assert registry.has_tool("add_numbers")


def test_custom_only_mode_excludes_builtin_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "custom_tools.py"
    module_path.write_text(
        textwrap.dedent(
            '''
            from ollama_agent_kit.tools import ToolDefinition


            def build_tools():
                def greet(arguments):
                    return f"hello {arguments['name']}"

                return [
                    ToolDefinition(
                        name="greet",
                        description="Greet a person.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        handler=greet,
                    )
                ]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    settings = Settings(tool_modules="custom_tools", tool_mode="custom-only")
    registry = build_tool_registry(Path.cwd(), settings=settings)

    assert registry.has_tool("greet")
    assert not registry.has_tool("add_numbers")


def test_duplicate_custom_tool_names_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "custom_tools_duplicate.py"
    module_path.write_text(
        textwrap.dedent(
            '''
            from ollama_agent_kit.tools import ToolDefinition


            def build_tools():
                def duplicate(arguments):
                    return "duplicate"

                return [
                    ToolDefinition(
                        name="add_numbers",
                        description="Duplicate builtin tool.",
                        parameters={
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                        handler=duplicate,
                    )
                ]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__("sys").modules, "custom_tools_duplicate", raising=False)

    settings = Settings(tool_modules="custom_tools_duplicate", tool_mode="builtin+custom", tool_registry_strict=True)

    with pytest.raises(ValueError, match="Duplicate tool name: add_numbers"):
        build_tool_registry(Path.cwd(), settings=settings)


def test_load_mcp_tools_builds_prefixed_tool_definitions() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeMcpClient:
        def __init__(self, config, timeout_seconds) -> None:
            self.config = config
            self.timeout_seconds = timeout_seconds

        def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "search_docs",
                    "description": "Search docs over MCP.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                }
            ]

        def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            calls.append((name, arguments))
            return {
                "content": [
                    {"type": "text", "text": f"matched: {arguments['query']}"},
                ]
            }

    settings = Settings(
        mcp_servers='{"docs":{"command":"fake-mcp","args":["serve"]}}',
        mcp_timeout_seconds=3,
    )

    tools = load_mcp_tools(settings, client_factory=FakeMcpClient)

    assert [tool.name for tool in tools] == ["docs__search_docs"]
    assert tools[0].handler({"query": "rag"}) == "matched: rag"
    assert calls == [("search_docs", {"query": "rag"})]


def test_build_tool_registry_can_combine_builtin_custom_and_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "custom_tools.py"
    module_path.write_text(
        textwrap.dedent(
            '''
            from ollama_agent_kit.tools import ToolDefinition


            def build_tools():
                def greet(arguments):
                    return f"hello {arguments['name']}"

                return [
                    ToolDefinition(
                        name="greet",
                        description="Greet a person.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                        handler=greet,
                    )
                ]
            '''
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    class FakeMcpClient:
        def __init__(self, config, timeout_seconds) -> None:
            self.config = config

        def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "search_docs",
                    "description": "Search docs over MCP.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                }
            ]

        def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr("ollama_agent_kit.tools.StdioMcpClient", FakeMcpClient)

    settings = Settings(
        tool_modules="custom_tools",
        tool_mode="builtin+custom+mcp",
        mcp_servers='{"docs":{"command":"fake-mcp","args":[]}}',
    )
    registry = build_tool_registry(Path.cwd(), settings=settings)

    assert registry.has_tool("add_numbers")
    assert registry.has_tool("greet")
    assert registry.has_tool("docs__search_docs")


def test_mcp_only_mode_excludes_builtin_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMcpClient:
        def __init__(self, config, timeout_seconds) -> None:
            self.config = config

        def list_tools(self) -> list[dict[str, object]]:
            return [
                {
                    "name": "search_docs",
                    "description": "Search docs over MCP.",
                    "inputSchema": {"type": "object", "properties": {}, "required": []},
                }
            ]

        def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
            return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr("ollama_agent_kit.tools.StdioMcpClient", FakeMcpClient)

    settings = Settings(
        tool_mode="mcp-only",
        mcp_servers='{"docs":{"command":"fake-mcp","args":[]}}',
    )
    registry = build_tool_registry(Path.cwd(), settings=settings)

    assert registry.has_tool("docs__search_docs")
    assert not registry.has_tool("add_numbers")


def test_real_stdio_mcp_server_can_be_loaded_and_called() -> None:
    settings = Settings(
        tool_mode="mcp-only",
        mcp_servers=json.dumps(
            {
                "test": {
                    "command": sys.executable,
                    "args": ["-m", "ollama_agent_kit.mcp_test_server"],
                }
            }
        ),
        mcp_timeout_seconds=3,
    )

    registry = build_tool_registry(Path.cwd(), settings=settings)

    assert registry.has_tool("test__echo_text")
    assert registry.has_tool("test__sum_numbers")
    assert registry.execute("test__echo_text", {"text": "hello mcp"}) == "Echo: hello mcp"
    assert registry.execute("test__sum_numbers", {"a": 2, "b": 5}) == "7.0"
