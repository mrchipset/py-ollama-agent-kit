from types import SimpleNamespace
from pathlib import Path

from typer.testing import CliRunner

from ollama_agent_kit.agent import AgentTurn, ResponseMetrics
from ollama_agent_kit.cli import DEFAULT_PYTHON_DEMO_PROMPT, app
from ollama_agent_kit.tools import ToolExecution


def test_demo_python_command_uses_default_prompt(monkeypatch) -> None:
    calls: list[tuple[object, str, bool]] = []

    def fake_build_agent(model=None, **kwargs):
        return SimpleNamespace(
            settings=SimpleNamespace(rag_auto_enabled=True, ollama_host="http://127.0.0.1:11434", ollama_model="llama3.2"),
            rag_store=object(),
        )

    def fake_run_single_turn(agent, user_input: str, *, stream: bool = False):
        calls.append((agent, user_input, stream))

    monkeypatch.setattr("ollama_agent_kit.cli._build_agent", fake_build_agent)
    monkeypatch.setattr("ollama_agent_kit.cli._run_single_turn", fake_run_single_turn)

    result = CliRunner().invoke(app, ["demo", "python"])

    assert result.exit_code == 0
    assert "Python demo:" in result.output
    assert len(calls) == 1
    assert calls[0][1] == DEFAULT_PYTHON_DEMO_PROMPT
    assert calls[0][2] is True


def test_demo_python_command_accepts_prompt_override(monkeypatch) -> None:
    calls: list[tuple[object, str, bool]] = []

    def fake_build_agent(model=None, **kwargs):
        return SimpleNamespace(
            settings=SimpleNamespace(rag_auto_enabled=True, ollama_host="http://127.0.0.1:11434", ollama_model="llama3.2"),
            rag_store=object(),
        )

    def fake_run_single_turn(agent, user_input: str, *, stream: bool = False):
        calls.append((agent, user_input, stream))

    monkeypatch.setattr("ollama_agent_kit.cli._build_agent", fake_build_agent)
    monkeypatch.setattr("ollama_agent_kit.cli._run_single_turn", fake_run_single_turn)

    result = CliRunner().invoke(app, ["demo", "python", "--prompt", "Use Python to add 1 and 2.", "--no-stream"])

    assert result.exit_code == 0
    assert calls[0][1] == "Use Python to add 1 and 2."
    assert calls[0][2] is False


def test_run_single_turn_prints_tool_output_before_streamed_text(capsys) -> None:
    from ollama_agent_kit.cli import _run_single_turn

    class _ToolStreamingAgent:
        def run_turn(self, user_input: str, *, rag_hits=None, on_text_chunk=None) -> AgentTurn:
            if on_text_chunk is not None:
                on_text_chunk("Final answer.")
            return AgentTurn(
                reply="Final answer.",
                tool_events=[
                    ToolExecution(
                        name="run_python_code",
                        arguments={"code": "print('hello')"},
                        result='{"ok": true, "stdout": "hello\\n"}',
                    )
                ],
            )

    _run_single_turn(_ToolStreamingAgent(), "Use Python to add 1 and 2.", stream=True)

    output = capsys.readouterr().out
    assert output.index("tool") < output.index("Final answer.")


def test_run_single_turn_prints_generation_rate_when_metrics_exist(capsys) -> None:
    from ollama_agent_kit.cli import _run_single_turn

    class _StreamingAgentWithMetrics:
        def run_turn(self, user_input: str, *, rag_hits=None, on_text_chunk=None) -> AgentTurn:
            if on_text_chunk is not None:
                on_text_chunk("Final answer.")
            return AgentTurn(
                reply="Final answer.",
                tool_events=[],
                response_metrics=ResponseMetrics(eval_count=12, eval_duration=2_000_000_000),
            )

    _run_single_turn(_StreamingAgentWithMetrics(), "Use Python to add 1 and 2.", stream=True)

    output = capsys.readouterr().out
    assert "Generation speed: 6.00 token/s" in output


def test_chat_command_passes_custom_tool_options(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_agent(model=None, *, tool_modules=None, tool_mode=None, tool_registry_strict=None):
        captured["model"] = model
        captured["tool_modules"] = tool_modules
        captured["tool_mode"] = tool_mode
        captured["tool_registry_strict"] = tool_registry_strict
        return SimpleNamespace(
            settings=SimpleNamespace(rag_auto_enabled=True, ollama_host="http://127.0.0.1:11434", ollama_model="llama3.2"),
            rag_store=object(),
            should_use_rag=lambda user_input: False,
            _search_rag=lambda user_input: [],
            run_turn=lambda user_input, rag_hits=None, on_text_chunk=None: AgentTurn(reply="ok", tool_events=[]),
        )

    monkeypatch.setattr("ollama_agent_kit.cli._build_agent", fake_build_agent)

    result = CliRunner().invoke(
        app,
        [
            "chat",
            "hello",
            "--tool-modules",
            "demo_tools",
            "--tool-mode",
            "builtin+custom",
            "--no-strict-tools",
            "--no-rag",
        ],
    )

    assert result.exit_code == 0
    assert captured["tool_modules"] == "demo_tools"
    assert captured["tool_mode"] == "builtin+custom"
    assert captured["tool_registry_strict"] is False


def test_chat_command_passes_image_option(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image-bytes")

    def fake_build_agent(model=None, **kwargs):
        return SimpleNamespace(
            settings=SimpleNamespace(rag_auto_enabled=True, ollama_host="http://127.0.0.1:11434", ollama_model="llama3.2"),
            rag_store=object(),
            should_use_rag=lambda user_input: False,
            _search_rag=lambda user_input: [],
            run_turn=lambda user_input, rag_hits=None, on_text_chunk=None, images=None: (
                captured.update({"user_input": user_input, "images": images})
                or AgentTurn(reply="ok", tool_events=[])
            ),
        )

    monkeypatch.setattr("ollama_agent_kit.cli._build_agent", fake_build_agent)

    result = CliRunner().invoke(
        app,
        ["chat", "hello", "--no-rag", "--image", str(image_path)],
    )

    assert result.exit_code == 0
    assert captured["user_input"] == "hello"
    assert captured["images"] == [image_path]


def test_chat_command_parses_inline_image_turn(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"image-bytes")

    def fake_build_agent(model=None, **kwargs):
        return SimpleNamespace(
            settings=SimpleNamespace(rag_auto_enabled=True, ollama_host="http://127.0.0.1:11434", ollama_model="llama3.2"),
            rag_store=object(),
            should_use_rag=lambda user_input: False,
            _search_rag=lambda user_input: [],
            run_turn=lambda user_input, rag_hits=None, on_text_chunk=None, images=None: (
                calls.append({"user_input": user_input, "images": images})
                or AgentTurn(reply="ok", tool_events=[])
            ),
        )

    monkeypatch.setattr("ollama_agent_kit.cli._build_agent", fake_build_agent)

    result = CliRunner().invoke(
        app,
        ["chat", "--no-rag"],
        input=f":image {image_path} -- 这张图里有什么？\nexit\n",
    )

    assert result.exit_code == 0
    assert calls[0]["user_input"] == "这张图里有什么？"
    assert calls[0]["images"] == [image_path]