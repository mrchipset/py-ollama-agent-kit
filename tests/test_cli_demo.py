from types import SimpleNamespace

from typer.testing import CliRunner

from ollama_agent_kit.agent import AgentTurn
from ollama_agent_kit.cli import DEFAULT_PYTHON_DEMO_PROMPT, app
from ollama_agent_kit.tools import ToolExecution


def test_demo_python_command_uses_default_prompt(monkeypatch) -> None:
    calls: list[tuple[object, str, bool]] = []

    def fake_build_agent(model=None):
        return SimpleNamespace(settings=SimpleNamespace(rag_auto_enabled=True), rag_store=object())

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

    def fake_build_agent(model=None):
        return SimpleNamespace(settings=SimpleNamespace(rag_auto_enabled=True), rag_store=object())

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