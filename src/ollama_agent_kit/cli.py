from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from .agent import TeachingAgent
from .config import get_settings
from .ollama_client import OllamaAPIError, OllamaClient

app = typer.Typer(add_completion=False, help="Teaching-oriented Ollama agent CLI.")
console = Console()


def _build_agent(model: str | None = None) -> TeachingAgent:
    settings = get_settings()
    if model:
        settings.ollama_model = model
    return TeachingAgent(settings=settings)


@app.command()
def doctor() -> None:
    settings = get_settings()
    client = OllamaClient(settings.ollama_host)

    try:
        models = client.list_models()
    except OllamaAPIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Host: {settings.ollama_host}")
    console.print(f"Default model: {settings.ollama_model}")
    console.print(f"Available models: {len(models)}")


@app.command()
def models() -> None:
    settings = get_settings()
    client = OllamaClient(settings.ollama_host)

    try:
        available_models = client.list_models()
    except OllamaAPIError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Ollama models")
    table.add_column("Name")
    table.add_column("Size", justify="right")

    for model in available_models:
        size = model.get("size", "")
        table.add_row(model.get("name", "unknown"), str(size))

    console.print(table)


@app.command()
def chat(
    prompt: Optional[str] = typer.Argument(None, help="Optional one-shot prompt."),
    model: Optional[str] = typer.Option(None, help="Override the model name for this session."),
) -> None:
    agent = _build_agent(model=model)
    console.print(
        f"Connected to {agent.settings.ollama_host} with model {agent.settings.ollama_model}."
    )

    if prompt:
        _run_single_turn(agent, prompt)
        return

    while True:
        user_input = Prompt.ask("You")
        if user_input.strip().lower() in {"quit", "exit", ":q"}:
            break
        _run_single_turn(agent, user_input)


def _run_single_turn(agent: TeachingAgent, user_input: str) -> None:
    try:
        turn = agent.run_turn(user_input)
    except (OllamaAPIError, RuntimeError, KeyError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    for event in turn.tool_events:
        console.print(f"[yellow]tool[/yellow] {event.name}({event.arguments})")
        console.print(event.result)

    console.print(Markdown(turn.reply))


def main() -> None:
    app()
