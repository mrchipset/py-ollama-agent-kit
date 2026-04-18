from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from .agent import TeachingAgent
from .config import get_settings
from .ollama_client import OllamaAPIError, OllamaClient
from .rag import MarkdownRagStore

app = typer.Typer(add_completion=False, help="Teaching-oriented Ollama agent CLI.")
rag_app = typer.Typer(add_completion=False, help="Markdown RAG knowledge base commands.")
app.add_typer(rag_app, name="rag")
console = Console()


def _build_agent(model: str | None = None) -> TeachingAgent:
    settings = get_settings()
    if model:
        settings.ollama_model = model
    return TeachingAgent(settings=settings)


def _build_rag_store() -> MarkdownRagStore:
    settings = get_settings()
    return MarkdownRagStore(
        workspace_root=Path.cwd(),
        index_path=Path(settings.rag_index_path),
        client=OllamaClient(settings.ollama_host),
        embedding_model=settings.rag_embedding_model,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )


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
    rag: bool = typer.Option(True, "--rag/--no-rag", help="Enable or disable automatic Markdown RAG context injection."),
    stream: bool = typer.Option(
        False,
        "--stream/--no-stream",
        help="Stream assistant text to the terminal as it is generated.",
    ),
    debug_log_path: Optional[str] = typer.Option(
        None,
        help="Write user requests and Ollama responses to this JSONL file.",
    ),
) -> None:
    agent = _build_agent(model=model)
    agent.settings.rag_auto_enabled = rag
    if not rag:
        agent.rag_store = None
    if debug_log_path:
        agent.settings.debug_log_path = debug_log_path
    console.print(
        f"Connected to {agent.settings.ollama_host} with model {agent.settings.ollama_model}."
    )

    if prompt:
        _run_single_turn(agent, prompt, stream=stream)
        return

    while True:
        user_input = Prompt.ask("You")
        if user_input.strip().lower() in {"quit", "exit", ":q"}:
            break
        _run_single_turn(agent, user_input, stream=stream)


def _run_single_turn(agent: TeachingAgent, user_input: str, *, stream: bool = False) -> None:
    streamed_reply: list[str] = []
    rag_hits = agent._search_rag(user_input) if stream and hasattr(agent, "_search_rag") else None
    printed_rag_hits = False

    if stream and rag_hits:
        console.print("[cyan]Retrieved references:[/cyan]")
        for index, hit in enumerate(rag_hits, start=1):
            console.print(f"[cyan]{index}. {hit.citation}[/cyan] [dim](score {hit.score:.3f})[/dim]")
            if hit.heading:
                console.print(f"[dim]{hit.heading}[/dim]")
            console.print(hit.excerpt)
        printed_rag_hits = True

    def on_text_chunk(chunk: str) -> None:
        streamed_reply.append(chunk)
        console.print(chunk, end="")

    try:
        if stream and _supports_stream_callback(agent):
            turn = agent.run_turn(user_input, rag_hits=rag_hits, on_text_chunk=on_text_chunk)
        else:
            turn = agent.run_turn(user_input)
    except (OllamaAPIError, RuntimeError, KeyError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if turn.rag_hits and not printed_rag_hits:
        console.print("[cyan]Retrieved references:[/cyan]")
        for index, hit in enumerate(turn.rag_hits, start=1):
            console.print(f"[cyan]{index}. {hit.citation}[/cyan] [dim](score {hit.score:.3f})[/dim]")
            if hit.heading:
                console.print(f"[dim]{hit.heading}[/dim]")
            console.print(hit.excerpt)

    for event in turn.tool_events:
        console.print(f"[yellow]tool[/yellow] {event.name}({event.arguments})")
        console.print(event.result)

    if stream:
        console.print()
    else:
        console.print(Markdown(turn.reply))


def _supports_stream_callback(agent: object) -> bool:
    try:
        signature = inspect.signature(agent.run_turn)
    except (AttributeError, TypeError, ValueError):
        return False

    return "on_text_chunk" in signature.parameters


def main() -> None:
    app()


@rag_app.command("add")
def rag_add(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Markdown file to add."),
) -> None:
    try:
        result = _build_rag_store().add_markdown_file(path)
    except (FileNotFoundError, ValueError, OllamaAPIError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Added {result.chunks_added} chunks from {result.source_path} into {result.total_chunks} total chunks."
    )


@rag_app.command("search")
def rag_search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int | None = typer.Option(None, min=1, help="Maximum number of results to return."),
) -> None:
    store = _build_rag_store()
    try:
        results = store.search(query, top_k=top_k or get_settings().rag_top_k)
    except (ValueError, OllamaAPIError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("No matches found.")
        return

    table = Table(title="RAG search results")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Citation")
    table.add_column("Excerpt")

    for index, result in enumerate(results, start=1):
        table.add_row(
            str(index),
            f"{result.score:.3f}",
            result.citation,
            result.excerpt,
        )

    console.print(table)


@rag_app.command("clear")
def rag_clear() -> None:
    store = _build_rag_store()
    store.clear()
    console.print("RAG index cleared.")
