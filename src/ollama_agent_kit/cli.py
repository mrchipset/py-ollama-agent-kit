from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table

from .agent import ResponseMetrics, TeachingAgent
from .config import get_settings
from .ollama_client import OllamaAPIError, OllamaClient
from .rag import MarkdownRagStore
from .session_export import export_session_log

app = typer.Typer(add_completion=False, help="Teaching-oriented Ollama agent CLI.")
rag_app = typer.Typer(add_completion=False, help="Markdown RAG knowledge base commands.")
session_app = typer.Typer(add_completion=False, help="Session export commands.")
demo_app = typer.Typer(add_completion=False, help="Guided demo scenarios for teaching the agent.")
app.add_typer(rag_app, name="rag")
app.add_typer(session_app, name="session")
app.add_typer(demo_app, name="demo")
console = Console()

DEFAULT_PYTHON_DEMO_PROMPT = "Use Python to add 17 and 25, then show the result."


def _format_generation_rate(metrics: ResponseMetrics) -> str | None:
    if metrics.eval_count is None or metrics.eval_duration is None or metrics.eval_duration <= 0:
        return None

    tokens_per_second = metrics.eval_count / (metrics.eval_duration / 1_000_000_000)
    return f"Generation speed: {tokens_per_second:.2f} token/s"


def _apply_tool_settings(
    settings,
    *,
    tool_modules: str | None = None,
    tool_mode: str | None = None,
    tool_registry_strict: bool | None = None,
    mcp_servers: str | None = None,
) -> None:
    if tool_modules is not None:
        settings.tool_modules = tool_modules
    if tool_mode is not None:
        settings.tool_mode = tool_mode
    if tool_registry_strict is not None:
        settings.tool_registry_strict = tool_registry_strict
    if mcp_servers is not None:
        settings.mcp_servers = mcp_servers


def _build_agent(
    model: str | None = None,
    *,
    tool_modules: str | None = None,
    tool_mode: str | None = None,
    tool_registry_strict: bool | None = None,
    mcp_servers: str | None = None,
) -> TeachingAgent:
    settings = get_settings()
    if model:
        settings.ollama_model = model
    _apply_tool_settings(
        settings,
        tool_modules=tool_modules,
        tool_mode=tool_mode,
        tool_registry_strict=tool_registry_strict,
        mcp_servers=mcp_servers,
    )
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


def _run_python_demo(
    agent: TeachingAgent,
    *,
    prompt: str = DEFAULT_PYTHON_DEMO_PROMPT,
    stream: bool = True,
) -> None:
    console.print("[cyan]Python demo:[/cyan] controlled tool execution with structured output.")
    console.print(f"[cyan]Prompt:[/cyan] {prompt}")
    _run_single_turn(agent, prompt, stream=stream)


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
    images: Optional[list[Path]] = typer.Option(
        None,
        "--image",
        "-i",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Attach one or more images to the next turn.",
    ),
    tool_modules: Optional[str] = typer.Option(
        None,
        help="Comma-separated Python modules that expose build_tools() or TOOLS.",
    ),
    tool_mode: Optional[str] = typer.Option(
        None,
        help="Tool registry mode: builtin, builtin+custom, builtin+mcp, builtin+custom+mcp, custom-only, or mcp-only.",
    ),
    tool_registry_strict: bool = typer.Option(
        True,
        "--strict-tools/--no-strict-tools",
        help="Fail on duplicate tool names instead of skipping them.",
    ),
    mcp_servers: Optional[str] = typer.Option(
        None,
        help="JSON object or array describing MCP stdio servers to load as tools.",
    ),
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
    agent = _build_agent(
        model=model,
        tool_modules=tool_modules,
        tool_mode=tool_mode,
        tool_registry_strict=tool_registry_strict,
        mcp_servers=mcp_servers,
    )
    agent.settings.rag_auto_enabled = rag
    if not rag:
        agent.rag_store = None
    if debug_log_path:
        agent.settings.debug_log_path = debug_log_path
    console.print(
        f"Connected to {agent.settings.ollama_host} with model {agent.settings.ollama_model}."
    )

    if prompt:
        _run_single_turn(agent, prompt, stream=stream, images=images)
        return

    while True:
        user_input = Prompt.ask("You")
        if user_input.strip().lower() in {"quit", "exit", ":q"}:
            break
        inline_images, inline_prompt = _parse_inline_image_turn(user_input)
        if inline_images is not None:
            if not inline_images:
                image_input = Prompt.ask("Image paths")
                inline_images = [Path(part) for part in _split_image_arguments(image_input)]
            if not inline_prompt:
                inline_prompt = Prompt.ask("Question")
            _run_single_turn(agent, inline_prompt, stream=stream, images=inline_images)
            continue

        _run_single_turn(agent, user_input, stream=stream)


def _run_single_turn(
    agent: TeachingAgent,
    user_input: str,
    *,
    stream: bool = False,
    images: Optional[list[Path]] = None,
) -> None:
    streamed_reply: list[str] = []
    use_rag = stream and hasattr(agent, "should_use_rag") and agent.should_use_rag(user_input)
    rag_hits = agent._search_rag(user_input) if use_rag and hasattr(agent, "_search_rag") else None
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

    try:
        if stream and _supports_stream_callback(agent):
            turn = _invoke_turn(agent, user_input, rag_hits=rag_hits, on_text_chunk=on_text_chunk, images=images)
        else:
            turn = _invoke_turn(agent, user_input, images=images)
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
        if streamed_reply:
            console.print("".join(streamed_reply), end="")
        console.print()
    else:
        console.print(Markdown(turn.reply))

    generation_rate = _format_generation_rate(turn.response_metrics)
    if generation_rate is not None:
        console.print(f"[dim]{generation_rate}[/dim]")


def _supports_stream_callback(agent: object) -> bool:
    try:
        signature = inspect.signature(agent.run_turn)
    except (AttributeError, TypeError, ValueError):
        return False

    return "on_text_chunk" in signature.parameters


def _supports_image_callback(agent: object) -> bool:
    try:
        signature = inspect.signature(agent.run_turn)
    except (AttributeError, TypeError, ValueError):
        return False

    return "images" in signature.parameters


def _invoke_turn(
    agent: TeachingAgent,
    user_input: str,
    *,
    rag_hits=None,
    on_text_chunk=None,
    images: Optional[list[Path]] = None,
):
    if images is not None and not _supports_image_callback(agent):
        raise RuntimeError("This agent does not support image attachments.")

    kwargs = {}
    if rag_hits is not None:
        kwargs["rag_hits"] = rag_hits
    if on_text_chunk is not None:
        kwargs["on_text_chunk"] = on_text_chunk
    if images is not None:
        kwargs["images"] = images

    return agent.run_turn(user_input, **kwargs)


def _parse_inline_image_turn(user_input: str) -> tuple[list[Path] | None, str]:
    stripped = user_input.strip()
    if not stripped.startswith((":image", "/image")):
        return None, user_input

    remainder = stripped.split(None, 1)
    if len(remainder) == 1:
        return [], ""

    payload = remainder[1]
    if " -- " in payload:
        image_part, prompt = payload.split(" -- ", 1)
    else:
        image_part, prompt = payload, ""

    return [Path(part) for part in _split_image_arguments(image_part)], prompt.strip()


def _split_image_arguments(image_input: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote_char: str | None = None

    for character in image_input.strip():
        if character in {'"', "'"}:
            if quote_char is None:
                quote_char = character
                continue
            if quote_char == character:
                quote_char = None
                continue
        if character.isspace() and quote_char is None:
            if current:
                parts.append("".join(current))
                current = []
            continue
        current.append(character)

    if current:
        parts.append("".join(current))

    return parts


def main() -> None:
    app()


@session_app.command("export")
def session_export(
    log_path: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="JSONL debug log to export."),
    output_path: Path = typer.Argument(..., help="Destination file for the exported session."),
    session_id: Optional[str] = typer.Option(None, help="Export only the session with this id."),
    format: str = typer.Option("markdown", help="Output format: markdown or json."),
) -> None:
    try:
        export_session_log(log_path, output_path, session_id=session_id, output_format=format)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"Exported session from {log_path} to {output_path}.")


@demo_app.command("python")
def demo_python(
    prompt: Optional[str] = typer.Option(None, help="Override the default Python demo prompt."),
    model: Optional[str] = typer.Option(None, help="Override the model name for this demo."),
    tool_modules: Optional[str] = typer.Option(
        None,
        help="Comma-separated Python modules that expose build_tools() or TOOLS.",
    ),
    tool_mode: Optional[str] = typer.Option(
        None,
        help="Tool registry mode: builtin, builtin+custom, or custom-only.",
    ),
    tool_registry_strict: bool = typer.Option(
        True,
        "--strict-tools/--no-strict-tools",
        help="Fail on duplicate tool names instead of skipping them.",
    ),
    stream: bool = typer.Option(
        True,
        "--stream/--no-stream",
        help="Stream assistant text to the terminal as it is generated.",
    ),
    rag: bool = typer.Option(
        False,
        "--rag/--no-rag",
        help="Enable or disable automatic Markdown RAG context injection for the demo.",
    ),
) -> None:
    agent = _build_agent(
        model=model,
        tool_modules=tool_modules,
        tool_mode=tool_mode,
        tool_registry_strict=tool_registry_strict,
    )
    agent.settings.rag_auto_enabled = rag
    if not rag:
        agent.rag_store = None

    _run_python_demo(agent, prompt=prompt or DEFAULT_PYTHON_DEMO_PROMPT, stream=stream)


@rag_app.command("add")
def rag_add(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Markdown file or directory to add."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recursively import Markdown files when the path is a directory."),
) -> None:
    try:
        store = _build_rag_store()
        if path.is_dir():
            result = store.add_markdown_directory(path, recursive=recursive)
            console.print(
                f"Added {result.sources_added} source files with {result.chunks_added} chunks into {result.total_chunks} total chunks."
            )
            return

        result = store.add_markdown_file(path)
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


@rag_app.command("delete")
def rag_delete(
    path: Path = typer.Argument(..., dir_okay=False, help="Indexed Markdown file to remove."),
) -> None:
    try:
        result = _build_rag_store().delete_source(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Deleted {result.chunks_deleted} chunks from {result.source_path}; {result.total_chunks} chunks remain."
    )


@rag_app.command("rebuild")
def rag_rebuild() -> None:
    store = _build_rag_store()
    result = store.rebuild_index()
    console.print(
        f"Rebuilt embeddings for {result.chunks_rebuilt} chunks across {result.total_chunks} total chunks."
    )


@rag_app.command("refresh")
def rag_refresh() -> None:
    store = _build_rag_store()
    result = store.refresh_index()
    console.print(
        f"Scanned {result.sources_scanned} sources and rebuilt {result.sources_rebuilt} sources with {result.chunks_rebuilt} chunks."
    )
    if result.stale_sources:
        console.print(f"Stale sources: {', '.join(result.stale_sources)}")
    if result.missing_sources:
        console.print(f"Missing sources: {', '.join(result.missing_sources)}")


@rag_app.command("health")
def rag_health() -> None:
    store = _build_rag_store()
    result = store.health_check()

    console.print(f"Healthy: {'yes' if result.healthy else 'no'}")
    console.print(f"Indexed documents: {result.source_count}")
    console.print(f"Indexed chunks: {result.chunk_count}")
    console.print(f"Stale sources: {len(result.stale_sources)}")
    console.print(f"Missing sources: {len(result.missing_sources)}")

    if not result.issues:
        return

    table = Table(title="RAG health issues")
    table.add_column("Severity")
    table.add_column("Code")
    table.add_column("Source")
    table.add_column("Message")

    for issue in result.issues:
        table.add_row(issue.severity, issue.code, issue.source_path or "-", issue.message)

    console.print(table)


@rag_app.command("list")
def rag_list() -> None:
    store = _build_rag_store()
    documents = store.list_documents()

    if not documents:
        console.print("No documents have been indexed yet.")
        return

    table = Table(title="Indexed documents")
    table.add_column("Source path")
    table.add_column("Chunks", justify="right")
    table.add_column("Lines")
    table.add_column("Citation")

    for document in documents:
        table.add_row(
            document["source_path"],
            str(document["chunk_count"]),
            f'L{document["line_start"]}-L{document["line_end"]}',
            document["citation"],
        )

    console.print(table)


@rag_app.command("stats")
def rag_stats() -> None:
    store = _build_rag_store()
    stats = store.stats()

    console.print(f"Indexed documents: {stats.source_count}")
    console.print(f"Indexed chunks: {stats.chunk_count}")
    console.print(f"Embedding model: {stats.embedding_model}")
    console.print(f"Chunk size: {stats.chunk_size}")
    console.print(f"Chunk overlap: {stats.chunk_overlap}")
