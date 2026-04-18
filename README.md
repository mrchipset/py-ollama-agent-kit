# Ollama Agent Kit

This workspace contains a teaching-oriented Python agent that talks to Ollama, can execute local tools, and exposes a simple CLI chat loop.

## Project layout

```text
.
|-- .env.example
|-- pyproject.toml
|-- README.md
|-- src/
|   `-- ollama_agent_kit/
|       |-- __init__.py
|       |-- __main__.py
|       |-- agent.py
|       |-- cli.py
|       |-- config.py
|       |-- ollama_client.py
|       |-- rag.py
|       `-- tools.py
`-- tests/
    `-- test_tools.py
```

## Quick start

1. Make sure Ollama is installed and running locally.
2. Pull at least one model, for example `ollama pull llama3.2`.
3. Copy `.env.example` to `.env` if you want to override host or model.
4. Install the project in editable mode:

```bash
pip install -e .
```

5. Check the Ollama connection:

```bash
ollama-agent doctor
```

6. Start the interactive chat:

```bash
ollama-agent chat
```

To write each user prompt and raw Ollama response to a JSONL file for debugging, set `OLLAMA_DEBUG_LOG_PATH` in `.env` or pass `--debug-log-path` to `chat`:

```bash
ollama-agent chat --debug-log-path logs/ollama-debug.jsonl
```

You can disable automatic RAG injection when you want a pure chat session:

```bash
ollama-agent chat --no-rag
```

## Markdown RAG MVP

This workspace now includes a minimal Markdown RAG flow for teaching and debugging.

```bash
ollama-agent rag add docs/intro.md
ollama-agent rag search "what does the project do?"
ollama-agent rag clear
```

The first version only supports local Markdown files, explicit file adds, and retrieval output with citations. Model-generated answers will come in a later step.

## What this sample covers

- Plain HTTP calls to the Ollama chat API
- Tool schema registration
- Tool-call execution loop
- Interactive CLI chat for teaching and debugging
- Optional JSONL debug logging of requests and responses
- Markdown retrieval with explicit add/search/clear commands
- Automatic RAG injection in chat, with a `--no-rag` override

## Built-in tools

- `get_current_time`
- `add_numbers`
- `list_workspace`
- `read_workspace_file`

The file tools are limited to the current workspace root.
