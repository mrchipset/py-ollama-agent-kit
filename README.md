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

## What this sample covers

- Plain HTTP calls to the Ollama chat API
- Tool schema registration
- Tool-call execution loop
- Interactive CLI chat for teaching and debugging

## Built-in tools

- `get_current_time`
- `add_numbers`
- `list_workspace`
- `read_workspace_file`

The file tools are limited to the current workspace root.
