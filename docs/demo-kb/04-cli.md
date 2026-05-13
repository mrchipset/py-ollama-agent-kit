# CLI Usage

The CLI is meant to be easy to demonstrate in a terminal.

Common commands:

- `ollama-agent doctor` — verify the Ollama host and model
- `ollama-agent chat` — start an interactive chat session
- `ollama-agent chat --no-rag` — run chat without automatic retrieval
- `ollama-agent chat --tool-mode builtin+mcp` — run chat with built-in tools plus MCP tools
- `ollama-agent demo python` — run the guided controlled Python execution demo
- `ollama-agent rag add <markdown-file>` — add a Markdown file to the knowledge base
- `ollama-agent rag search <query>` — retrieve matching chunks with citations
- `ollama-agent rag clear` — reset the local RAG index

The CLI prints retrieved references before the final assistant answer when automatic RAG is enabled. That makes it easier to show the boundary between retrieval and generation.

The sample environment is configured to use:

- `OLLAMA_HOST=http://192.168.71.21:11434`
- `OLLAMA_MODEL=qwen2.5-coder:1.5b`
- `RAG_INDEX_PATH=data/rag_index.json`
- `OLLAMA_TOOL_MODE=builtin`
- `OLLAMA_TOOL_MODULES=`
- `OLLAMA_MCP_SERVERS=`
- `OLLAMA_MCP_TIMEOUT_SECONDS=15`
- `OLLAMA_TOOL_REGISTRY_STRICT=true`

The embedding model can be configured separately so the knowledge base can remain stable even if the chat model changes.

## MCP CLI Notes

The CLI can combine three tool sources:

- built-in tools
- custom Python tool modules
- MCP tools loaded from stdio MCP servers

Typical modes:

- `builtin`
- `builtin+custom`
- `builtin+mcp`
- `builtin+custom+mcp`
- `custom-only`
- `mcp-only`

Example MCP session:

```bash
ollama-agent chat --no-rag --stream --tool-mode mcp-only
```

If you want to use the included test MCP server from this repository, the most reliable approach on Windows is to set `OLLAMA_MCP_SERVERS` through `.env` or a VS Code debug configuration instead of typing raw JSON directly in PowerShell.