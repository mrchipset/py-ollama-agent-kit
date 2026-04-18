# CLI Usage

The CLI is meant to be easy to demonstrate in a terminal.

Common commands:

- `ollama-agent doctor` — verify the Ollama host and model
- `ollama-agent chat` — start an interactive chat session
- `ollama-agent chat --no-rag` — run chat without automatic retrieval
- `ollama-agent rag add <markdown-file>` — add a Markdown file to the knowledge base
- `ollama-agent rag search <query>` — retrieve matching chunks with citations
- `ollama-agent rag clear` — reset the local RAG index

The CLI prints retrieved references before the final assistant answer when automatic RAG is enabled. That makes it easier to show the boundary between retrieval and generation.

The sample environment is configured to use:

- `OLLAMA_HOST=http://192.168.71.21:11434`
- `OLLAMA_MODEL=qwen2.5-coder:1.5b`
- `RAG_INDEX_PATH=data/rag_index.json`

The embedding model can be configured separately so the knowledge base can remain stable even if the chat model changes.