# Demo Knowledge Base

This folder is a virtual knowledge base for demo and retrieval testing.

The corpus is intentionally small and structured so you can demonstrate:
- local Markdown ingestion
- retrieval with citations
- answer generation from retrieved context
- explicit incremental updates

## Corpus map

- `01-overview.md` — what the project does
- `02-chat-and-tools.md` — chat loop, tool calls, and debug logs
- `03-rag.md` — Markdown RAG workflow and indexing behavior
- `04-cli.md` — CLI commands and usage patterns
- `05-troubleshooting.md` — common issues and fixes
- `06-sample-questions.md` — example queries for demos

## Suggested demo flow

1. Add the files one by one with `ollama-agent rag add`.
2. Ask a question about the project.
3. Show the retrieved citations before the final answer.
4. Toggle RAG off with `--no-rag` to compare behavior.

## Notes

This corpus is virtual. It is not pulled from external sources and should be treated as a teaching dataset.