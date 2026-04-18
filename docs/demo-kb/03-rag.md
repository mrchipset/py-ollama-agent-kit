# Markdown RAG Workflow

The demo knowledge base supports a lightweight Markdown RAG flow.

The first version is deliberately narrow:

- input format: local Markdown only
- update style: explicit file add, not directory scanning
- retrieval output: citations first, then model-generated answer
- storage: a local JSON index file

When a Markdown file is added, it is split into chunks using headings and paragraphs. Each chunk is embedded with Ollama and stored with metadata such as:

- source path
- section heading
- line range
- chunk text
- embedding vector

At query time, the agent retrieves the most relevant chunks by cosine similarity. The retrieved context is then injected into the chat turn so the model can generate a grounded answer.

The retrieval output includes a citation string such as:

- `docs/demo-kb/01-overview.md#L1-L8`

That citation is meant for quick review during demos. It shows exactly where the answer came from and makes the retrieval step visible before generation.

The automatic RAG behavior can be disabled with `--no-rag` when you want a pure chat session.