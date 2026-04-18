# Project Overview

Ollama Agent Kit is a teaching-oriented Python project that connects a local agent to Ollama, executes local tools, and exposes a CLI chat loop.

The project focuses on clarity rather than abstraction. It is designed so that the whole request flow is easy to inspect:

- user input enters the agent loop
- the agent sends messages to Ollama
- Ollama can request tool calls
- tool results are added back into the conversation
- the final assistant reply is printed in the terminal

The sample environment uses these defaults:

- Ollama host: `http://192.168.71.21:11434`
- chat model: `qwen2.5-coder:1.5b`
- embedding model: `llama3.2`
- automatic RAG: enabled
- RAG index path: `data/rag_index.json`

The project is intentionally simple so it can be used to demonstrate debugging, prompt design, tool calling, and retrieval augmented generation without a large framework around it.