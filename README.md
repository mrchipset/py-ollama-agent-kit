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

Tool registration can also be configured through `.env`:

- `OLLAMA_TOOL_MODE=builtin` keeps only the built-in tools.
- `OLLAMA_TOOL_MODE=builtin+custom` loads built-in tools and any modules listed in `OLLAMA_TOOL_MODULES`.
- `OLLAMA_TOOL_MODE=custom-only` loads only custom tools from `OLLAMA_TOOL_MODULES`.
- `OLLAMA_TOOL_MODULES=my_tools,other_tools` points to Python modules that expose `build_tools()`, `get_tools()`, or a `TOOLS` collection.
- `OLLAMA_TOOL_REGISTRY_STRICT=true` fails on duplicate tool names instead of skipping them.
- A minimal working example lives in [examples/custom_tools.py](examples/custom_tools.py).

Example:

```bash
OLLAMA_TOOL_MODE=builtin+custom
OLLAMA_TOOL_MODULES=my_project.tools
OLLAMA_TOOL_REGISTRY_STRICT=true
```

You can disable automatic RAG injection when you want a pure chat session:

```bash
ollama-agent chat --no-rag
```

To run the guided Python tool demo directly:

```bash
ollama-agent demo python
```

## Todo List

### 已完成

- Markdown RAG 基础接入
- 受控 Python 执行工具
- 多轮工具调用与任务执行提示
- 自定义工具注册机制
- `.env` 中的 tool 相关配置
- 最小自定义工具示例

### 待完成

- 会话导出功能
- 更强的教学场景模板
- RAG 索引管理增强
- 更丰富的 RAG 数据源
- 对话上下文管理
- 更完整的错误恢复与重试
- 更完整的可观测性
- 更强的 RAG 检索质量
- 多模型策略
- 更完整的测试与文档体系

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
- A controlled Python execution tool with structured stdout, stderr, and result output

## Built-in tools

- `get_current_time`
- `add_numbers`
- `run_python_code`
- `list_workspace`
- `read_workspace_file`

The file tools are limited to the current workspace root.
The Python execution tool is intentionally constrained: it runs in a subprocess, enforces a timeout, limits output, and only allows a small import allowlist for teaching demos.
