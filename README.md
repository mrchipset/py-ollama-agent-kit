# Ollama Agent Kit

This workspace contains a teaching-oriented Python agent that talks to Ollama, can execute local tools, and exposes a simple CLI chat loop.

The tool system now supports three sources at the same time:

- built-in tools shipped with this repository
- custom Python modules that return `ToolDefinition` objects
- MCP tools discovered from stdio MCP servers

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
|       |-- mcp.py
|       |-- mcp_test_server.py
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
- `OLLAMA_TOOL_MODE=builtin+mcp` loads built-in tools and any MCP servers listed in `OLLAMA_MCP_SERVERS`.
- `OLLAMA_TOOL_MODE=builtin+custom+mcp` loads all three sources together.
- `OLLAMA_TOOL_MODE=custom-only` loads only custom tools from `OLLAMA_TOOL_MODULES`.
- `OLLAMA_TOOL_MODE=mcp-only` loads only MCP tools from `OLLAMA_MCP_SERVERS`.
- `OLLAMA_TOOL_MODULES=my_tools,other_tools` points to Python modules that expose `build_tools()`, `get_tools()`, or a `TOOLS` collection.
- `OLLAMA_MCP_SERVERS={"docs":{"command":"python","args":["-m","my_mcp_server"]}}` registers stdio MCP servers. Tools are exposed as `server__tool_name` to avoid collisions.
- `OLLAMA_MCP_SERVERS={"docs":{"transport":"http","url":"http://127.0.0.1:8080/mcp"}}` registers a remote MCP server over HTTP.
- `OLLAMA_MCP_TIMEOUT_SECONDS=15` controls how long the MCP client waits for a response before failing the request.
- `OLLAMA_TOOL_REGISTRY_STRICT=true` fails on duplicate tool names instead of skipping them.
- A minimal working example lives in [examples/custom_tools.py](examples/custom_tools.py).

Example:

```bash
OLLAMA_TOOL_MODE=builtin+custom+mcp
OLLAMA_TOOL_MODULES=my_project.tools
OLLAMA_MCP_SERVERS={"docs":{"command":"python","args":["-m","my_project.mcp_server"]}}
OLLAMA_MCP_TIMEOUT_SECONDS=15
OLLAMA_TOOL_REGISTRY_STRICT=true
```

## MCP Usage

The current MCP integration supports two transport styles:

- `stdio` for local subprocess MCP servers
- `http` for remote MCP endpoints that accept JSON-RPC POST requests

At startup, the agent:

1. parses `OLLAMA_MCP_SERVERS`
2. connects to each configured MCP server over its declared transport
3. sends `initialize`
4. fetches `tools/list`
5. registers each discovered MCP tool in the local `ToolRegistry`

Tool names are prefixed as `server__tool_name`. For example, if the MCP server is named `docs` and exposes a tool named `search`, the agent registers `docs__search`.

This prefixing is intentional. It avoids collisions with built-in tools and custom Python tools.

For local MCP testing in this repository, a minimal stdio server is included and can be started with:

```bash
python -m ollama_agent_kit.mcp_test_server
```

You can wire it into chat like this:

```bash
ollama-agent chat --tool-mode mcp-only --mcp-servers '{"test":{"command":"python","args":["-m","ollama_agent_kit.mcp_test_server"]}}'
```

Example remote MCP configuration:

```bash
OLLAMA_TOOL_MODE=mcp-only
OLLAMA_MCP_SERVERS={"remote-docs":{"transport":"http","url":"http://127.0.0.1:8080/mcp","headers":{"Authorization":"Bearer demo-token"}}}
```

For local remote-MCP testing in this repository, a simple HTTP MCP server is included and can be started with:

```bash
python -m ollama_agent_kit.remote_mcp_test_server --host 127.0.0.1 --port 8080 --token demo-token
```

Recommended startup flow:

1. Start the remote HTTP MCP server in one terminal:

```bash
python -m ollama_agent_kit.remote_mcp_test_server --host 127.0.0.1 --port 8080 --token demo-token
```

2. Point the agent at that server through `.env`, a VS Code launch configuration, or a shell environment variable:

```bash
OLLAMA_MCP_SERVERS={"remote":{"transport":"http","url":"http://127.0.0.1:8080/mcp","headers":{"Authorization":"Bearer demo-token"}}}
```

3. Start chat with MCP enabled:

```bash
ollama-agent chat --no-rag --stream --tool-mode builtin+mcp
```

If you are using the included VS Code launch configuration, start the HTTP server first and then run `Python: Ollama Agent Chat with Remote MCP`.

The test server currently exposes:

- `remote__remote_echo`
- `remote__remote_sum`

Example prompt:

```text
Please call the MCP tool remote_echo with "hello from remote mcp", then call remote_sum with 17 and 25, and finally explain which tools you used.
```

### VS Code Launch

The workspace includes a ready-to-run launch configuration for MCP testing in [.vscode/launch.json](c:/Users/Zouyu/repos/py-ollama-agent-kit/.vscode/launch.json):

- `Python: Ollama Agent Chat with Test MCP`
- `Python: Ollama Agent Chat with Remote MCP`

These launch configurations start chat with `builtin+mcp`, disable RAG for a cleaner MCP demo, and inject `OLLAMA_MCP_SERVERS` through the debug environment.

### Windows Note

On Windows PowerShell, passing JSON directly through `--mcp-servers` is easy to get wrong because shell parsing rewrites quotes before Python receives the argument. If you see `OLLAMA_MCP_SERVERS must be valid JSON`, prefer one of these approaches:

- put `OLLAMA_MCP_SERVERS` in `.env`
- set `OLLAMA_MCP_SERVERS` in the launch configuration `env`
- set the environment variable in PowerShell before starting the program

Example:

```powershell
$env:OLLAMA_MCP_SERVERS = '{"test":{"command":"c:/Users/Zouyu/repos/py-ollama-agent-kit/.venv/Scripts/python.exe","args":["-m","ollama_agent_kit.mcp_test_server"]}}'
c:/Users/Zouyu/repos/py-ollama-agent-kit/.venv/Scripts/python.exe -m ollama_agent_kit chat --stream --tool-mode builtin+mcp
```

You can disable automatic RAG injection when you want a pure chat session:

```bash
ollama-agent chat --no-rag
```

The low-level client also supports multimodal chat requests. Pass base64 strings,
raw bytes, or `Path` objects through the `images` argument and the client will
attach them to the last user message:

```python
from pathlib import Path

from ollama_agent_kit.ollama_client import OllamaClient

client = OllamaClient("http://localhost:11434")
response = client.chat(
    model="llava",
    messages=[{"role": "user", "content": "What is in this image?"}],
    images=[Path("sample.png")],
)
```

For the CLI chat loop, you can attach images to a turn in two ways:

- one-shot: `ollama-agent chat --image sample.png "What is in this image?"`
- interactive: type `:image sample.png -- What is in this image?`

In both cases, the image is attached to that turn's user message and remains in the conversation history for later turns.

If you are using the no-stream entry point and want to send an image in the current turn, the full command is:

```bash
ollama-agent chat --no-stream --image sample.png "What is in this image?"
```

If you are already inside the interactive no-stream session, you can make the current turn image-enabled by typing:

```text
:image sample.png -- What is in this image?
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
- MCP stdio tool 接入
- MCP 测试服务器与调试配置
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
- MCP stdio server discovery and execution
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

## MCP Implementation Notes

- MCP transport support is currently stdio only.
- The MCP client lives in [src/ollama_agent_kit/mcp.py](c:/Users/Zouyu/repos/py-ollama-agent-kit/src/ollama_agent_kit/mcp.py).
- The sample test server lives in [src/ollama_agent_kit/mcp_test_server.py](c:/Users/Zouyu/repos/py-ollama-agent-kit/src/ollama_agent_kit/mcp_test_server.py).
- Registry composition still flows through [src/ollama_agent_kit/tools.py](c:/Users/Zouyu/repos/py-ollama-agent-kit/src/ollama_agent_kit/tools.py), so built-in, custom, and MCP tools all share the same execution loop.
