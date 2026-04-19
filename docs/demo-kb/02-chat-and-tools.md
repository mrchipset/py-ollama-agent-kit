# Chat Loop and Tools

The chat loop is intentionally small. A single turn follows this shape:

1. append the user message
2. send the full conversation to Ollama
3. inspect the assistant response
4. execute tool calls when present
5. feed tool results back into the conversation
6. stop when the assistant returns a normal answer

The built-in tools are:

- `get_current_time`
- `add_numbers`
- `run_python_code`
- `list_workspace`
- `read_workspace_file`

The agent also includes guardrails for malformed model output. If the assistant writes fake tool JSON into normal content, the turn is rejected and retried. If the model requests an unknown tool name, that is also rejected before execution.

For debugging, every turn can be written to a JSONL file. The log captures:

- user input
- the exact request payload sent to Ollama
- the raw Ollama response payload
- hallucination guard events when the model invents tool-like content

This makes it easy to explain what the model actually did, even when its visible response looks plausible but is not a real tool call.