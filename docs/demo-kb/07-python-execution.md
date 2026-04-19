# Controlled Python Execution Demo

This page is a teaching scenario for the controlled Python execution tool.

The goal is to show a complete agent loop where the model delegates a small computation to a tool, the tool runs in a constrained environment, and the structured result comes back into the conversation.

## What this demo shows

- a model choosing a tool instead of answering from memory
- structured execution output with `stdout`, `stderr`, `result`, and `timed_out`
- a clear boundary between safe, short snippets and unsupported code
- how tool output can be used to correct or complete an answer

## Suggested demo prompts

- Use Python to add 17 and 25, then show the result.
- Write a short snippet that prints the first five square numbers.
- Calculate the average of `[10, 12, 15, 18]` and explain the result.
- Try to import `os` and show how the tool rejects it.
- Run a loop that never ends and show the timeout behavior.

## Expected behavior

Good inputs should produce a structured JSON payload similar to:

- `ok: true`
- `stdout: "..."`
- `result: "..."`
- `result_type: "int"` or another Python type name
- `timed_out: false`

Rejected or unsafe inputs should return a failure payload with fields such as:

- `ok: false`
- `validation_error`
- `error_type`
- `timed_out: true` for long-running snippets

## Why this is useful in a teaching demo

This tool makes an important agent concept visible: the model does not need to solve every problem directly. It can hand off bounded work to a tool and then reason over the returned result.

That makes it easy to explain:

1. tool selection
2. constrained execution
3. structured return values
4. failure handling

## Safety notes

This is a controlled execution helper, not a general-purpose sandbox.

It is intentionally limited to short snippets, a small import allowlist, a timeout, and truncated output. That is enough for demos and teaching, but it should not be treated as a security boundary for untrusted code.