from __future__ import annotations

import json
import sys
from typing import Any


SERVER_INFO = {
    "name": "ollama-agent-kit-test-server",
    "version": "0.1.0",
}

TOOLS = [
    {
        "name": "echo_text",
        "description": "Echo the provided text back to the caller.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to echo.",
                }
            },
            "required": ["text"],
        },
    },
    {
        "name": "sum_numbers",
        "description": "Add two numbers and return the result as text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number."},
                "b": {"type": "number", "description": "Second number."},
            },
            "required": ["a", "b"],
        },
    },
]


def _write_message(payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in {b"\r\n", b"\n"}:
            break

        decoded = line.decode("ascii", errors="strict").strip()
        key, separator, value = decoded.partition(":")
        if not separator:
            raise RuntimeError(f"Invalid header line: {decoded!r}")
        headers[key.lower()] = value.strip()

    content_length = headers.get("content-length")
    if content_length is None:
        raise RuntimeError("Missing Content-Length header")

    body = sys.stdin.buffer.read(int(content_length))
    if not body:
        return None
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Message payload must be a JSON object")
    return payload


def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _handle_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "echo_text":
        text = str(arguments.get("text", ""))
        return {
            "content": [
                {"type": "text", "text": f"Echo: {text}"},
            ]
        }
    if name == "sum_numbers":
        left = float(arguments["a"])
        right = float(arguments["b"])
        return {
            "content": [
                {"type": "text", "text": str(left + right)},
            ]
        }
    raise KeyError(name)


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break

        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if request_id is None:
            if method == "notifications/initialized":
                continue
            continue

        if method == "initialize":
            _write_message(
                _ok(
                    request_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                )
            )
            continue

        if method == "tools/list":
            _write_message(_ok(request_id, {"tools": TOOLS}))
            continue

        if method == "tools/call":
            try:
                result = _handle_tool_call(str(params.get("name", "")), dict(params.get("arguments") or {}))
            except KeyError:
                _write_message(_error(request_id, -32601, f"Unknown tool: {params.get('name')!r}"))
            except (TypeError, ValueError) as exc:
                _write_message(_error(request_id, -32602, str(exc)))
            else:
                _write_message(_ok(request_id, result))
            continue

        _write_message(_error(request_id, -32601, f"Unknown method: {method!r}"))


if __name__ == "__main__":
    main()