from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


SERVER_INFO = {
    "name": "ollama-agent-kit-remote-test-server",
    "version": "0.1.0",
}

TOOLS = [
    {
        "name": "remote_echo",
        "description": "Echo the provided text back to the caller from the remote MCP server.",
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
        "name": "remote_sum",
        "description": "Add two numbers and return the result as text from the remote MCP server.",
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
    if name == "remote_echo":
        text = str(arguments.get("text", ""))
        return {
            "content": [
                {"type": "text", "text": f"Remote echo: {text}"},
            ]
        }
    if name == "remote_sum":
        left = float(arguments["a"])
        right = float(arguments["b"])
        return {
            "content": [
                {"type": "text", "text": str(left + right)},
            ]
        }
    raise KeyError(name)


def build_handler(*, token: str | None = None, path: str = "/mcp") -> type[BaseHTTPRequestHandler]:
    normalized_path = path if path.startswith("/") else f"/{path}"

    class RemoteMcpHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != normalized_path:
                self.send_error(404)
                return

            if token is not None:
                expected = f"Bearer {token}"
                actual = self.headers.get("Authorization")
                if actual != expected:
                    self.send_error(401)
                    return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._write_json(400, {"error": "Invalid JSON"})
                return

            if not isinstance(payload, dict):
                self._write_json(400, {"error": "JSON payload must be an object"})
                return

            request_id = payload.get("id")
            method = payload.get("method")
            params = payload.get("params") or {}

            if request_id is None:
                self.send_response(202)
                self.end_headers()
                return

            if method == "initialize":
                body = _ok(
                    request_id,
                    {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                )
            elif method == "tools/list":
                body = _ok(request_id, {"tools": TOOLS})
            elif method == "tools/call":
                try:
                    body = _ok(
                        request_id,
                        _handle_tool_call(str(params.get("name", "")), dict(params.get("arguments") or {})),
                    )
                except KeyError:
                    body = _error(request_id, -32601, f"Unknown tool: {params.get('name')!r}")
                except (TypeError, ValueError) as exc:
                    body = _error(request_id, -32602, str(exc))
            else:
                body = _error(request_id, -32601, f"Unknown method: {method!r}")

            self._write_json(200, body)

        def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    return RemoteMcpHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the remote HTTP MCP test server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on.")
    parser.add_argument("--path", default="/mcp", help="HTTP path for the MCP endpoint.")
    parser.add_argument("--token", default=None, help="Optional bearer token required by the server.")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), build_handler(token=args.token, path=args.path))
    print(f"Remote MCP test server listening on http://{args.host}:{args.port}{args.path}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()