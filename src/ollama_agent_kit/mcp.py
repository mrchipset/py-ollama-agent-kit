from __future__ import annotations

import atexit
import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class McpServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    cwd: str | None = None


class StdioMcpClient:
    def __init__(self, config: McpServerConfig, *, timeout_seconds: float = 15.0) -> None:
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[bytes] | None = None
        self._request_id = 0
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mcp-{config.name}")
        atexit.register(self.close)

    def list_tools(self) -> list[dict[str, Any]]:
        self._ensure_initialized()
        result = self._send_request("tools/list", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise RuntimeError(f"MCP server {self._config.name!r} returned invalid tools/list payload")
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        result = self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"MCP server {self._config.name!r} returned invalid tools/call payload")
        return result

    def close(self) -> None:
        if self._process is None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return
        try:
            if self._process.poll() is None:
                self._process.terminate()
                self._process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1)
        finally:
            self._process = None
            self._initialized = False
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        cwd = None
        if self._config.cwd:
            cwd = str(Path(self._config.cwd).expanduser())

        env = None
        if self._config.env is not None:
            env = {**os.environ, **self._config.env}

        self._process = subprocess.Popen(
            [self._config.command, *self._config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "ollama-agent-kit",
                    "version": "0.1.0",
                },
            },
        )
        self._send_notification("notifications/initialized", {})
        self._initialized = True

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        self._write_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
            }
        )

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._request_id += 1
        request_id = self._request_id
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }
        )

        while True:
            message = self._read_message_with_timeout()
            if "id" not in message:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message["error"]
                raise RuntimeError(f"MCP request {method!r} failed for server {self._config.name!r}: {error}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(
                    f"MCP request {method!r} returned non-object result for server {self._config.name!r}"
                )
            return result

    def _write_message(self, payload: dict[str, Any]) -> None:
        process = self._require_process()
        stdin = process.stdin
        if stdin is None:
            raise RuntimeError(f"MCP server {self._config.name!r} stdin is not available")

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        stdin.write(header)
        stdin.write(body)
        stdin.flush()

    def _read_message_with_timeout(self) -> dict[str, Any]:
        future = self._executor.submit(self._read_message)
        try:
            return future.result(timeout=self._timeout_seconds)
        except FutureTimeoutError as exc:
            self.close()
            raise RuntimeError(
                f"MCP server {self._config.name!r} did not respond within {self._timeout_seconds} seconds"
            ) from exc

    def _read_message(self) -> dict[str, Any]:
        process = self._require_process()
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError(f"MCP server {self._config.name!r} stdout is not available")

        headers: dict[str, str] = {}
        while True:
            line = stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self._config.name!r} closed stdout unexpectedly")
            if line in {b"\r\n", b"\n"}:
                break

            decoded = line.decode("ascii", errors="strict").strip()
            key, separator, value = decoded.partition(":")
            if not separator:
                raise RuntimeError(f"MCP server {self._config.name!r} returned an invalid header line: {decoded!r}")
            headers[key.lower()] = value.strip()

        content_length = headers.get("content-length")
        if content_length is None:
            raise RuntimeError(f"MCP server {self._config.name!r} response did not include Content-Length")

        payload = stdout.read(int(content_length))
        if not payload:
            raise RuntimeError(f"MCP server {self._config.name!r} returned an empty payload")
        message = json.loads(payload.decode("utf-8"))
        if not isinstance(message, dict):
            raise RuntimeError(f"MCP server {self._config.name!r} returned a non-object message")
        return message

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise RuntimeError(f"MCP server {self._config.name!r} is not running")
        if self._process.poll() is not None:
            raise RuntimeError(
                f"MCP server {self._config.name!r} exited with code {self._process.returncode}"
            )
        return self._process


def parse_mcp_server_configs(raw_config: str) -> list[McpServerConfig]:
    if not raw_config.strip():
        return []

    try:
        raw_config = raw_config.replace("\\", "/")
        parsed = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError("OLLAMA_MCP_SERVERS must be valid JSON") from exc

    configs: list[McpServerConfig] = []
    if isinstance(parsed, dict):
        items = [{"name": name, **value} for name, value in parsed.items()]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("OLLAMA_MCP_SERVERS must be a JSON object or array")

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each MCP server definition must be a JSON object")
        name = item.get("name")
        command = item.get("command")
        args = item.get("args", [])
        env = item.get("env")
        cwd = item.get("cwd")

        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each MCP server definition needs a non-empty string name")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"MCP server {name!r} needs a non-empty string command")
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            raise ValueError(f"MCP server {name!r} args must be a list of strings")
        if env is not None:
            if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
                raise ValueError(f"MCP server {name!r} env must be an object of string values")
        if cwd is not None and not isinstance(cwd, str):
            raise ValueError(f"MCP server {name!r} cwd must be a string")

        configs.append(
            McpServerConfig(
                name=name,
                command=command,
                args=args,
                env=env,
                cwd=cwd,
            )
        )

    return configs