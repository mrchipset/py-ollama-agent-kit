from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .config import ROOT_DIR, Settings

ToolHandler = Callable[[dict[str, Any]], str]

DEFAULT_ALLOWED_PYTHON_IMPORTS = {
    "bisect",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "typing",
}

_PYTHON_EXEC_BOOTSTRAP = r'''
from __future__ import annotations

import ast
import builtins
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout

payload = json.load(sys.stdin)
code = payload.get("code", "")
allowed_imports = set(payload.get("allowed_imports", []))
max_output_chars = int(payload.get("max_output_chars", 4000))

banned_call_names = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
    "exit",
}


def truncate(text: str) -> tuple[str, bool]:
    if len(text) <= max_output_chars:
        return text, False
    if max_output_chars <= 3:
        return text[:max_output_chars], True
    return text[: max_output_chars - 3] + "...", True


def validate_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".", 1)[0]
                if top_level not in allowed_imports:
                    raise ValueError(f"Import not allowed: {top_level}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Relative imports are not allowed")
            module = node.module or ""
            top_level = module.split(".", 1)[0] if module else ""
            if top_level not in allowed_imports:
                raise ValueError(f"Import not allowed: {top_level or '<unknown>'}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in banned_call_names:
                raise ValueError(f"Call not allowed: {func.id}")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise ValueError(f"Name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError(f"Attribute not allowed: {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("Global and nonlocal statements are not allowed")


def safe_import(name: str, globals=None, locals=None, fromlist=(), level=0):
    if level != 0:
        raise ImportError("Relative imports are not allowed")
    top_level = name.split(".", 1)[0]
    if top_level not in allowed_imports:
        raise ImportError(f"Import not allowed: {top_level}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def make_safe_builtins() -> dict[str, Any]:
    return {
        "ArithmeticError": builtins.ArithmeticError,
        "AssertionError": builtins.AssertionError,
        "AttributeError": builtins.AttributeError,
        "BaseException": builtins.BaseException,
        "Exception": builtins.Exception,
        "False": False,
        "IndexError": builtins.IndexError,
        "KeyError": builtins.KeyError,
        "NameError": builtins.NameError,
        "None": None,
        "NotImplemented": NotImplemented,
        "NotImplementedError": builtins.NotImplementedError,
        "OverflowError": builtins.OverflowError,
        "RuntimeError": builtins.RuntimeError,
        "True": True,
        "TypeError": builtins.TypeError,
        "ValueError": builtins.ValueError,
        "abs": builtins.abs,
        "all": builtins.all,
        "any": builtins.any,
        "ascii": builtins.ascii,
        "bin": builtins.bin,
        "bool": builtins.bool,
        "bytes": builtins.bytes,
        "callable": builtins.callable,
        "chr": builtins.chr,
        "classmethod": builtins.classmethod,
        "complex": builtins.complex,
        "dict": builtins.dict,
        "divmod": builtins.divmod,
        "enumerate": builtins.enumerate,
        "filter": builtins.filter,
        "float": builtins.float,
        "format": builtins.format,
        "frozenset": builtins.frozenset,
        "getattr": builtins.getattr,
        "hasattr": builtins.hasattr,
        "hash": builtins.hash,
        "hex": builtins.hex,
        "id": builtins.id,
        "int": builtins.int,
        "isinstance": builtins.isinstance,
        "issubclass": builtins.issubclass,
        "iter": builtins.iter,
        "len": builtins.len,
        "list": builtins.list,
        "map": builtins.map,
        "max": builtins.max,
        "min": builtins.min,
        "next": builtins.next,
        "object": builtins.object,
        "oct": builtins.oct,
        "ord": builtins.ord,
        "pow": builtins.pow,
        "print": builtins.print,
        "property": builtins.property,
        "range": builtins.range,
        "repr": builtins.repr,
        "reversed": builtins.reversed,
        "round": builtins.round,
        "set": builtins.set,
        "slice": builtins.slice,
        "sorted": builtins.sorted,
        "staticmethod": builtins.staticmethod,
        "str": builtins.str,
        "sum": builtins.sum,
        "super": builtins.super,
        "tuple": builtins.tuple,
        "type": builtins.type,
        "zip": builtins.zip,
        "__build_class__": builtins.__build_class__,
        "__import__": safe_import,
    }


result: dict[str, Any] = {
    "ok": True,
    "stdout": "",
    "stdout_truncated": False,
    "stderr": "",
    "stderr_truncated": False,
    "result": None,
    "result_type": None,
    "exit_code": 0,
    "timed_out": False,
    "validation_error": None,
    "error": None,
    "error_type": None,
    "traceback": None,
}
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

try:
    tree = ast.parse(code, mode="exec")
    validate_tree(tree)
except Exception as exc:
    result["ok"] = False
    result["exit_code"] = 1
    result["validation_error"] = str(exc)
    result["error_type"] = "ValidationError"
    result["traceback"] = traceback.format_exc()
else:
    globals_dict = {
        "__builtins__": make_safe_builtins(),
        "__name__": "__main__",
    }
    locals_dict: dict[str, Any] = {}
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(compile(tree, "<python-exec>", "exec"), globals_dict, locals_dict)
        produced = locals_dict.get("result", globals_dict.get("result"))
        if produced is not None:
            result["result"] = repr(produced)
            result["result_type"] = type(produced).__name__
    except Exception as exc:
        result["ok"] = False
        result["exit_code"] = 1
        result["error"] = str(exc)
        result["error_type"] = exc.__class__.__name__
        result["traceback"] = traceback.format_exc()

stdout_text, stdout_truncated = truncate(stdout_buffer.getvalue())
stderr_text, stderr_truncated = truncate(stderr_buffer.getvalue())
result["stdout"] = stdout_text
result["stdout_truncated"] = stdout_truncated
result["stderr"] = stderr_text
result["stderr_truncated"] = stderr_truncated
json.dump(result, sys.stdout, ensure_ascii=False)
'''


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def as_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolExecution:
    name: str
    arguments: dict[str, Any]
    result: str


class ToolRegistry:
    def __init__(self, tools: list[ToolDefinition]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.as_ollama_tool() for tool in self._tools.values()]

    def execute_tool_call(self, tool_call: dict[str, Any]) -> ToolExecution:
        function_payload = tool_call.get("function", tool_call)
        name = function_payload["name"]
        arguments = self._coerce_arguments(function_payload.get("arguments", {}))
        result = self.execute(name, arguments)
        return ToolExecution(name=name, arguments=arguments, result=result)

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].handler(arguments)

    @staticmethod
    def _coerce_arguments(raw_arguments: Any) -> dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return raw_arguments
        if isinstance(raw_arguments, str):
            if not raw_arguments.strip():
                return {}
            return json.loads(raw_arguments)
        raise TypeError(f"Unsupported tool arguments: {type(raw_arguments)!r}")


def build_default_registry(
    workspace_root: Path | None = None,
    settings: Settings | None = None,
) -> ToolRegistry:
    root = (workspace_root or ROOT_DIR).resolve()
    active_settings = settings or Settings()

    def get_current_time(_: dict[str, Any]) -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_numbers(arguments: dict[str, Any]) -> str:
        left = float(arguments["a"])
        right = float(arguments["b"])
        return json.dumps({"sum": left + right})

    def list_workspace(arguments: dict[str, Any]) -> str:
        relative_path = arguments.get("path", ".")
        limit = int(arguments.get("limit", 20))
        target = _resolve_workspace_path(root, relative_path)

        if target.is_file():
            return json.dumps({"path": str(target.relative_to(root)), "type": "file"}, indent=2)

        entries = []
        for child in sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name.lower()))[:limit]:
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(root)),
                    "type": "dir" if child.is_dir() else "file",
                }
            )
        return json.dumps({"path": str(target.relative_to(root)) or ".", "entries": entries}, indent=2)

    def read_workspace_file(arguments: dict[str, Any]) -> str:
        relative_path = arguments["path"]
        max_chars = int(arguments.get("max_chars", 2000))
        target = _resolve_workspace_path(root, relative_path)

        if not target.is_file():
            raise FileNotFoundError(f"Not a file: {relative_path}")

        content = target.read_text(encoding="utf-8")
        return content[:max_chars]

    def run_python_code(arguments: dict[str, Any]) -> str:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")

        timeout_seconds = float(active_settings.python_exec_timeout_seconds)
        max_output_chars = int(active_settings.python_exec_max_output_chars)
        allowed_imports = sorted(
            _parse_allowed_imports(active_settings.python_exec_allowed_imports) or DEFAULT_ALLOWED_PYTHON_IMPORTS
        )

        payload = {
            "code": code,
            "allowed_imports": allowed_imports,
            "max_output_chars": max_output_chars,
        }

        with tempfile.TemporaryDirectory(prefix="python-exec-") as temp_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", _PYTHON_EXEC_BOOTSTRAP],
                    input=json.dumps(payload, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    cwd=temp_dir,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return json.dumps(
                    {
                        "ok": False,
                        "stdout": exc.stdout or "",
                        "stdout_truncated": False,
                        "stderr": exc.stderr or "",
                        "stderr_truncated": False,
                        "result": None,
                        "result_type": None,
                        "exit_code": None,
                        "timed_out": True,
                        "validation_error": None,
                        "error": f"Python execution exceeded {timeout_seconds} seconds.",
                        "error_type": "TimeoutError",
                        "traceback": None,
                    },
                    ensure_ascii=False,
                )

        if not completed.stdout.strip():
            return json.dumps(
                {
                    "ok": False,
                    "stdout": "",
                    "stdout_truncated": False,
                    "stderr": completed.stderr.strip(),
                    "stderr_truncated": False,
                    "result": None,
                    "result_type": None,
                    "exit_code": completed.returncode,
                    "timed_out": False,
                    "validation_error": "Python execution bootstrap returned no result.",
                    "error": None,
                    "error_type": "ExecutionError",
                    "traceback": None,
                },
                ensure_ascii=False,
            )

        result = json.loads(completed.stdout)
        return json.dumps(result, ensure_ascii=False)

    tools = [
        ToolDefinition(
            name="get_current_time",
            description="Return the current UTC time in ISO 8601 format.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=get_current_time,
        ),
        ToolDefinition(
            name="add_numbers",
            description="Add two numbers and return the sum.",
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number."},
                    "b": {"type": "number", "description": "Second number."},
                },
                "required": ["a", "b"],
            },
            handler=add_numbers,
        ),
        ToolDefinition(
            name="run_python_code",
            description="Execute a short Python snippet in a controlled subprocess and return structured stdout, stderr, and result data.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() or assign a result variable to return values.",
                    },
                },
                "required": ["code"],
            },
            handler=run_python_code,
        ),
        ToolDefinition(
            name="list_workspace",
            description="List files or directories inside the current workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path inside the workspace."},
                    "limit": {"type": "integer", "description": "Maximum number of entries to return."},
                },
                "required": [],
            },
            handler=list_workspace,
        ),
        ToolDefinition(
            name="read_workspace_file",
            description="Read a UTF-8 text file from the current workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path inside the workspace."},
                    "max_chars": {"type": "integer", "description": "Maximum number of characters to return."},
                },
                "required": ["path"],
            },
            handler=read_workspace_file,
        ),
    ]
    return ToolRegistry(tools)


def _parse_allowed_imports(raw_imports: str) -> set[str]:
    return {
        item.strip()
        for item in raw_imports.split(",")
        if item.strip()
    }


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = (workspace_root / raw_path).resolve()
    candidate.relative_to(workspace_root)
    return candidate