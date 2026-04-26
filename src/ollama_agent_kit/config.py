from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_SYSTEM_PROMPT = (
    "You are a teaching assistant for Ollama agent demos. "
    "Answer clearly, use tools when they help, and explain what you used. "
    "Only use the tool names provided in the tool list. "
    "Never invent tool names or fake tool-call JSON in assistant content. "
    "If you need a tool, emit a real tool call using tool_calls."
)

DEFAULT_TASK_EXECUTION_PROMPT = (
    "When a task needs multiple steps, keep using tools until the task is complete. "
    "Inspect each tool result, decide whether another tool call is needed, and only give the final answer once you have enough evidence."
)


@dataclass(slots=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    system_prompt: str = os.getenv("OLLAMA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    task_execution_prompt: str = os.getenv("OLLAMA_TASK_EXECUTION_PROMPT", DEFAULT_TASK_EXECUTION_PROMPT)
    context_max_messages: int = int(os.getenv("CONTEXT_MAX_MESSAGES", "24"))
    context_summary_enabled: bool = os.getenv("CONTEXT_SUMMARY_ENABLED", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    context_summary_max_chars: int = int(os.getenv("CONTEXT_SUMMARY_MAX_CHARS", "1200"))
    tool_mode: str = os.getenv("OLLAMA_TOOL_MODE", "builtin")
    tool_modules: str = os.getenv("OLLAMA_TOOL_MODULES", "")
    tool_registry_strict: bool = os.getenv("OLLAMA_TOOL_REGISTRY_STRICT", "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    debug_log_path: str | None = os.getenv("OLLAMA_DEBUG_LOG_PATH") or None
    rag_auto_enabled: bool = os.getenv("RAG_AUTO_ENABLED", "true").lower() not in {"0", "false", "no", "off"}
    rag_index_path: str = os.getenv("RAG_INDEX_PATH", str(ROOT_DIR / "data" / "rag_index.json"))
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", os.getenv("OLLAMA_EMBEDDING_MODEL", os.getenv("OLLAMA_MODEL", "llama3.2")))
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    python_exec_timeout_seconds: float = float(os.getenv("PYTHON_EXEC_TIMEOUT_SECONDS", "5"))
    python_exec_max_output_chars: int = int(os.getenv("PYTHON_EXEC_MAX_OUTPUT_CHARS", "4000"))
    python_exec_allowed_imports: str = os.getenv(
        "PYTHON_EXEC_ALLOWED_IMPORTS",
        "math,statistics,json,re,datetime,decimal,fractions,itertools,functools,collections,operator,random,string,textwrap,heapq,bisect,typing,dataclasses",
    )


def get_settings() -> Settings:
    return Settings()
