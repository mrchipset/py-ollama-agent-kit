from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_SYSTEM_PROMPT = (
    "You are a teaching assistant for Ollama agent demos. "
    "Answer clearly, use tools when they help, and explain what you used."
)


@dataclass(slots=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    system_prompt: str = os.getenv("OLLAMA_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


def get_settings() -> Settings:
    return Settings()
