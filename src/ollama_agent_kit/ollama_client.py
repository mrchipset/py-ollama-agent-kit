from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


class OllamaAPIError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaClient:
    host: str
    timeout: float = 120.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(base_url=self.host.rstrip("/"), timeout=self.timeout)

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/tags")
        return payload.get("models", [])

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
        return self._request("POST", "/api/chat", json=body)

    def embeddings(self, *, model: str, prompt: str) -> list[float]:
        payload = self._request("POST", "/api/embeddings", json={"model": model, "prompt": prompt})
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaAPIError("Ollama embeddings response missing embedding list.")

        return [float(value) for value in embedding]

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            message = exc.response.text.strip() or str(exc)
            raise OllamaAPIError(f"Ollama request failed: {message}") from exc
        except httpx.HTTPError as exc:
            raise OllamaAPIError(f"Cannot reach Ollama at {self.host}: {exc}") from exc

        return response.json()
