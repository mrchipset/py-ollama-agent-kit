from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

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
        images: list[str | bytes | Path] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        body = self._build_chat_body(
            model=model,
            messages=messages,
            tools=tools,
            images=images,
            stream=stream,
        )
        return self._request("POST", "/api/chat", json=body)

    def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        images: list[str | bytes | Path] | None = None,
    ) -> Iterator[dict[str, Any]]:
        body = self._build_chat_body(
            model=model,
            messages=messages,
            tools=tools,
            images=images,
            stream=True,
        )

        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    yield json.loads(line)
        except httpx.HTTPStatusError as exc:
            message = exc.response.text.strip() or str(exc)
            raise OllamaAPIError(f"Ollama request failed: {message}") from exc
        except httpx.HTTPError as exc:
            raise OllamaAPIError(f"Cannot reach Ollama at {self.host}: {exc}") from exc

    def embeddings(self, *, model: str, prompt: str) -> list[float]:
        payload = self._request("POST", "/api/embeddings", json={"model": model, "prompt": prompt})
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaAPIError("Ollama embeddings response missing embedding list.")

        return [float(value) for value in embedding]

    def _build_chat_body(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        images: list[str | bytes | Path] | None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": self._attach_images(messages, images),
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
        return body

    def _attach_images(
        self,
        messages: list[dict[str, Any]],
        images: list[str | bytes | Path] | None,
    ) -> list[dict[str, Any]]:
        if not images:
            return messages

        normalized_images = [self._encode_image(image) for image in images]
        updated_messages = [dict(message) for message in messages]

        for message in reversed(updated_messages):
            if message.get("role") == "user":
                existing_images = list(message.get("images", []))
                message["images"] = existing_images + normalized_images
                return updated_messages

        if updated_messages:
            updated_messages[-1]["images"] = normalized_images
            return updated_messages

        return [{"role": "user", "content": "", "images": normalized_images}]

    def _encode_image(self, image: str | bytes | Path) -> str:
        if isinstance(image, Path):
            data = image.read_bytes()
            return base64.b64encode(data).decode("utf-8")

        if isinstance(image, (bytes, bytearray)):
            return base64.b64encode(bytes(image)).decode("utf-8")

        return image

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
