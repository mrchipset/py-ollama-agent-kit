from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from ollama_agent_kit.ollama_client import OllamaClient


@dataclass
class _FakeStreamResponse:
    lines: list[str]

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self):
        yield from self.lines


@dataclass
class _FakeStreamContext:
    response: _FakeStreamResponse

    def __enter__(self) -> _FakeStreamResponse:
        return self.response

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_stream_chat_yields_ndjson_chunks(monkeypatch) -> None:
    client = OllamaClient("http://127.0.0.1:11434")
    captured_bodies: list[dict] = []

    def fake_stream(method, path, **kwargs):
        captured_bodies.append(kwargs["json"])
        response = _FakeStreamResponse(
            lines=[
                '{"message": {"role": "assistant", "content": "Hel"}, "done": false}',
                '{"message": {"role": "assistant", "content": "lo"}, "done": false}',
                '{"message": {"role": "assistant", "content": ""}, "done": true}',
            ]
        )
        return _FakeStreamContext(response)

    monkeypatch.setattr(client._client, "stream", fake_stream)

    chunks = list(client.stream_chat(model="demo", messages=[{"role": "user", "content": "hi"}]))

    assert captured_bodies[0]["stream"] is True
    assert chunks[0]["message"]["content"] == "Hel"
    assert chunks[1]["message"]["content"] == "lo"


def test_stream_chat_raises_on_http_error(monkeypatch) -> None:
    client = OllamaClient("http://127.0.0.1:11434")

    class _BrokenResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "bad",
                request=httpx.Request("POST", "http://127.0.0.1:11434/api/chat"),
                response=httpx.Response(500, text="boom"),
            )

    @contextmanager
    def fake_stream(method, path, **kwargs):
        yield _BrokenResponse()

    monkeypatch.setattr(client._client, "stream", fake_stream)

    try:
        list(client.stream_chat(model="demo", messages=[]))
    except Exception as exc:
        assert "Ollama request failed" in str(exc)
