from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class _FakeJSONResponse:
    payload: dict

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


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


def test_chat_with_images_attaches_base64_image_to_last_user_message(monkeypatch, tmp_path: Path) -> None:
    client = OllamaClient("http://127.0.0.1:11434")
    captured_bodies: list[dict] = []

    image_path = tmp_path / "sample.bin"
    image_path.write_bytes(b"image-bytes")

    def fake_request(method, path, **kwargs):
        captured_bodies.append(kwargs["json"])
        return _FakeJSONResponse({"message": {"role": "assistant", "content": "ok"}})

    monkeypatch.setattr(client._client, "request", fake_request)

    client.chat(
        model="demo",
        messages=[
            {"role": "system", "content": "keep context"},
            {"role": "user", "content": "what is in this image?"},
        ],
        images=[image_path],
    )

    body = captured_bodies[0]
    assert body["stream"] is False
    assert body["messages"][-1]["images"] == ["aW1hZ2UtYnl0ZXM="]
    assert body["messages"][0]["role"] == "system"


def test_stream_chat_with_images_attaches_to_existing_user_message(monkeypatch) -> None:
    client = OllamaClient("http://127.0.0.1:11434")
    captured_bodies: list[dict] = []

    def fake_stream(method, path, **kwargs):
        captured_bodies.append(kwargs["json"])
        response = _FakeStreamResponse(lines=['{"message": {"role": "assistant", "content": "done"}, "done": true}'])
        return _FakeStreamContext(response)

    monkeypatch.setattr(client._client, "stream", fake_stream)

    list(
        client.stream_chat(
            model="demo",
            messages=[{"role": "user", "content": "describe this"}],
            images=[b"raw-bytes"],
        )
    )

    body = captured_bodies[0]
    assert body["stream"] is True
    assert body["messages"][0]["images"] == ["cmF3LWJ5dGVz"]
