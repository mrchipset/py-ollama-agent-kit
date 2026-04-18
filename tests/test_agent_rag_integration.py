from __future__ import annotations

from dataclasses import dataclass, field

from ollama_agent_kit.agent import TeachingAgent
from ollama_agent_kit.config import Settings
from ollama_agent_kit.rag import RagSearchHit


@dataclass
class _FakeClient:
    calls: list[dict] = field(default_factory=list)

    def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        return {
            "message": {
                "role": "assistant",
                "content": "Answer with citation.",
            }
        }


@dataclass
class _SequencedFakeClient:
    responses: list[dict]
    calls: list[dict] = field(default_factory=list)

    def chat(self, *, model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        return self.responses.pop(0)


@dataclass
class _FakeRagStore:
    hits: list[RagSearchHit]
    queries: list[str] = field(default_factory=list)

    def search(self, query: str, *, top_k: int = 5) -> list[RagSearchHit]:
        self.queries.append(query)
        return self.hits[:top_k]


def test_agent_injects_retrieved_context_into_request() -> None:
    client = _FakeClient()
    rag_store = _FakeRagStore(
        hits=[
            RagSearchHit(
                score=0.912,
                citation="docs/guide.md#L1-L4",
                source_path="docs/guide.md",
                heading="Introduction",
                heading_line=1,
                line_start=2,
                line_end=4,
                excerpt="The project explains how the agent works.",
                text="# Introduction\n\nThe project explains how the agent works.",
            )
        ]
    )
    agent = TeachingAgent(settings=Settings(), client=client, rag_store=rag_store)

    turn = agent.run_turn("How does the project work?")

    assert turn.reply == "Answer with citation."
    assert len(turn.rag_hits) == 1
    assert rag_store.queries == ["How does the project work?"]
    assert len(client.calls) == 1
    request_messages = client.calls[0]["messages"]
    assert request_messages[-1]["role"] == "system"
    assert "Retrieved Markdown context" in request_messages[-1]["content"]
    assert "docs/guide.md#L1-L4" in request_messages[-1]["content"]
    assert "How does the project work?" in request_messages[0]["content"] or request_messages[1]["content"] == "How does the project work?"


def test_agent_skips_rag_when_disabled() -> None:
    client = _FakeClient()
    rag_store = _FakeRagStore(
        hits=[
            RagSearchHit(
                score=0.912,
                citation="docs/guide.md#L1-L4",
                source_path="docs/guide.md",
                heading="Introduction",
                heading_line=1,
                line_start=2,
                line_end=4,
                excerpt="The project explains how the agent works.",
                text="# Introduction\n\nThe project explains how the agent works.",
            )
        ]
    )
    settings = Settings()
    settings.rag_auto_enabled = False
    agent = TeachingAgent(settings=settings, client=client, rag_store=rag_store)

    turn = agent.run_turn("How does the project work?")

    assert turn.reply == "Answer with citation."
    assert turn.rag_hits == []
    assert rag_store.queries == []
    request_messages = client.calls[0]["messages"]
    assert all("Retrieved Markdown context" not in message.get("content", "") for message in request_messages)


def test_agent_falls_back_when_model_returns_empty_response() -> None:
    client = _SequencedFakeClient(
        responses=[
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                }
            },
        ]
    )
    rag_store = _FakeRagStore(
        hits=[
            RagSearchHit(
                score=0.654,
                citation="docs/demo-kb/05-troubleshooting.md#L19-L21",
                source_path="docs/demo-kb/05-troubleshooting.md",
                heading="Auto RAG is getting in the way",
                heading_line=19,
                line_start=19,
                line_end=21,
                excerpt="Use --no-rag for pure chat sessions.",
                text="# Auto RAG is getting in the way\n\nUse --no-rag for pure chat sessions.",
            )
        ]
    )
    agent = TeachingAgent(settings=Settings(), client=client, rag_store=rag_store)

    turn = agent.run_turn("How do I disable automatic retrieval?")

    assert turn.reply.startswith("Use --no-rag for pure chat sessions.")
    assert "docs/demo-kb/05-troubleshooting.md#L19-L21" in turn.reply
    assert len(client.calls) == 2
    assert client.calls[1]["tools"] is None
    assert any(
        message["role"] == "system"
        and "Answer the user directly in plain text" in message["content"]
        for message in client.calls[1]["messages"]
    )
    assert turn.rag_hits[0].citation == "docs/demo-kb/05-troubleshooting.md#L19-L21"