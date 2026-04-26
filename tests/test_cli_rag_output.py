from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from ollama_agent_kit.agent import AgentTurn
from ollama_agent_kit.cli import _run_single_turn, app
from ollama_agent_kit.rag import RagSearchHit


@dataclass
class _FakeAgent:
    def run_turn(self, user_input: str) -> AgentTurn:
        return AgentTurn(
            reply="Answer with citation.",
            tool_events=[],
            rag_hits=[
                RagSearchHit(
                    score=0.987,
                    citation="docs/guide.md#L1-L4",
                    source_path="docs/guide.md",
                    heading="Introduction",
                    heading_line=1,
                    line_start=2,
                    line_end=4,
                    excerpt="The project explains how the agent works.",
                    text="# Introduction\n\nThe project explains how the agent works.",
                )
            ],
        )


@dataclass
class _FakeStreamingAgent:
    def should_use_rag(self, user_input: str) -> bool:
        return True

    def _search_rag(self, user_input: str) -> list[RagSearchHit]:
        return [
            RagSearchHit(
                score=0.987,
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

    def run_turn(self, user_input: str, *, rag_hits=None, on_text_chunk=None) -> AgentTurn:
        if on_text_chunk is not None:
            on_text_chunk("Answer with citation.")
        return AgentTurn(
            reply="Answer with citation.",
            tool_events=[],
            rag_hits=rag_hits or self._search_rag(user_input),
        )


def test_run_single_turn_prints_retrieved_references(capsys) -> None:
    _run_single_turn(_FakeAgent(), "How does the project work?")

    output = capsys.readouterr().out
    assert "Retrieved references:" in output
    assert "docs/guide.md#L1-L4" in output
    assert "The project explains how the agent works." in output


def test_run_single_turn_streams_answer_after_references(capsys) -> None:
    _run_single_turn(_FakeStreamingAgent(), "How does the project work?", stream=True)

    output = capsys.readouterr().out
    assert output.index("Retrieved references:") < output.index("Answer with citation.")
    assert output.count("Retrieved references:") == 1


@dataclass
class _FakeRagStore:
    def list_documents(self) -> list[dict[str, object]]:
        return [
            {
                "source_path": "docs/guide.md",
                "chunk_count": 2,
                "line_start": 1,
                "line_end": 7,
                "citation": "docs/guide.md#L1-L3",
            }
        ]

    def stats(self):
        return type(
            "Stats",
            (),
            {
                "source_count": 1,
                "chunk_count": 2,
                "embedding_model": "fake-embed",
                "chunk_size": 120,
                "chunk_overlap": 20,
            },
        )()


def test_rag_list_command_shows_indexed_documents(monkeypatch, capsys) -> None:
    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _FakeRagStore())

    result = CliRunner().invoke(app, ["rag", "list"])

    assert result.exit_code == 0
    output = capsys.readouterr().out + result.output
    assert "docs/guide.md" in output
    assert "Chunks" in output


def test_rag_stats_command_prints_summary(monkeypatch) -> None:
    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _FakeRagStore())

    result = CliRunner().invoke(app, ["rag", "stats"])

    assert result.exit_code == 0
    assert "Indexed documents: 1" in result.output
    assert "Indexed chunks: 2" in result.output
    assert "Embedding model: fake-embed" in result.output


def test_rag_rebuild_command_prints_summary(monkeypatch) -> None:
    class _RebuildStore(_FakeRagStore):
        def rebuild_index(self):
            return type(
                "RebuildResult",
                (),
                {
                    "chunks_rebuilt": 2,
                    "total_chunks": 2,
                },
            )()

    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _RebuildStore())

    result = CliRunner().invoke(app, ["rag", "rebuild"])

    assert result.exit_code == 0
    assert "Rebuilt embeddings for 2 chunks" in result.output


def test_rag_delete_command_prints_summary(monkeypatch) -> None:
    class _DeleteStore(_FakeRagStore):
        def delete_source(self, path):
            return type(
                "DeleteResult",
                (),
                {
                    "source_path": str(path).replace("\\", "/"),
                    "chunks_deleted": 1,
                    "total_chunks": 1,
                },
            )()

    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _DeleteStore())

    result = CliRunner().invoke(app, ["rag", "delete", "docs/guide.md"])

    assert result.exit_code == 0
    assert "Deleted 1 chunks from docs/guide.md" in result.output


def test_rag_refresh_command_prints_summary(monkeypatch) -> None:
    class _RefreshStore(_FakeRagStore):
        def refresh_index(self):
            return type(
                "RefreshResult",
                (),
                {
                    "sources_scanned": 2,
                    "sources_rebuilt": 1,
                    "chunks_rebuilt": 1,
                    "total_chunks": 2,
                    "stale_sources": ["docs/guide.md"],
                    "missing_sources": [],
                },
            )()

    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _RefreshStore())

    result = CliRunner().invoke(app, ["rag", "refresh"])

    assert result.exit_code == 0
    assert "Scanned 2 sources and rebuilt 1 sources with 1 chunks." in result.output


def test_rag_health_command_prints_summary(monkeypatch) -> None:
    class _HealthStore(_FakeRagStore):
        def health_check(self):
            return type(
                "HealthResult",
                (),
                {
                    "healthy": False,
                    "source_count": 1,
                    "chunk_count": 2,
                    "stale_sources": ["docs/guide.md"],
                    "missing_sources": ["docs/missing.md"],
                    "issues": [
                        type(
                            "Issue",
                            (),
                            {
                                "severity": "error",
                                "code": "missing_source",
                                "source_path": "docs/missing.md",
                                "message": "Indexed source file is missing: docs/missing.md",
                            },
                        )()
                    ],
                },
            )()

    monkeypatch.setattr("ollama_agent_kit.cli._build_rag_store", lambda: _HealthStore())

    result = CliRunner().invoke(app, ["rag", "health"])

    assert result.exit_code == 0
    assert "Healthy: no" in result.output
    assert "Missing sources: 1" in result.output