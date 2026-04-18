from __future__ import annotations

from dataclasses import dataclass

from ollama_agent_kit.agent import AgentTurn
from ollama_agent_kit.cli import _run_single_turn
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