from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ollama_agent_kit.rag import MarkdownRagStore


@dataclass
class _FakeOllamaClient:
    calls: list[str] = field(default_factory=list)

    def embeddings(self, *, model: str, prompt: str) -> list[float]:
        self.calls.append(prompt)
        lower_prompt = prompt.lower()
        if "alpha" in lower_prompt:
            return [1.0, 0.0]
        if "beta" in lower_prompt:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_markdown_rag_store_add_search_and_clear(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    markdown_file = tmp_path / "docs" / "guide.md"
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text(
        "# Alpha\n\nAlpha content for the first section.\n\n# Beta\n\nBeta content for the second section.",
        encoding="utf-8",
    )

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    result = store.add_markdown_file(Path("docs/guide.md"))
    assert result.source_path == "docs/guide.md"
    assert result.chunks_added == 2
    assert index_path.exists()

    hits = store.search("alpha question", top_k=2)
    assert len(hits) == 2
    assert hits[0].citation == "docs/guide.md#L1-L3"
    assert "Alpha" in hits[0].excerpt
    assert hits[0].score >= hits[1].score

    store.clear()
    assert not index_path.exists()
    assert store.search("alpha question", top_k=2) == []


def test_markdown_rag_store_replaces_same_source(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    markdown_file = tmp_path / "docs" / "note.md"
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text("# Alpha\n\nOriginal alpha content.", encoding="utf-8")

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    first_result = store.add_markdown_file(Path("docs/note.md"))
    markdown_file.write_text("# Alpha\n\nUpdated alpha content with beta mention.", encoding="utf-8")
    second_result = store.add_markdown_file(Path("docs/note.md"))

    assert first_result.chunks_added == 1
    assert second_result.chunks_added == 1
    hits = store.search("beta", top_k=1)
    assert len(hits) == 1
    assert "Updated alpha content" in hits[0].text