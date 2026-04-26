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


def test_markdown_rag_store_lists_documents_and_stats(tmp_path: Path) -> None:
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

    store.add_markdown_file(Path("docs/guide.md"))

    documents = store.list_documents()
    stats = store.stats()

    assert documents == [
        {
            "source_path": "docs/guide.md",
            "chunk_count": 2,
            "line_start": 1,
            "line_end": 7,
            "citation": "docs/guide.md#L1-L3",
        }
    ]
    assert stats.source_count == 1
    assert stats.chunk_count == 2
    assert stats.embedding_model == "fake-embed"
    assert stats.chunk_size == 120
    assert stats.chunk_overlap == 20


def test_markdown_rag_store_add_directory_and_rebuild(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "guide.md").write_text("# Alpha\n\nAlpha content.", encoding="utf-8")
    (nested_dir / "notes.md").write_text("# Beta\n\nBeta content.", encoding="utf-8")

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    batch_result = store.add_markdown_directory(Path("docs"))
    assert batch_result.sources_added == 2
    assert batch_result.chunks_added == 2
    assert batch_result.total_chunks == 2

    rebuild_result = store.rebuild_index()
    assert rebuild_result.chunks_rebuilt == 2
    assert rebuild_result.total_chunks == 2
    assert len(client.calls) == 4


def test_markdown_rag_store_refreshes_only_changed_sources(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    alpha_path = tmp_path / "docs" / "alpha.md"
    beta_path = tmp_path / "docs" / "beta.md"
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    alpha_path.write_text("# Alpha\n\nAlpha content.", encoding="utf-8")
    beta_path.write_text("# Beta\n\nBeta content.", encoding="utf-8")

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    store.add_markdown_directory(Path("docs"))
    alpha_path.write_text("# Alpha\n\nAlpha content updated.", encoding="utf-8")

    result = store.refresh_index()

    assert result.sources_scanned == 2
    assert result.sources_rebuilt == 1
    assert result.chunks_rebuilt == 1
    assert result.stale_sources == ["docs/alpha.md"]
    assert result.missing_sources == []
    assert len(client.calls) == 3


def test_markdown_rag_store_health_check_reports_missing_source(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    markdown_file = tmp_path / "docs" / "guide.md"
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text("# Alpha\n\nAlpha content.", encoding="utf-8")

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    store.add_markdown_file(Path("docs/guide.md"))
    markdown_file.unlink()

    result = store.health_check()

    assert result.healthy is False
    assert result.missing_sources == ["docs/guide.md"]
    assert any(issue.code == "missing_source" for issue in result.issues)


def test_markdown_rag_store_delete_source(tmp_path: Path) -> None:
    workspace_root = tmp_path
    index_path = tmp_path / "data" / "rag_index.json"
    source_path = tmp_path / "docs" / "guide.md"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("# Alpha\n\nAlpha content.", encoding="utf-8")

    client = _FakeOllamaClient()
    store = MarkdownRagStore(
        workspace_root=workspace_root,
        index_path=index_path,
        client=client,
        embedding_model="fake-embed",
        chunk_size=120,
        chunk_overlap=20,
    )

    store.add_markdown_file(Path("docs/guide.md"))
    delete_result = store.delete_source(Path("docs/guide.md"))

    assert delete_result.source_path == "docs/guide.md"
    assert delete_result.chunks_deleted == 1
    assert delete_result.total_chunks == 0
    assert store.list_documents() == []