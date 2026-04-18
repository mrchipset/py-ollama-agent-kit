from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .ollama_client import OllamaClient

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SCHEMA_VERSION = 1


@dataclass(slots=True)
class RagAddResult:
    source_path: str
    chunks_added: int
    total_chunks: int


@dataclass(slots=True)
class RagSearchHit:
    score: float
    citation: str
    source_path: str
    heading: str | None
    heading_line: int | None
    line_start: int
    line_end: int
    excerpt: str
    text: str


def format_rag_context(hits: list[RagSearchHit]) -> str:
    if not hits:
        return "No relevant Markdown context was found."

    lines = ["Retrieved Markdown context:"]
    for index, hit in enumerate(hits, start=1):
        lines.append(f"[{index}] {hit.citation} (score: {hit.score:.3f})")
        if hit.heading:
            lines.append(f"Heading: {hit.heading}")
        lines.append(hit.excerpt)
        lines.append("")

    lines.append(
        "Use only the retrieved context when it is relevant, and cite the source paths in your answer."
    )
    return "\n".join(lines).strip()


@dataclass(slots=True)
class _ChunkRecord:
    chunk_id: str
    source_path: str
    heading: str | None
    heading_line: int | None
    line_start: int
    line_end: int
    text: str
    embedding: list[float]


@dataclass(slots=True)
class _Paragraph:
    line_start: int
    line_end: int
    text: str


@dataclass(slots=True)
class _Section:
    heading: str | None
    heading_line: int | None
    body_lines: list[tuple[int, str]]


class MarkdownRagStore:
    def __init__(
        self,
        *,
        workspace_root: Path,
        index_path: Path,
        client: OllamaClient,
        embedding_model: str,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.index_path = self._resolve_path(index_path)
        self.client = client
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[_ChunkRecord] = []
        self._load()

    def add_markdown_file(self, path: Path) -> RagAddResult:
        source_path = self._resolve_source_path(path)
        if source_path.suffix.lower() not in {".md", ".markdown", ".mdown"}:
            raise ValueError(f"Not a Markdown file: {path}")

        text = source_path.read_text(encoding="utf-8")
        relative_source_path = source_path.relative_to(self.workspace_root).as_posix()
        self._chunks = [chunk for chunk in self._chunks if chunk.source_path != relative_source_path]

        chunks = self._build_chunks(relative_source_path, text, source_path.stem)
        for chunk in chunks:
            chunk.embedding = self.client.embeddings(model=self.embedding_model, prompt=chunk.text)

        self._chunks.extend(chunks)
        self._save()

        return RagAddResult(
            source_path=relative_source_path,
            chunks_added=len(chunks),
            total_chunks=len(self._chunks),
        )

    def clear(self) -> None:
        self._chunks = []
        if self.index_path.exists():
            self.index_path.unlink()

    def search(self, query: str, *, top_k: int = 5) -> list[RagSearchHit]:
        if top_k <= 0 or not self._chunks:
            return []

        query_embedding = self.client.embeddings(model=self.embedding_model, prompt=query)
        scored = [
            (self._cosine_similarity(query_embedding, chunk.embedding), chunk)
            for chunk in self._chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)

        hits: list[RagSearchHit] = []
        for score, chunk in scored[:top_k]:
            hits.append(
                RagSearchHit(
                    score=score,
                    citation=self._build_citation(chunk),
                    source_path=chunk.source_path,
                    heading=chunk.heading,
                    heading_line=chunk.heading_line,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    excerpt=_make_excerpt(chunk.text),
                    text=chunk.text,
                )
            )
        return hits

    def _build_chunks(self, source_path: str, text: str, fallback_title: str) -> list[_ChunkRecord]:
        lines = text.splitlines()
        sections = self._split_sections(lines)
        chunks: list[_ChunkRecord] = []

        chunk_counter = 1
        for section in sections:
            paragraphs = self._split_paragraphs(section.body_lines)
            heading = section.heading or fallback_title
            section_chunks = self._split_paragraphs_into_chunks(paragraphs, heading)
            for section_chunk in section_chunks:
                line_start = section_chunk["line_start"]
                line_end = section_chunk["line_end"]
                chunk_text = section_chunk["text"]
                chunks.append(
                    _ChunkRecord(
                        chunk_id=f"{source_path}:{chunk_counter}",
                        source_path=source_path,
                        heading=section.heading,
                        heading_line=section.heading_line,
                        line_start=line_start,
                        line_end=line_end,
                        text=chunk_text,
                        embedding=[],
                    )
                )
                chunk_counter += 1

        return chunks

    def _split_sections(self, lines: list[str]) -> list[_Section]:
        sections: list[_Section] = []
        current_heading: str | None = None
        current_heading_line: int | None = None
        current_body: list[tuple[int, str]] = []

        for line_number, line in enumerate(lines, start=1):
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                if current_heading is not None or current_body:
                    sections.append(
                        _Section(
                            heading=current_heading,
                            heading_line=current_heading_line,
                            body_lines=current_body,
                        )
                    )
                current_heading = heading_match.group(2).strip() or None
                current_heading_line = line_number
                current_body = []
                continue

            current_body.append((line_number, line))

        if current_heading is not None or current_body:
            sections.append(
                _Section(
                    heading=current_heading,
                    heading_line=current_heading_line,
                    body_lines=current_body,
                )
            )

        return sections

    def _split_paragraphs(self, body_lines: list[tuple[int, str]]) -> list[_Paragraph]:
        paragraphs: list[_Paragraph] = []
        current_lines: list[str] = []
        paragraph_start: int | None = None
        paragraph_end: int | None = None

        for line_number, line in body_lines:
            if line.strip():
                if paragraph_start is None:
                    paragraph_start = line_number
                paragraph_end = line_number
                current_lines.append(line.rstrip())
                continue

            if current_lines and paragraph_start is not None and paragraph_end is not None:
                paragraphs.append(
                    _Paragraph(
                        line_start=paragraph_start,
                        line_end=paragraph_end,
                        text="\n".join(current_lines).strip(),
                    )
                )
            current_lines = []
            paragraph_start = None
            paragraph_end = None

        if current_lines and paragraph_start is not None and paragraph_end is not None:
            paragraphs.append(
                _Paragraph(
                    line_start=paragraph_start,
                    line_end=paragraph_end,
                    text="\n".join(current_lines).strip(),
                )
            )

        return paragraphs

    def _split_paragraphs_into_chunks(self, paragraphs: list[_Paragraph], heading: str) -> list[dict[str, Any]]:
        if not paragraphs:
            return []

        chunks: list[dict[str, Any]] = []
        current_paragraphs: list[_Paragraph] = []
        current_length = len(heading) + 2

        def flush() -> None:
            nonlocal current_paragraphs, current_length
            if not current_paragraphs:
                return
            body_text = "\n\n".join(paragraph.text for paragraph in current_paragraphs).strip()
            line_start = current_paragraphs[0].line_start
            line_end = current_paragraphs[-1].line_end
            text = f"# {heading}\n\n{body_text}" if heading else body_text
            chunks.append(
                {
                    "line_start": line_start,
                    "line_end": line_end,
                    "text": text,
                }
            )
            current_paragraphs = []
            current_length = len(heading) + 2

        for paragraph in paragraphs:
            paragraph_length = len(paragraph.text)
            if current_paragraphs and current_length + paragraph_length + 2 > self.chunk_size:
                flush()

            current_paragraphs.append(paragraph)
            current_length += paragraph_length + 2

        flush()
        return chunks

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                    "heading": chunk.heading,
                    "heading_line": chunk.heading_line,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "text": chunk.text,
                    "embedding": chunk.embedding,
                }
                for chunk in self._chunks
            ],
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.index_path.exists():
            self._chunks = []
            return

        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("Unsupported RAG index schema version.")

        stored_embedding_model = payload.get("embedding_model")
        if stored_embedding_model != self.embedding_model:
            raise ValueError(
                "RAG index was built with a different embedding model. Clear and rebuild the index."
            )

        self._chunks = [
            _ChunkRecord(
                chunk_id=chunk["chunk_id"],
                source_path=chunk["source_path"],
                heading=chunk.get("heading"),
                heading_line=chunk.get("heading_line"),
                line_start=int(chunk["line_start"]),
                line_end=int(chunk["line_end"]),
                text=chunk["text"],
                embedding=[float(value) for value in chunk.get("embedding", [])],
            )
            for chunk in payload.get("chunks", [])
        ]

    def _resolve_source_path(self, path: Path) -> Path:
        candidate = (self.workspace_root / path).resolve()
        candidate.relative_to(self.workspace_root)
        if not candidate.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        return candidate

    def _resolve_path(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else (self.workspace_root / path)
        return candidate.resolve()

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        dot_product = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot_product / (left_norm * right_norm)

    def _build_citation(self, chunk: _ChunkRecord) -> str:
        citation_start = chunk.heading_line or chunk.line_start
        if citation_start == chunk.line_end:
            return f"{chunk.source_path}#L{citation_start}"
        return f"{chunk.source_path}#L{citation_start}-L{chunk.line_end}"


def _make_excerpt(text: str, limit: int = 240) -> str:
    excerpt = " ".join(text.split())
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: limit - 1].rstrip() + "…"