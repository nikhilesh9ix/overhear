"""Chunking strategies over the MSMARCO-XI passage stream.

MS MARCO passages are already retrieval-sized (83-473 chars, mean ~290 in the
Hindi validation split), so "chunking" here is not about cutting up long
documents. Each query_id contributes ~10 passages that share a topic; we treat
that group as one document and let each strategy decide where the retrieval
units fall inside it. That is a real decision with real recall consequences:
a fixed window straddles passage boundaries and dilutes a gold passage, while a
structural split preserves it intact.

Every strategy returns Chunks that record which source passages they cover, so
recall@k can be scored against the dataset's own is_selected labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np


@dataclass(slots=True)
class Passage:
    """One MS MARCO passage, with its gold label."""

    doc_id: int  # query_id it was retrieved for
    passage_idx: int  # position within that query's passage list
    text: str
    is_selected: bool


@dataclass(slots=True)
class Chunk:
    """A retrieval unit produced by a chunking strategy."""

    chunk_id: int
    doc_id: int
    text: str
    passage_idxs: tuple[int, ...]  # source passages this chunk draws from
    strategy: str
    meta: dict = field(default_factory=dict)


class ChunkStrategy(Protocol):
    name: str

    def chunk(self, doc_id: int, passages: list[Passage]) -> list[Chunk]: ...


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class FixedChunker:
    """Baseline: fixed character window with overlap, blind to passage edges.

    Deliberately ignores structure so it can serve as the control. It
    concatenates the document and slides a window, which means a gold passage
    can end up split across two chunks or diluted by a neighbour.
    """

    name = "fixed"

    def __init__(self, size: int = 512, overlap: int = 96) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc_id: int, passages: list[Passage]) -> list[Chunk]:
        # Track passage spans in the concatenated text so chunks stay attributable.
        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []  # (start, end, passage_idx)
        cursor = 0
        sep = " "
        for p in passages:
            if not p.text.strip():
                continue
            start = cursor
            parts.append(p.text)
            cursor += len(p.text)
            spans.append((start, cursor, p.passage_idx))
            parts.append(sep)
            cursor += len(sep)
        joined = "".join(parts).strip()
        if not joined:
            return []

        out: list[Chunk] = []
        step = self.size - self.overlap
        for start in range(0, len(joined), step):
            end = min(start + self.size, len(joined))
            text = joined[start:end].strip()
            if text:
                covered = tuple(pi for (s, e, pi) in spans if s < end and e > start)
                out.append(
                    Chunk(
                        chunk_id=-1,
                        doc_id=doc_id,
                        text=text,
                        passage_idxs=covered,
                        strategy=self.name,
                        meta={"char_start": start, "char_end": end},
                    )
                )
            if end == len(joined):
                break
        return out


class RecursiveChunker:
    """Structural: respect the dataset's own boundaries, then sentences.

    MS MARCO's natural unit is the passage, so we start there and only subdivide
    a passage on sentence boundaries when it exceeds max_chars. Short adjacent
    passages are packed together up to target_chars so we don't emit a swarm of
    tiny vectors, but packing never splits a passage mid-sentence.
    """

    name = "recursive"

    def __init__(self, target_chars: int = 480, max_chars: int = 700) -> None:
        self.target_chars = target_chars
        self.max_chars = max_chars

    def chunk(self, doc_id: int, passages: list[Passage]) -> list[Chunk]:
        units: list[tuple[str, int]] = []  # (text, passage_idx)
        for p in passages:
            t = p.text.strip()
            if not t:
                continue
            if len(t) <= self.max_chars:
                units.append((t, p.passage_idx))
                continue
            # Too long: split on sentences, greedily refilling to target.
            buf = ""
            for s in _sentences(t):
                if buf and len(buf) + 1 + len(s) > self.target_chars:
                    units.append((buf, p.passage_idx))
                    buf = s
                else:
                    buf = f"{buf} {s}".strip()
            if buf:
                units.append((buf, p.passage_idx))

        # Pack small neighbouring units, never splitting one.
        out: list[Chunk] = []
        buf_text = ""
        buf_idxs: list[int] = []
        for text, pidx in units:
            if buf_text and len(buf_text) + 1 + len(text) > self.target_chars:
                out.append(self._emit(doc_id, buf_text, buf_idxs))
                buf_text, buf_idxs = text, [pidx]
            else:
                buf_text = f"{buf_text} {text}".strip()
                buf_idxs.append(pidx)
        if buf_text:
            out.append(self._emit(doc_id, buf_text, buf_idxs))
        return out

    def _emit(self, doc_id: int, text: str, idxs: list[int]) -> Chunk:
        return Chunk(
            chunk_id=-1,
            doc_id=doc_id,
            text=text,
            passage_idxs=tuple(dict.fromkeys(idxs)),
            strategy=self.name,
            meta={"n_units": len(idxs)},
        )


class SemanticChunker:
    """Embedding-similarity boundary detection over the passage sequence.

    Embeds each passage, then walks the sequence and cuts wherever the cosine
    similarity between consecutive passages drops below a percentile-derived
    threshold. Topically coherent runs merge into one chunk; a topic shift forces
    a boundary. The percentile keeps this adaptive per document instead of
    hard-coding a similarity that only suits one topic mix.

    This is the only strategy that needs an embedder, and that cost is paid once
    at ingest, never at query time.
    """

    name = "semantic"

    def __init__(self, embed_fn, percentile: float = 25.0, max_chars: int = 900) -> None:
        self.embed_fn = embed_fn
        self.percentile = percentile
        self.max_chars = max_chars

    def chunk(self, doc_id: int, passages: list[Passage]) -> list[Chunk]:
        live = [p for p in passages if p.text.strip()]
        if not live:
            return []
        if len(live) == 1:
            return [self._emit(doc_id, live)]

        vecs = np.asarray(self.embed_fn([p.text for p in live]), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        sims = np.einsum("ij,ij->i", vecs[:-1], vecs[1:])
        threshold = float(np.percentile(sims, self.percentile))

        out: list[Chunk] = []
        group: list[Passage] = [live[0]]
        for i, p in enumerate(live[1:]):
            merged_len = sum(len(g.text) for g in group) + 1 + len(p.text)
            if sims[i] < threshold or merged_len > self.max_chars:
                out.append(self._emit(doc_id, group))
                group = [p]
            else:
                group.append(p)
        if group:
            out.append(self._emit(doc_id, group))
        return out

    def _emit(self, doc_id: int, group: list[Passage]) -> Chunk:
        return Chunk(
            chunk_id=-1,
            doc_id=doc_id,
            text=" ".join(g.text.strip() for g in group),
            passage_idxs=tuple(g.passage_idx for g in group),
            strategy=self.name,
            meta={"n_passages": len(group)},
        )


def assign_ids(chunks: Iterable[Chunk]) -> list[Chunk]:
    out = list(chunks)
    for i, c in enumerate(out):
        c.chunk_id = i
    return out
