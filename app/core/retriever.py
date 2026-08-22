"""Speculative retrieval.

The idea: interim transcripts arrive while the user is still talking, so embed
and search on those partials instead of waiting. By the time speech ends, the
candidate set is already warm and the post-speech critical path is generation
alone.

Mechanics:
- Debounce partials so we don't embed on every keystroke-sized update.
- Cancel an in-flight speculation when a newer partial supersedes it.
- Keep a rolling candidate set keyed by the text it was computed from.
- On final transcript, reuse the cached candidates when the final text matches
  what we speculated on; otherwise retrieve fresh and log the miss. Misses are
  counted and reported, not hidden -- a speculative system that silently falls
  back is just a slow system with extra steps.

Embedding is the expensive part here (~48ms on the build machine), which is why
the debounce exists at all.
"""
from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from app.config import settings
from app.core.embedder import Embedder
from app.core.index import HnswIndex
from app.core.types import RetrievalOut, RetrievedChunk

log = logging.getLogger("overhear.retriever")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .,?!")


def _score_gap(scores: list[float]) -> float:
    """top1 minus the mean of the rest. Near zero means the index found nothing
    distinctly better than anything else, which is the flat-distribution refusal
    signal in guardrails.py."""
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - float(np.mean(scores[1:])))


class SpeculativeRetriever:
    def __init__(self, index: HnswIndex, embedder: Embedder, top_k: int | None = None) -> None:
        self.index = index
        self.embedder = embedder
        self.top_k = top_k or settings.top_k
        self.debounce_s = settings.speculation_debounce_ms / 1000
        self.prefix_coverage = settings.speculation_prefix_coverage
        self.min_prefix_words = settings.speculation_min_prefix_words

        self._task: asyncio.Task | None = None
        self._pending_key: str | None = None
        self._cached_key: str | None = None
        self._cached: RetrievalOut | None = None
        self._last_partial_at = 0.0

        # stats, surfaced in the trace
        self.speculations = 0
        self.cancelled = 0
        self.hits = 0
        self.misses = 0

    # ---- blocking path -------------------------------------------------

    def _search_sync(self, query: str) -> RetrievalOut:
        t0 = time.perf_counter()
        vec = self.embedder.embed_query(query)
        hits = self.index.search(vec, k=self.top_k)
        chunks = [
            RetrievedChunk(
                chunk_id=p["chunk_id"], doc_id=p["doc_id"], text=p["text"],
                score=s, passage_idxs=p.get("passage_idxs", []),
            )
            for p, s in hits
        ]
        scores = [c.score for c in chunks]
        return RetrievalOut(
            query=query,
            chunks=chunks,
            top_score=scores[0] if scores else 0.0,
            score_gap=_score_gap(scores),
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )

    async def _search(self, query: str) -> RetrievalOut:
        # fastembed's ONNX call is blocking; keep it off the event loop so audio
        # forwarding and token streaming are not stalled by it.
        return await asyncio.to_thread(self._search_sync, query)

    # ---- speculative path ----------------------------------------------

    def on_partial(self, text: str) -> None:
        """Fire and forget. Called for every interim transcript."""
        key = _normalize(text)
        if len(key) < 8:  # too little signal to be worth an embed
            return
        if key == self._cached_key:
            return

        if self._task is not None and not self._task.done():
            self._task.cancel()
            self.cancelled += 1

        self._last_partial_at = time.perf_counter()
        self._pending_key = key
        self._task = asyncio.create_task(self._speculate(key, text))

    async def _speculate(self, key: str, text: str) -> None:
        try:
            await asyncio.sleep(self.debounce_s)
            out = await self._search(text)
            out.speculative = True
            self._cached_key = key
            self._cached = out
            self.speculations += 1
            log.debug("speculated on %r -> top=%.3f", text[:40], out.top_score)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a failed speculation must never break the session
            log.warning("speculation failed: %s: %s", type(e).__name__, e)

    def _key_accepts(self, cached: str | None, final_key: str) -> bool:
        if not cached:
            return False
        if cached == final_key:
            return True
        fw, cw = final_key.split(), cached.split()
        if len(cw) < self.min_prefix_words or len(cw) > len(fw):
            return False
        if fw[: len(cw)] != cw:
            return False
        return len(cw) / len(fw) >= self.prefix_coverage

    def _accepts(self, final_key: str) -> bool:
        """Is the cached speculation good enough to answer `final_key`?

        Exact match is far too strict. Sarvam's last interim before end-of-speech is
        usually a prefix of the final ("how many women did frank" vs "how many women
        did frank gifford marry"), and a prefix that already carries most of the
        query's content words retrieves the same neighbourhood. Requiring equality
        threw away 5 of 6 usable speculations in the first voice benchmark.

        We accept when the cached text is a word-prefix of the final and covers at
        least PREFIX_COVERAGE of its words. Whether that actually returns the same
        top-k is not assumed -- scripts/bench_speculation.py measures the agreement
        and the README reports it.
        """
        if self._cached is None:
            return False
        return self._key_accepts(self._cached_key, final_key)

    async def on_final(self, text: str) -> RetrievalOut:
        """The critical path. Returns immediately on a speculation hit."""
        key = _normalize(text)

        # Wait for an in-flight speculation only if it could actually satisfy this
        # final. Waiting unconditionally cost ~120ms on every miss -- we blocked on a
        # result we were then going to throw away.
        if (self._task is not None and not self._task.done()
                and self._key_accepts(self._pending_key, key)):
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=0.12)
            except (asyncio.TimeoutError, Exception):
                pass

        if self._accepts(key):
            self.hits += 1
            out = self._cached.model_copy(deep=True)
            out.cache_hit = True
            out.speculative = True
            out.elapsed_ms = 0.0  # already paid for, before end of speech
            exact = self._cached_key == key
            log.info("speculation HIT (%s) on %r  [spec: %r]",
                     "exact" if exact else "prefix", text[:50], (self._cached_key or "")[:50])
            return out

        self.misses += 1
        log.info("speculation MISS on %r (cached=%r)", text[:50], (self._cached_key or "")[:50])
        if self._task is not None and not self._task.done():
            self._task.cancel()
            self.cancelled += 1
        out = await self._search(text)
        out.cache_hit = False
        return out

    def reset(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None
        self._pending_key = None
        self._cached = None
        self._cached_key = None

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "speculations": self.speculations,
            "cancelled": self.cancelled,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else None,
        }
