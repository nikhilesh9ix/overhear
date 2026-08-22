"""The query-time path, from end-of-speech to the last generated token.

Latency definitions used throughout (also stated in the README):
  T-STT : first audio byte -> Sarvam's final transcript. Reported separately
          because the task's wording is ambiguous about whether STT counts.
  T1    : end of speech -> first generated token. This is the number the task
          asks for, and the one the 200ms target applies to.
  T2    : end of speech -> last generated token.

Retrieval usually contributes ~0ms to T1 because speculation already paid for it
before the user stopped talking. When speculation misses, it lands on T1 and the
trace says so.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator

from app.config import settings
from app.core import guardrails
from app.core.generation import Generator
from app.core.harness import Deadline
from app.core.retriever import SpeculativeRetriever
from app.core.types import (
    AnswerOut,
    ErrorCode,
    RetrievalOut,
    StageError,
    Timings,
    TranscriptIn,
)

log = logging.getLogger("overhear.pipeline")


async def answer_query(
    transcript: TranscriptIn,
    retriever: SpeculativeRetriever,
    generator: Generator,
    *,
    speech_end_t: float,
    stt_ms: float | None = None,
) -> AsyncIterator[dict]:
    """Yields protocol events. speech_end_t is a perf_counter() stamp taken when
    Sarvam reported vad.speech_end -- every T1/T2 number is measured from it."""
    deadline = Deadline(settings.request_deadline_ms)
    timings = Timings(t_stt_ms=stt_ms)

    def since_speech_end() -> float:
        return (time.perf_counter() - speech_end_t) * 1000

    # --- gate 1: is the question usable at all ---------------------------
    v = guardrails.check_input(transcript)
    if not v.allowed:
        timings.t1_ms = timings.t2_ms = since_speech_end()
        yield {"type": "refusal", "code": v.code.value if v.code else None,
               "message": v.user_message, "detail": v.detail, "gate": "input"}
        yield {"type": "timings", "timings": timings.model_dump()}
        return

    # --- retrieval (usually already done, speculatively) -----------------
    try:
        retrieval: RetrievalOut = await retriever.on_final(transcript.text)
    except StageError as e:
        yield {"type": "error", **e.to_dict()}
        return
    timings.cache_hit = retrieval.cache_hit
    timings.t_retrieval_ms = retrieval.elapsed_ms
    yield {
        "type": "retrieval",
        "cache_hit": retrieval.cache_hit,
        "elapsed_ms": round(retrieval.elapsed_ms, 2),
        "top_score": round(retrieval.top_score, 4),
        "score_gap": round(retrieval.score_gap, 4),
        "chunks": [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "score": round(c.score, 4), "preview": c.text[:180]}
            for c in retrieval.chunks
        ],
        "stats": retriever.stats(),
    }

    # --- gate 2: did we retrieve anything worth answering from ------------
    v = guardrails.check_retrieval(retrieval)
    if not v.allowed:
        timings.t1_ms = timings.t2_ms = since_speech_end()
        yield {"type": "refusal", "code": v.code.value if v.code else None,
               "message": v.user_message, "detail": v.detail, "gate": "retrieval"}
        yield {"type": "timings", "timings": timings.model_dump()}
        return

    # --- generation -------------------------------------------------------
    answer: AnswerOut | None = None
    try:
        deadline.check("generation", need_ms=200)
        async for ev in generator.stream(transcript.text, retrieval.chunks):
            if ev["type"] == "token":
                if timings.t1_ms is None:
                    timings.t1_ms = since_speech_end()
                    yield {"type": "first_token", "t1_ms": round(timings.t1_ms, 1)}
                yield {"type": "token", "text": ev["text"]}
            elif ev["type"] == "answer":
                answer = ev["answer"]
    except StageError as e:
        yield {"type": "error", **e.to_dict()}
        yield {"type": "timings", "timings": timings.model_dump()}
        return

    timings.t2_ms = since_speech_end()
    if generator.warm_rtt_ms:
        timings.provider_rtt_ms = min(generator.warm_rtt_ms.values())

    if answer is None:
        yield {"type": "error", "code": ErrorCode.PROVIDER_ERROR.value,
               "message": "generation produced no parseable answer"}
        yield {"type": "timings", "timings": timings.model_dump()}
        return

    # --- gate 3: does the model stand behind what it just said -------------
    v = guardrails.check_answer(answer, retrieval)
    if not v.allowed:
        yield {"type": "refusal", "code": v.code.value if v.code else None,
               "message": v.user_message, "detail": v.detail, "gate": "groundedness"}
        yield {"type": "timings", "timings": timings.model_dump()}
        return

    yield {"type": "answer", "answer": answer.model_dump()}
    yield {"type": "timings", "timings": timings.model_dump()}
