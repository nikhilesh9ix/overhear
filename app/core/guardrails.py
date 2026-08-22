"""Three gates. Every refusal names the specific reason -- a generic error is not
a guardrail, it's a bug report aimed at the user.

Gate 1 (pre-retrieval): the transcript is unusable, or the input is unsafe.
Gate 2 (post-retrieval): the index has nothing convincingly relevant.
Gate 3 (post-generation): the model itself says it wasn't grounded, or its
        citations don't hold up.
"""
from __future__ import annotations

import re

from app.core.types import AnswerOut, ErrorCode, GuardVerdict, RetrievalOut, TranscriptIn

# Deliberately narrow. This catches requests to produce harm, not merely
# unpleasant topics -- a QA system over MS MARCO will legitimately be asked about
# disease, weapons history, drugs and crime, and refusing those would be wrong.
UNSAFE_PATTERNS = [
    re.compile(r"\b(how (do|to|can) (i|we|you) )?(make|build|synthesi[sz]e|manufacture)\b.{0,40}"
               r"\b(bomb|explosive|nerve agent|sarin|ricin|meth(amphetamine)?|fentanyl)\b", re.I),
    re.compile(r"\bhow to\b.{0,30}\b(kill|murder|poison)\b.{0,20}\b(someone|person|people|him|her|them)\b", re.I),
    re.compile(r"\b(child (porn|sexual abuse)|csam)\b", re.I),
]

MIN_QUERY_CHARS = 8
MIN_LANG_CONFIDENCE = 0.35
MIN_TOP_SCORE = 0.55
MIN_SCORE_GAP = 0.015


def check_input(t: TranscriptIn) -> GuardVerdict:
    text = t.text.strip()

    if not text or len(text) < MIN_QUERY_CHARS:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.EMPTY_TRANSCRIPT,
            user_message="I only caught a word or two. Could you ask that again, "
                         "a little closer to the mic?",
            detail={"chars": len(text)},
        )

    # Sarvam does not expose a per-token acoustic confidence on transcript.final;
    # language_confidence is the closest real signal it gives us, so that is what
    # we gate on rather than inventing a score.
    if t.language_confidence is not None and t.language_confidence < MIN_LANG_CONFIDENCE:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.STT_LOW_CONFIDENCE,
            user_message="I couldn't make out which language that was, so I'd rather "
                         "not guess at the question. Could you repeat it?",
            detail={"language_confidence": t.language_confidence, "language": t.language},
        )

    for pat in UNSAFE_PATTERNS:
        if pat.search(text):
            return GuardVerdict(
                allowed=False,
                code=ErrorCode.UNSAFE_INPUT,
                user_message="I can't help with that one. Ask me something I can look "
                             "up in the passages instead.",
                detail={"matched": pat.pattern[:60]},
            )

    return GuardVerdict(allowed=True)


def check_retrieval(r: RetrievalOut) -> GuardVerdict:
    if not r.chunks:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.RETRIEVAL_EMPTY,
            user_message="I don't know — nothing in the indexed passages relates to that.",
            detail={"n_chunks": 0},
        )

    if r.top_score < MIN_TOP_SCORE:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.OFF_TOPIC,
            user_message=(
                f"I don't know. That looks outside what I've indexed — the closest "
                f"passage only scored {r.top_score:.2f}, and I'd rather say nothing "
                f"than answer from something that loosely rhymes with your question."
            ),
            detail={"top_score": round(r.top_score, 4), "threshold": MIN_TOP_SCORE},
        )

    # A flat top-k means everything matched equally weakly. That is the classic
    # shape of a query the corpus has no real answer for.
    if r.score_gap < MIN_SCORE_GAP:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.RETRIEVAL_LOW_CONFIDENCE,
            user_message=(
                "I don't know. Several passages matched your question about equally "
                "well, which usually means none of them actually answers it."
            ),
            detail={"score_gap": round(r.score_gap, 4), "threshold": MIN_SCORE_GAP,
                    "top_score": round(r.top_score, 4)},
        )

    return GuardVerdict(allowed=True)


def check_answer(a: AnswerOut, r: RetrievalOut) -> GuardVerdict:
    if not a.grounded:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.NOT_GROUNDED,
            user_message=a.answer or "I don't know — the passages I found don't answer that.",
            detail={"model_said_grounded": False, "provider": a.provider},
        )

    if not a.answer.strip():
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.NOT_GROUNDED,
            user_message="I don't know — I couldn't form an answer from those passages.",
            detail={"empty_answer": True},
        )

    # Claiming grounded while citing nothing is exactly the failure mode this gate
    # exists for.
    if not a.citations:
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.NOT_GROUNDED,
            user_message="I don't know — I couldn't point to a specific passage that "
                         "supports an answer, so I won't give you one.",
            detail={"claimed_grounded": True, "citations": 0},
        )

    valid_ids = {c.chunk_id for c in r.chunks}
    if not any(c.chunk_id in valid_ids for c in a.citations):
        return GuardVerdict(
            allowed=False,
            code=ErrorCode.NOT_GROUNDED,
            user_message="I don't know — the support I found doesn't actually match "
                         "the passages I retrieved.",
            detail={"cited": [c.chunk_id for c in a.citations]},
        )

    return GuardVerdict(allowed=True)
