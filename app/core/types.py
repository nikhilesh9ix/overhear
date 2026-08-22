"""Typed I/O for every stage. Nothing crosses a stage boundary as a loose dict."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    STT_UNAVAILABLE = "stt_unavailable"
    STT_LOW_CONFIDENCE = "stt_low_confidence"
    EMPTY_TRANSCRIPT = "empty_transcript"
    RETRIEVAL_EMPTY = "retrieval_empty"
    RETRIEVAL_LOW_CONFIDENCE = "retrieval_low_confidence"
    OFF_TOPIC = "off_topic"
    UNSAFE_INPUT = "unsafe_input"
    NOT_GROUNDED = "not_grounded"
    PROVIDER_ERROR = "provider_error"
    ALL_PROVIDERS_DOWN = "all_providers_down"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    INTERNAL = "internal"


class StageError(Exception):
    """Structured failure. Carries whether a retry could plausibly help."""

    def __init__(self, code: ErrorCode, message: str, *, retryable: bool = False,
                 detail: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message,
                "retryable": self.retryable, "detail": self.detail}


class TranscriptIn(BaseModel):
    text: str
    is_final: bool
    utterance_idx: int = 0
    language: str | None = None
    language_confidence: float | None = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    doc_id: int
    text: str
    score: float
    passage_idxs: list[int] = Field(default_factory=list)


class RetrievalOut(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    top_score: float
    score_gap: float = Field(
        description="top1 - mean(top2..k). A flat distribution means the index has "
                    "nothing distinctly relevant, which is the refusal signal."
    )
    speculative: bool = False
    cache_hit: bool = False
    elapsed_ms: float = 0.0


class GuardVerdict(BaseModel):
    allowed: bool
    code: ErrorCode | None = None
    user_message: str | None = None
    detail: dict = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: int
    doc_id: int


class AnswerOut(BaseModel):
    """The generation contract. `grounded` is asked of the model directly so a
    refusal is a first-class outcome rather than something we regex out of prose."""

    answer: str
    grounded: bool
    citations: list[Citation] = Field(default_factory=list)
    provider: str = ""
    model: str = ""


class Timings(BaseModel):
    """Every number the README reports. Kept in one place so the benchmark and the
    live trace cannot drift apart."""

    t_stt_ms: float | None = None  # first audio byte -> final transcript
    t_speculation_ms: float | None = None  # embed+search done before end of speech
    t_retrieval_ms: float | None = None  # query-time retrieval on the critical path
    t1_ms: float | None = None  # end of speech -> first generated token
    t2_ms: float | None = None  # end of speech -> last generated token
    provider_rtt_ms: float | None = None  # network RTT to the LLM provider
    cache_hit: bool = False


class TraceEvent(BaseModel):
    kind: Literal[
        "session", "partial", "final", "speculate", "retrieval",
        "guard", "token", "answer", "error", "timings",
    ]
    at_ms: float
    data: dict = Field(default_factory=dict)
