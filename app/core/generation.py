"""Streaming generation with a Groq primary and a Cerebras fallback.

One persistent HTTP/2 client, pre-warmed at startup, because on a 200ms budget
a cold TLS handshake is most of the budget. Both providers speak the OpenAI
chat-completions shape, so they sit behind one interface and the circuit breaker
treats them identically.

The model is asked for JSON with an explicit `grounded` flag. That makes refusal
a first-class output rather than something we try to detect in prose afterwards.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from app.config import settings
from app.core.harness import CircuitBreaker
from app.core.types import AnswerOut, Citation, ErrorCode, RetrievedChunk, StageError

log = logging.getLogger("overhear.generation")

SYSTEM = """You answer questions using ONLY the numbered passages provided.

Rules:
- If the passages contain the answer, answer in one or two sentences and cite the
  passage numbers you used.
- If the passages do NOT contain the answer, you MUST set grounded to false and
  say what you would have needed. Never guess, never use outside knowledge.
- Do not mention "passages" or "context" in the answer text itself.

Respond with JSON only, no markdown fence:
{"answer": "...", "grounded": true|false, "citations": [1, 2]}"""


@dataclass
class Provider:
    name: str
    url: str
    api_key: str
    model: str
    breaker: CircuitBreaker


def build_providers() -> list[Provider]:
    out = []
    if settings.groq_api_key:
        out.append(Provider("groq", settings.groq_url, settings.groq_api_key,
                            settings.groq_model, CircuitBreaker("groq")))
    if settings.cerebras_api_key:
        out.append(Provider("cerebras", settings.cerebras_url, settings.cerebras_api_key,
                            settings.cerebras_model, CircuitBreaker("cerebras")))
    return out


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    ctx = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, start=1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {query}"},
    ]


class Generator:
    def __init__(self) -> None:
        self.providers = build_providers()
        self._client: httpx.AsyncClient | None = None
        self.warm_rtt_ms: dict[str, float] = {}

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=8, keepalive_expiry=300.0),
        )
        for p in self.providers:
            try:
                self.warm_rtt_ms[p.name] = await self._warm(p)
                log.info("warmed %s: %.1fms rtt", p.name, self.warm_rtt_ms[p.name])
            except Exception as e:
                log.warning("warm %s failed: %s", p.name, e)

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _warm(self, p: Provider) -> float:
        """Open the connection and measure a real round trip, reported in the README
        as its own line so the deploy region's network cost is visible."""
        assert self._client is not None
        t0 = time.perf_counter()
        r = await self._client.post(
            p.url,
            headers={"Authorization": f"Bearer {p.api_key}"},
            json={"model": p.model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "stream": False},
        )
        r.raise_for_status()
        return (time.perf_counter() - t0) * 1000

    async def stream(self, query: str, chunks: list[RetrievedChunk]) -> AsyncIterator[dict]:
        """Yields {"type":"token","text":...} then {"type":"answer","answer":AnswerOut}.

        Tries each provider in order; a provider whose circuit is open is skipped.
        A failure after the first token has already been emitted is not retried on
        another provider, because the client has already seen partial output.
        """
        if not self.providers:
            raise StageError(ErrorCode.ALL_PROVIDERS_DOWN,
                             "No LLM provider configured. Set GROQ_API_KEY or CEREBRAS_API_KEY.")
        messages = build_prompt(query, chunks)
        errors: list[str] = []

        for p in self.providers:
            if p.breaker.is_open:
                errors.append(f"{p.name}: circuit open")
                continue
            emitted = False
            buf: list[str] = []
            try:
                async for tok in self._stream_one(p, messages):
                    emitted = True
                    buf.append(tok)
                    yield {"type": "token", "text": tok}
                p.breaker.record_success()
                yield {"type": "answer",
                       "answer": _parse("".join(buf), chunks, p.name, p.model)}
                return
            except Exception as e:
                p.breaker.record_failure()
                errors.append(f"{p.name}: {type(e).__name__}: {e}")
                log.warning("provider %s failed: %s", p.name, e)
                if emitted:
                    raise StageError(
                        ErrorCode.PROVIDER_ERROR,
                        f"{p.name} failed mid-stream after emitting tokens",
                        detail={"errors": errors},
                    ) from e

        raise StageError(ErrorCode.ALL_PROVIDERS_DOWN,
                         "every generation provider failed",
                         detail={"errors": errors})

    async def _stream_one(self, p: Provider, messages: list[dict]) -> AsyncIterator[str]:
        assert self._client is not None
        body = {
            "model": p.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"},
        }
        # qwen3.6 streams a <think> block before any answer content unless reasoning
        # is switched off. Left on, the "first token" we would time is a reasoning
        # token, which would make T1 a lie: measured 127ms to first token but 602ms
        # to a usable answer. With reasoning off it is 108ms and 149ms.
        if "qwen" in p.model.lower():
            body["reasoning_effort"] = "none"
        async with self._client.stream(
            "POST", p.url,
            headers={"Authorization": f"Bearer {p.api_key}"},
            json=body,
        ) as r:
            if r.status_code != 200:
                detail = (await r.aread()).decode()[:300]
                raise StageError(ErrorCode.PROVIDER_ERROR,
                                 f"{p.name} HTTP {r.status_code}: {detail}",
                                 retryable=r.status_code in (429, 500, 502, 503, 529))
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                choices = d.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta


def _parse(text: str, chunks: list[RetrievedChunk], provider: str, model: str) -> AnswerOut:
    """Parse the model's JSON. A model that returns unparseable output is treated as
    ungrounded rather than having its raw text passed through as an answer."""
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1] if "```" in raw[3:] else raw.lstrip("`")
        raw = raw.removeprefix("json").strip()
    try:
        d = json.loads(raw)
    except Exception:
        log.warning("ungrounded: unparseable generation output: %r", text[:200])
        return AnswerOut(
            answer="I could not produce a reliable answer from the retrieved passages.",
            grounded=False, citations=[], provider=provider, model=model,
        )

    cites: list[Citation] = []
    for c in d.get("citations") or []:
        try:
            i = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(chunks):
            ch = chunks[i - 1]
            cites.append(Citation(chunk_id=ch.chunk_id, doc_id=ch.doc_id))

    return AnswerOut(
        answer=str(d.get("answer", "")).strip(),
        grounded=bool(d.get("grounded", False)),
        citations=cites,
        provider=provider,
        model=model,
    )
