"""Sarvam saaras:v3-realtime WebSocket client.

Protocol per https://docs.sarvam.ai/api-reference/speech-to-text/transcribe/realtime/ws

Client -> server: audio_input {event, audio:b64}, flush, end, ping
Server -> client: session.begin, vad.speech_start, vad.speech_end,
                  transcript.partial, transcript.final, error, session.end

We run in translate mode so Indic speech arrives as English text, matching the
English index (see Phase 0 commit). vad endpointing gives us end-of-speech for
free, which is the clock we measure T1 and T2 against.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import urlencode

import websockets

from app.config import settings
from app.core.types import ErrorCode, StageError

log = logging.getLogger("overhear.sarvam")


@dataclass
class SttEvent:
    kind: str  # partial | final | speech_start | speech_end | session_begin | error | end
    text: str = ""
    utterance_idx: int = 0
    language: str | None = None
    language_confidence: float | None = None
    confidence: float | None = None
    at_ms: float = 0.0
    raw: dict | None = None


def build_url() -> str:
    q = {
        "language_code": settings.sarvam_language,
        "model": settings.sarvam_model,
        "stream_type": settings.sarvam_stream_type,
        "mode": settings.sarvam_mode,
        "encoding": "linear16",
        "sample_rate": str(settings.sarvam_sample_rate),
        "endpointing": "vad",
        "threshold": str(settings.sarvam_vad_threshold),
        "silence_duration_ms": str(settings.sarvam_silence_duration_ms),
        "min_speech_duration_ms": str(settings.sarvam_min_speech_duration_ms),
    }
    return f"{settings.sarvam_ws_url}?{urlencode(q)}"


class SarvamStream:
    """One live transcription session. Audio in via send_audio, events out via events()."""

    def __init__(self) -> None:
        if not settings.sarvam_api_key:
            raise StageError(
                ErrorCode.STT_UNAVAILABLE,
                "SARVAM_API_KEY is not set. Real transcription is required; "
                "there is no offline fallback by design.",
                retryable=False,
            )
        self._ws: websockets.ClientConnection | None = None
        self._t0 = 0.0
        self._closed = False

    async def __aenter__(self) -> "SarvamStream":
        url = build_url()
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={"API-SUBSCRIPTION-KEY": settings.sarvam_api_key},
                    max_queue=64,
                    ping_interval=20,
                ),
                timeout=8.0,
            )
        except asyncio.TimeoutError as e:
            raise StageError(ErrorCode.STT_UNAVAILABLE,
                             "Sarvam connect timed out after 8s", retryable=True) from e
        except Exception as e:
            raise StageError(ErrorCode.STT_UNAVAILABLE,
                             f"Sarvam connect failed: {type(e).__name__}: {e}",
                             retryable=True) from e
        self._t0 = time.perf_counter()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    async def send_audio(self, pcm: bytes) -> None:
        """pcm must be 16-bit little-endian mono at settings.sarvam_sample_rate."""
        if self._ws is None or self._closed:
            return
        await self._ws.send(json.dumps({
            "event": "audio_input",
            "audio": base64.b64encode(pcm).decode(),
        }))

    async def flush(self) -> None:
        if self._ws is not None and not self._closed:
            await self._ws.send(json.dumps({"event": "flush"}))

    async def finish(self) -> None:
        if self._ws is not None and not self._closed:
            await self._ws.send(json.dumps({"event": "flush"}))
            await self._ws.send(json.dumps({"event": "end"}))

    async def close(self) -> None:
        self._closed = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def events(self) -> AsyncIterator[SttEvent]:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                at = self.elapsed_ms
                try:
                    m = json.loads(raw)
                except Exception:
                    continue
                ev = m.get("event")
                if ev == "transcript.partial":
                    yield SttEvent("partial", text=m.get("text", ""),
                                   utterance_idx=m.get("utterance_idx", 0),
                                   language=m.get("language"), at_ms=at, raw=m)
                elif ev == "transcript.final":
                    yield SttEvent("final", text=m.get("text", ""),
                                   utterance_idx=m.get("utterance_idx", 0),
                                   language=m.get("language"),
                                   language_confidence=m.get("language_confidence"),
                                   at_ms=at, raw=m)
                elif ev == "vad.speech_start":
                    yield SttEvent("speech_start", utterance_idx=m.get("utterance_idx", 0),
                                   confidence=m.get("confidence"), at_ms=at, raw=m)
                elif ev == "vad.speech_end":
                    yield SttEvent("speech_end", utterance_idx=m.get("utterance_idx", 0),
                                   confidence=m.get("confidence"), at_ms=at, raw=m)
                elif ev == "session.begin":
                    yield SttEvent("session_begin", at_ms=at, raw=m)
                elif ev == "error":
                    log.warning("sarvam error: %s", m)
                    yield SttEvent("error", text=m.get("message", ""), at_ms=at, raw=m)
                    if m.get("is_fatal"):
                        return
                elif ev == "session.end":
                    yield SttEvent("end", at_ms=at, raw=m)
                    return
        except websockets.ConnectionClosed:
            return
