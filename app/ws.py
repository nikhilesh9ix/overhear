"""The single WebSocket endpoint.

Browser -> us : binary frames of 16kHz mono PCM16, plus small JSON control frames.
us -> browser : JSON trace events (partials, retrieval, tokens, refusals, timings).

We hold two sockets per session: the browser's and Sarvam's. Audio is forwarded
as it arrives; Sarvam's interim transcripts drive speculative retrieval, and its
vad.speech_end starts the T1/T2 clock.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.retriever import SpeculativeRetriever
from app.core.sarvam import SarvamStream
from app.core.types import StageError, TranscriptIn
from app.pipeline import answer_query

log = logging.getLogger("overhear.ws")
router = APIRouter()


@router.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    app = ws.app.state

    if not getattr(app, "ready", False):
        await ws.send_json({"type": "error", "code": "warming_up",
                            "message": app.warmup_error
                            or "Server is still loading the model and index. Retry in a moment."})
        await ws.close()
        return

    if app.index is None:
        await ws.send_json({"type": "error", "code": "index_missing",
                            "message": "No index built. Run: make ingest"})
        await ws.close()
        return

    retriever = SpeculativeRetriever(app.index, app.embedder)
    session_t0 = time.perf_counter()

    try:
        async with SarvamStream() as stt:
            await ws.send_json({"type": "session", "message": "connected to Sarvam",
                                "index_size": len(app.index)})

            speech_end_t: float | None = None
            first_audio_t: float | None = None
            answering = False

            async def pump_browser_to_sarvam() -> None:
                nonlocal first_audio_t
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect()
                    if (data := msg.get("bytes")) is not None:
                        if first_audio_t is None:
                            first_audio_t = time.perf_counter()
                        await stt.send_audio(data)
                    elif (text := msg.get("text")) is not None:
                        import json as _json
                        try:
                            ctl = _json.loads(text)
                        except Exception:
                            continue
                        if ctl.get("event") == "stop":
                            await stt.finish()

            async def pump_sarvam_to_browser() -> None:
                nonlocal speech_end_t, answering
                async for ev in stt.events():
                    now_ms = (time.perf_counter() - session_t0) * 1000

                    if ev.kind == "partial":
                        await ws.send_json({"type": "partial", "text": ev.text,
                                            "at_ms": round(now_ms, 1)})
                        # This is the whole trick: retrieval starts here, mid-sentence.
                        retriever.on_partial(ev.text)

                    elif ev.kind == "speech_start":
                        await ws.send_json({"type": "speech_start",
                                            "at_ms": round(now_ms, 1)})

                    elif ev.kind == "speech_end":
                        speech_end_t = time.perf_counter()
                        await ws.send_json({"type": "speech_end",
                                            "at_ms": round(now_ms, 1)})

                    elif ev.kind == "final":
                        if answering:
                            continue
                        answering = True
                        stt_ms = ((time.perf_counter() - first_audio_t) * 1000
                                  if first_audio_t else None)
                        # If VAD never fired, fall back to now so the numbers are
                        # still measured from something real rather than dropped.
                        anchor = speech_end_t or time.perf_counter()
                        await ws.send_json({"type": "final", "text": ev.text,
                                            "language": ev.language,
                                            "language_confidence": ev.language_confidence,
                                            "t_stt_ms": round(stt_ms, 1) if stt_ms else None,
                                            "vad_anchored": speech_end_t is not None})
                        t = TranscriptIn(
                            text=ev.text, is_final=True,
                            utterance_idx=ev.utterance_idx,
                            language=ev.language,
                            language_confidence=ev.language_confidence,
                        )
                        async for out in answer_query(
                            t, retriever, app.generator,
                            speech_end_t=anchor, stt_ms=stt_ms,
                        ):
                            await ws.send_json(out)
                        retriever.reset()
                        answering = False
                        speech_end_t = None

                    elif ev.kind == "error":
                        await ws.send_json({"type": "error", "code": "stt_error",
                                            "message": ev.text, "detail": ev.raw})

                    elif ev.kind == "end":
                        await ws.send_json({"type": "stt_end"})
                        return

            a = asyncio.create_task(pump_browser_to_sarvam())
            b = asyncio.create_task(pump_sarvam_to_browser())
            done, pending = await asyncio.wait({a, b}, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc

    except WebSocketDisconnect:
        log.info("browser disconnected")
    except StageError as e:
        log.warning("session failed: %s", e.message)
        try:
            await ws.send_json({"type": "error", **e.to_dict()})
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001
        log.exception("ws session crashed")
        try:
            await ws.send_json({"type": "error", "code": "internal",
                                "message": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    finally:
        retriever.reset()
        try:
            await ws.close()
        except Exception:
            pass
