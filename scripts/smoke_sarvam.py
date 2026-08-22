"""Smoke test 1: Sarvam saaras:v3-realtime interim transcripts.

Streams a WAV file in real-time-paced chunks and prints EVERY server event with a
timestamp relative to stream start, so we can prove partials arrive DURING audio.

Usage:  python scripts/smoke_sarvam.py path/to/16k_mono.wav
"""
import asyncio
import base64
import json
import sys
import time
import wave
from urllib.parse import urlencode

import websockets

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

CHUNK_MS = 100


def load_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, f"need mono, got {w.getnchannels()}ch"
        assert w.getsampwidth() == 2, f"need 16-bit, got {w.getsampwidth()*8}-bit"
        assert w.getframerate() == settings.sarvam_sample_rate, (
            f"need {settings.sarvam_sample_rate}Hz, got {w.getframerate()}"
        )
        return w.readframes(w.getnframes())


def ws_url() -> str:
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


async def main(path: str) -> int:
    if not settings.sarvam_api_key:
        print("!! SARVAM_API_KEY not set in .env", file=sys.stderr)
        return 2

    pcm = load_pcm(path)
    bytes_per_chunk = int(settings.sarvam_sample_rate * 2 * CHUNK_MS / 1000)
    audio_s = len(pcm) / (settings.sarvam_sample_rate * 2)
    print(f"audio: {path}  {audio_s:.2f}s  {len(pcm)}B  chunk={CHUNK_MS}ms", flush=True)

    url = ws_url()
    print(f"connect: {url}", flush=True)
    t_conn = time.perf_counter()

    async with websockets.connect(
        url, additional_headers={"API-SUBSCRIPTION-KEY": settings.sarvam_api_key}
    ) as ws:
        t0 = time.perf_counter()
        print(f"[{(t0-t_conn)*1000:7.1f}ms] connected", flush=True)

        partials: list[tuple[float, str]] = []
        done = asyncio.Event()

        async def rx() -> None:
            async for raw in ws:
                dt = (time.perf_counter() - t0) * 1000
                try:
                    m = json.loads(raw)
                except Exception:
                    print(f"[{dt:7.1f}ms] <non-json> {raw[:200]!r}", flush=True)
                    continue
                ev = m.get("event")
                if ev == "transcript.partial":
                    partials.append((dt, m.get("text", "")))
                    print(f"[{dt:7.1f}ms] PARTIAL  u{m.get('utterance_idx')}  {m.get('text','')!r}", flush=True)
                elif ev == "transcript.final":
                    print(f"[{dt:7.1f}ms] FINAL    u{m.get('utterance_idx')}  {m.get('text','')!r} "
                          f"lang={m.get('language')} conf={m.get('language_confidence')}", flush=True)
                elif ev in ("vad.speech_start", "vad.speech_end"):
                    print(f"[{dt:7.1f}ms] {ev}  u{m.get('utterance_idx')} conf={m.get('confidence')}", flush=True)
                elif ev == "session.end":
                    print(f"[{dt:7.1f}ms] session.end {json.dumps(m)[:300]}", flush=True)
                    done.set()
                    return
                elif ev == "error":
                    print(f"[{dt:7.1f}ms] ERROR {json.dumps(m)}", flush=True)
                    if m.get("is_fatal"):
                        done.set()
                        return
                else:
                    print(f"[{dt:7.1f}ms] {ev} {json.dumps(m)[:300]}", flush=True)

        rx_task = asyncio.create_task(rx())

        # stream at wall-clock pace so partials must interleave with audio
        sent = 0
        for off in range(0, len(pcm), bytes_per_chunk):
            chunk = pcm[off : off + bytes_per_chunk]
            await ws.send(json.dumps({
                "event": "audio_input",
                "audio": base64.b64encode(chunk).decode(),
            }))
            sent += 1
            target = t0 + sent * (CHUNK_MS / 1000)
            slack = target - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)
        t_sent = (time.perf_counter() - t0) * 1000
        print(f"[{t_sent:7.1f}ms] all audio sent ({sent} chunks); flushing", flush=True)
        await ws.send(json.dumps({"event": "flush"}))
        await ws.send(json.dumps({"event": "end"}))

        try:
            await asyncio.wait_for(done.wait(), timeout=15)
        except asyncio.TimeoutError:
            print("!! timed out waiting for session.end", flush=True)
        rx_task.cancel()

        during = [p for p in partials if p[0] < t_sent]
        print("\n===== VERDICT =====", flush=True)
        print(f"partials total: {len(partials)}   during-audio: {len(during)}", flush=True)
        if during:
            print(f"first partial at {during[0][0]:.1f}ms (audio ends ~{t_sent:.1f}ms)  -> "
                  "TRUE INTERIM CONFIRMED, speculative retrieval is viable", flush=True)
            return 0
        print("!! NO partials during audio -> speculative retrieval architecture is DEAD, escalate", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(asyncio.run(main(sys.argv[1])))
