"""Synthesize spoken versions of golden queries with Sarvam TTS for the voice benchmark.

These are real audio files going through the real STT path -- not mocks. Using TTS
rather than human recordings is a stated limitation: it removes accent, background
noise and disfluency, so voice-mode T-STT here is a best case. Human recordings
dropped into bench/audio/ will be picked up the same way and are strictly better.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import sys
import wave

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import DATA, ROOT, settings  # noqa: E402

OUT = ROOT / "bench" / "audio"
TTS_URL = "https://api.sarvam.ai/text-to-speech"


def to_16k_mono(raw: bytes) -> bytes:
    """Sarvam returns a WAV; re-emit as 16kHz mono 16-bit, which is what /ws expects."""
    with wave.open(io.BytesIO(raw), "rb") as w:
        n, sw, ch, sr = w.getnframes(), w.getsampwidth(), w.getnchannels(), w.getframerate()
        pcm = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"expected 16-bit from TTS, got {sw*8}-bit")
    if ch == 2:  # downmix
        import array
        a = array.array("h", pcm)
        pcm = array.array("h", [(a[i] + a[i + 1]) // 2 for i in range(0, len(a), 2)]).tobytes()
    if sr != 16000:
        import array
        a = array.array("h", pcm)
        ratio = sr / 16000
        out = array.array("h", [a[min(int(i * ratio), len(a) - 1)] for i in range(int(len(a) / ratio))])
        pcm = out.tobytes()
    return pcm


def write_wav(path: pathlib.Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--lang", default="en-IN")
    ap.add_argument("--speaker", default="anushka")
    ap.add_argument("--model", default="bulbul:v2")
    args = ap.parse_args()

    if not settings.sarvam_api_key:
        print("!! SARVAM_API_KEY not set", file=sys.stderr)
        return 2

    golden = json.loads((DATA / "golden_queries.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    made = 0
    with httpx.Client(timeout=60.0) as c:
        for i, g in enumerate(golden):
            if made >= args.n:
                break
            text = g["eng_query"].strip().lstrip(". ").strip()
            if len(text) < 15 or len(text) > 180:
                continue
            dest = OUT / f"q{made:02d}_{g['query_id']}.wav"
            if dest.exists():
                made += 1
                continue
            try:
                r = c.post(
                    TTS_URL,
                    headers={"API-Subscription-Key": settings.sarvam_api_key},
                    json={"text": text, "target_language_code": args.lang,
                          "speaker": args.speaker, "model": args.model},
                )
                if r.status_code != 200:
                    print(f"  [{made}] HTTP {r.status_code}: {r.text[:200]}", flush=True)
                    if r.status_code in (401, 403):
                        return 1
                    continue
                audios = r.json().get("audios") or []
                if not audios:
                    print(f"  [{made}] no audio returned", flush=True)
                    continue
                pcm = to_16k_mono(base64.b64decode(audios[0]))
            except Exception as e:
                print(f"  [{made}] {type(e).__name__}: {e}", flush=True)
                continue
            write_wav(dest, pcm)
            dur = len(pcm) / (16000 * 2)
            print(f"  [{made}] {dur:4.1f}s  {dest.name}  \"{text[:60]}\"", flush=True)
            made += 1

    print(f"\n{made} WAVs in {OUT}")
    (OUT / "MANIFEST.txt").write_text(
        "Synthesized with Sarvam TTS from MSMARCO-XI golden queries.\n"
        "Real audio through the real STT path, but TTS-clean: no accent variation,\n"
        "background noise or disfluency. Voice-mode T-STT from these is a best case.\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
