"""Latency benchmark: P50 / P70 / P100 for T1, T2 and T-STT.

Metric definitions (same ones the README states):
  T-STT : first audio byte -> Sarvam's final transcript.
  T1    : end of speech -> first generated token.  <- the task's <200ms target
  T2    : end of speech -> last generated token.

Two modes:

  --mode text  (default)
      Drives the same pipeline through the text path over N golden queries from
      the dataset. T-STT is absent by construction and reported as such. This
      measures everything the 200ms target actually covers: retrieval on the
      critical path plus generation.

  --mode voice
      Requires WAV files (16kHz mono PCM16) in bench/audio/. Streams each through
      the real WebSocket at wall-clock pace, so T-STT is real and speculation is
      exercised exactly as in the demo. This is the honest end-to-end number.

Nothing here is simulated. If a run fails it is recorded as a failure and
excluded from percentiles, and the failure count is printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time
import wave

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import websockets  # noqa: E402

from app.config import DATA, ROOT  # noqa: E402

AUDIO_DIR = ROOT / "bench" / "audio"


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def run_text(base: str, queries: list[str], pace_s: float = 0.0) -> list[dict]:
    out = []
    async with httpx.AsyncClient(timeout=60.0) as c:
        for i, q in enumerate(queries, 1):
            if pace_s and i > 1:
                await asyncio.sleep(pace_s)
            t0 = time.perf_counter()
            try:
                r = await c.post(f"{base}/ask", json={"query": q})
                r.raise_for_status()
                d = r.json()
            except Exception as e:
                print(f"  [{i}/{len(queries)}] FAILED {type(e).__name__}: {e}", flush=True)
                out.append({"query": q, "ok": False, "error": str(e)})
                continue
            wall = (time.perf_counter() - t0) * 1000
            rec = {"query": q, "ok": True, "wall_ms": round(wall, 1)}
            for ev in d.get("events", []):
                if ev["type"] == "timings":
                    rec.update({k: v for k, v in ev["timings"].items() if v is not None})
                elif ev["type"] == "retrieval":
                    rec["top_score"] = ev["top_score"]
                    rec["cache_hit"] = ev["cache_hit"]
                elif ev["type"] == "refusal":
                    rec["refused"] = ev["code"]
                elif ev["type"] == "answer":
                    rec["grounded"] = ev["answer"]["grounded"]
                elif ev["type"] == "error":
                    rec["ok"] = False
                    rec["error"] = f"{ev.get('code')}: {ev.get('message')}"
            status = rec.get("refused") or ("ok" if rec.get("ok") else rec.get("error", "?"))
            print(f"  [{i}/{len(queries)}] T1={rec.get('t1_ms','—')} T2={rec.get('t2_ms','—')} "
                  f"{status} · {q[:52]}", flush=True)
            out.append(rec)
    return out


async def run_voice_one(ws_url: str, wav: pathlib.Path) -> dict:
    with wave.open(str(wav), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            return {"file": wav.name, "ok": False,
                    "error": f"need 16k mono 16-bit, got {w.getframerate()}Hz "
                             f"{w.getnchannels()}ch {w.getsampwidth()*8}bit"}
        pcm = w.readframes(w.getnframes())

    rec: dict = {"file": wav.name, "ok": False}
    chunk = int(16000 * 2 * 0.1)  # 100ms
    async with websockets.connect(ws_url, max_size=None) as ws:
        done = asyncio.Event()

        async def rx():
            async for raw in ws:
                m = json.loads(raw)
                t = m.get("type")
                if t == "final":
                    rec["text"] = m.get("text")
                    rec["t_stt_ms"] = m.get("t_stt_ms")
                elif t == "retrieval":
                    rec["cache_hit"] = m.get("cache_hit")
                    rec["top_score"] = m.get("top_score")
                elif t == "first_token":
                    rec["t1_ms"] = m.get("t1_ms")
                elif t == "refusal":
                    rec["refused"] = m.get("code")
                elif t == "answer":
                    rec["grounded"] = m["answer"]["grounded"]
                elif t == "timings":
                    for k, v in m["timings"].items():
                        if v is not None:
                            rec[k] = v
                    rec["ok"] = True
                    done.set()
                    return
                elif t == "error":
                    rec["error"] = f"{m.get('code')}: {m.get('message')}"
                    done.set()
                    return

        task = asyncio.create_task(rx())
        t0 = time.perf_counter()
        sent = 0
        for off in range(0, len(pcm), chunk):
            await ws.send(pcm[off:off + chunk])
            sent += 1
            slack = t0 + sent * 0.1 - time.perf_counter()
            if slack > 0:
                await asyncio.sleep(slack)
        await ws.send(json.dumps({"event": "stop"}))
        try:
            await asyncio.wait_for(done.wait(), timeout=30)
        except asyncio.TimeoutError:
            rec["error"] = "timeout waiting for answer"
        task.cancel()
    return rec


async def run_voice(base: str) -> list[dict]:
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        raise SystemExit(
            f"no WAVs in {AUDIO_DIR}. Record 20-30 spoken questions as 16kHz mono "
            f"16-bit WAV and drop them there, then rerun with --mode voice."
        )
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    out = []
    for i, wav in enumerate(wavs, 1):
        rec = await run_voice_one(ws_url, wav)
        print(f"  [{i}/{len(wavs)}] T-STT={rec.get('t_stt_ms','—')} T1={rec.get('t1_ms','—')} "
              f"T2={rec.get('t2_ms','—')} {rec.get('refused') or rec.get('error') or 'ok'} "
              f"· {wav.name}", flush=True)
        out.append(rec)
    return out


def summarize(records: list[dict], mode: str) -> dict:
    ok = [r for r in records if r.get("ok")]
    fields = ["t1_ms", "t2_ms", "t_stt_ms", "t_retrieval_ms", "provider_rtt_ms"]
    stats = {}
    for f in fields:
        vals = [r[f] for r in ok if isinstance(r.get(f), (int, float))]
        if not vals:
            continue
        stats[f] = {
            "n": len(vals),
            "p50": round(pct(vals, 50), 1),
            "p70": round(pct(vals, 70), 1),
            "p100": round(pct(vals, 100), 1),
            "mean": round(statistics.mean(vals), 1),
        }
    hits = [r for r in ok if r.get("cache_hit")]
    return {
        "mode": mode,
        "n_total": len(records),
        "n_ok": len(ok),
        "n_failed": len(records) - len(ok),
        "n_refused": len([r for r in records if r.get("refused")]),
        "speculation_hit_rate": round(len(hits) / len(ok), 3) if ok else None,
        "stats": stats,
    }


def print_table(s: dict) -> None:
    print("\n" + "=" * 74)
    print(f"{'metric':<26}{'n':>5}{'P50':>12}{'P70':>12}{'P100':>12}{'mean':>12}")
    print("-" * 74)
    labels = {
        "t1_ms": "T1 first token",
        "t2_ms": "T2 complete answer",
        "t_stt_ms": "T-STT (Sarvam)",
        "t_retrieval_ms": "retrieval on crit. path",
        "provider_rtt_ms": "LLM provider RTT",
    }
    for k, lab in labels.items():
        if k not in s["stats"]:
            continue
        v = s["stats"][k]
        print(f"{lab:<26}{v['n']:>5}{v['p50']:>12.1f}{v['p70']:>12.1f}"
              f"{v['p100']:>12.1f}{v['mean']:>12.1f}")
    print("=" * 74)
    print(f"runs {s['n_ok']} ok / {s['n_failed']} failed / {s['n_refused']} refused"
          f" · speculation hit rate {s['speculation_hit_rate']}")


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--mode", choices=["text", "voice"], default="text")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--pace", type=float, default=0.0,
                    help="seconds to wait between queries. Groq's free tier caps "
                         "tokens/minute, so an unpaced run measures rate limiting "
                         "rather than latency.")
    ap.add_argument("--out", default=str(DATA / "latency_bench.json"))
    args = ap.parse_args()

    if args.mode == "text":
        golden = json.loads((DATA / "golden_queries.json").read_text(encoding="utf-8"))
        queries = [g["eng_query"] for g in golden[: args.n]]
        print(f"text mode: {len(queries)} golden queries against {args.base}", flush=True)
        records = await run_text(args.base, queries, pace_s=args.pace)
    else:
        print(f"voice mode: streaming WAVs from {AUDIO_DIR}", flush=True)
        records = await run_voice(args.base)

    s = summarize(records, args.mode)
    print_table(s)
    if args.mode == "text":
        print("note: T-STT is absent in text mode by construction, not by omission.")

    pathlib.Path(args.out).write_text(
        json.dumps({"summary": s, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
