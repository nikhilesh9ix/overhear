"""Smoke test 3: Groq + Cerebras streaming TTFT with a realistic RAG-sized prompt."""
import asyncio
import json
import statistics
import sys
import time

import httpx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

N = 5
CONTEXT = ("\n\n".join(
    f"[{i}] Passage {i}: " + ("The Indian Railways network spans over 68,000 route kilometres and "
    "carries roughly 23 million passengers daily across 7,300 stations. " * 3)
    for i in range(1, 6)
))
MESSAGES = [
    {"role": "system", "content": "Answer only from the passages. Cite passage numbers."},
    {"role": "user", "content": f"{CONTEXT}\n\nQuestion: How many passengers does Indian Railways carry daily?"},
]


async def ttft(client: httpx.AsyncClient, url: str, key: str, model: str) -> tuple[float, float, str]:
    t0 = time.perf_counter()
    first = None
    out = []
    async with client.stream(
        "POST", url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": MESSAGES, "stream": True, "max_tokens": 120, "temperature": 0.1},
        timeout=30.0,
    ) as r:
        if r.status_code != 200:
            body = (await r.aread()).decode()[:400]
            raise RuntimeError(f"HTTP {r.status_code}: {body}")
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            delta = d["choices"][0]["delta"].get("content")
            if delta:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                out.append(delta)
    total = (time.perf_counter() - t0) * 1000
    return first or total, total, "".join(out)


async def bench(name: str, url: str, key: str, model: str) -> None:
    print(f"\n===== {name}  model={model} =====", flush=True)
    if not key:
        print(f"!! no API key for {name}, skipped", flush=True)
        return
    async with httpx.AsyncClient(http2=True) as client:
        firsts, totals = [], []
        for i in range(N):
            try:
                f, t, txt = await ttft(client, url, key, model)
            except Exception as e:
                print(f"  run {i+1}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            firsts.append(f)
            totals.append(t)
            tag = "(cold, incl. TLS+conn)" if i == 0 else ""
            print(f"  run {i+1}: TTFT {f:7.1f}ms   total {t:7.1f}ms  {tag}", flush=True)
            if i == 0:
                print(f"    sample: {txt[:120]!r}", flush=True)
        if len(firsts) > 1:
            warm = firsts[1:]
            print(f"  -> warm TTFT: median {statistics.median(warm):.1f}ms  min {min(warm):.1f}ms  max {max(warm):.1f}ms", flush=True)
            print(f"  -> warm total: median {statistics.median(totals[1:]):.1f}ms", flush=True)


async def main() -> int:
    await bench("GROQ", settings.groq_url, settings.groq_api_key, settings.groq_model)
    await bench("CEREBRAS", settings.cerebras_url, settings.cerebras_api_key, settings.cerebras_model)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
