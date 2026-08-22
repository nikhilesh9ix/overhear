# Overhear

**Voice RAG over `ai4bharat/MSMARCO-XI`, where retrieval happens while you are still talking.**

Built for Hacker House Goa, Task #2. `#RAGInGoa`

---

## The idea

Standard voice RAG is a queue: transcribe → embed → retrieve → generate. Every stage
waits for the one before it, and the user waits for all four.

Overhear removes one stage from the critical path. Sarvam's `saaras:v3-realtime`
emits **interim transcripts while audio is still streaming**. We embed those partials
on a debounce and keep a rolling top-k candidate set that refines with each word.
When VAD reports end of speech, retrieval is already done — the only thing left to
pay for is generation.

```
speech ──────────────────────────────► end of speech
  │ partial "what is a"                      │
  │   └─ embed + HNSW search (speculative)   │
  │ partial "what is a corp"                 │
  │   └─ cancel previous, embed + search     │
  │ partial "what is a corporation"          │
  │   └─ cancel previous, embed + search ────┤ candidates ready
                                             │
                                             ├──► T1: first generated token
                                             └──► T2: complete answer
```

A speculation that matches the final transcript costs **0ms** on the critical path.
One that misses falls back to a normal retrieval and **is logged and counted as a
miss** — the hit rate is reported, not hidden. A speculative system that silently
degrades is just a slow system with extra steps.

---

## What the latency numbers mean

The task asks for "the full process — chunking + vector DB retrieval + everything
through to final output — under 200ms." Chunking happens once at ingest, not per
query, so we read that as **everything from end-of-speech to the last generated
token**. Three numbers, all reported, including the unflattering ones:

| Metric | Definition | Target |
|---|---|---|
| **T1** | end of speech → **first generated token** | < 200ms — this is the number the task asks for |
| **T2** | end of speech → **last generated token** | reported regardless |
| **T-STT** | first audio byte → Sarvam's final transcript | reported **separately**, because the task's wording is ambiguous about whether STT counts. Reporting it apart means we are covered either way it is judged. |

---

## Measured latency

Two runs, both real, both reported. **Text mode** drives the pipeline without a mic
over 25 golden queries. **Voice mode** streams real WAVs through the WebSocket at
wall-clock pace, so T-STT and speculation are genuine.

### Text mode — 25 queries, 25/25 succeeded, 0 failed

| Metric | P50 | P70 | P100 |
|---|---|---|---|
| **T1** first token | **316.4ms** | 341.4ms | 609.2ms |
| **T2** complete answer | 318.7ms | 342.8ms | 613.2ms |
| Retrieval on critical path | 62.4ms | 70.7ms | 104.7ms |
| Groq network RTT | 135.9ms | — | — |
| HNSW search alone | 0.067ms | — | — |

### Voice mode — 6 spoken queries through the full WebSocket path

| Metric | P50 | P70 | P100 |
|---|---|---|---|
| **T1** first token | **490.0ms** | 514.0ms | 592.8ms |
| **T2** complete answer | 492.7ms | 516.6ms | 594.3ms |
| **T-STT** (Sarvam) | 1749.1ms | 1947.0ms | 2826.0ms |
| Retrieval on critical path | 77.9ms | 87.4ms | 113.4ms |
| Groq network RTT | 120.2ms | — | — |

**T-STT deserves a note:** it is measured from the *first audio byte*, so it
necessarily contains the entire duration of speech. These clips are 1.5–2.7s long,
so a T-STT of 1749ms means Sarvam finalized roughly **100–200ms after the speaker
stopped**. That is the number to judge the STT on, and it is good.

**T1 P50 is 316ms text / 490ms voice. Both are over the 200ms target.** Where it goes:

- **~120–136ms is network RTT** from India to Groq's US endpoint — a floor set by
  hosting, not by our code.
- **~41ms is the local embed.** `bge-small-en-v1.5` under onnxruntime 1.29 runs at
  **5.2 docs/sec** on this i7-13620H, roughly 40x below what that model should do.
  Verified on an idle machine, so not CPU contention. Root cause not found inside the
  time box.
- **~0.07ms is the actual vector search.** Retrieval was never the bottleneck.
- Voice mode carries additional WebSocket and audio-forwarding overhead over text.

> **Deployed numbers pending.** The highest-leverage fix is deploying near the
> provider: a US region should cut that RTT to ~10–30ms. That is a measurement we
> will make, not a claim we are making. Local numbers above stay for comparison.

Reproduce: `make bench-latency` / `make bench-latency-voice`.

---

## Does speculation actually work?

**Measured, and the first answer was no.** This is the most interesting result here.

Speculation only pays off if the candidate set computed from an interim transcript is
still right for the final one. The initial implementation required an exact text
match and hit **1 of 6**. Loosening it to accept word-prefixes raised the question of
whether a prefix retrieves the same thing, so we measured it over 74 golden queries:

| Prefix coverage | top-1 agreement | top-5 overlap | gold R@5 full → prefix |
|---|---|---|---|
| 50% | 0.216 | 0.511 | 0.960 → 0.649 |
| 60% | 0.270 | 0.597 | 0.960 → **0.730** |
| 70% | 0.446 | 0.727 | 0.960 → 0.865 |
| **80%** | 0.581 | 0.819 | 0.960 → **0.919** |
| 90% | 0.635 | 0.846 | 0.960 → 0.919 |

At the 60% threshold we had guessed at, speculation would have **cost 24% of answer
quality** to buy latency — a bad trade sold as a good one. The threshold is set to
**80%**, where the cost is 4 points, and `speculation_prefix_coverage` carries a
comment telling the next person not to lower it without re-running this.

**The deeper finding: Sarvam's interim transcripts are revised, not merely extended.**

```
MISS  final 'What is a corporation?'          spec 'what is this corporation'
MISS  final 'Why did Rachel Carson write...'  spec "why didn't rachel carson write..."
MISS  final 'Honesty and integrity...'        spec 'honesty or integrity definition'
HIT   final 'How fast does an eagle travel?'  spec 'how fast does an eagle travel'
```

Prefix logic cannot catch a revision, and detecting one would require embedding the
final text — which is exactly the 41ms speculation exists to avoid. So the honest
measured hit rate is **0.333** on this set, not the ~1.0 the architecture diagram
implies.

**Speculation is still worth it**, because the miss path is not a penalty: a miss is
just an ordinary retrieval, and the wasted speculative embed happens off the critical
path while the user is still talking. It is a free option that pays ~62ms one time in
three. One real bug was found here — `on_final` originally blocked up to 120ms
waiting for an in-flight speculation *before* checking whether it could ever be
accepted, so every miss paid for a result it then discarded. Fixing that moved voice
T1 P50 from 535ms to 490ms and the hit rate from 0.167 to 0.333.

Reproduce: `make bench-speculation`

---

## Chunking strategies

MS MARCO passages are already retrieval-sized (83–473 chars, mean ~290), so chunking
here is not about splitting long documents. Each `query_id` contributes ~10 passages
on a shared topic; we treat that group as the document and let each strategy decide
where retrieval units fall inside it. That is a real decision with real recall
consequences — a fixed window straddles passage boundaries and dilutes a gold
passage, while a structural split preserves it.

**The golden set is not synthesized.** Every MSMARCO-XI row ships `is_selected` flags
marking which passages actually answer the query. A retrieval counts as a hit only if
a returned chunk belongs to the right `query_id` **and** covers a gold passage index —
matching the document alone would be trivial when each document has ~10 passages.

**120 documents, 1196 passages, 131 gold labels:**

| Strategy | Chunks | /doc | p50 chars | R@1 | **R@5** | MRR@10 | Search p50 |
|---|---|---|---|---|---|---|---|
| **semantic** | 655 | 5.46 | 590 | 0.483 | **1.000** | 0.694 | 0.059ms |
| fixed | 929 | 7.74 | 512 | **0.517** | 0.975 | 0.692 | 0.067ms |
| recursive | 1098 | 9.15 | 311 | 0.417 | 0.950 | 0.629 | 0.069ms |

- **`fixed`** — 512-char window, 96-char overlap, blind to passage edges. The control.
- **`recursive`** — respects passage boundaries first, subdivides on sentences only
  when a passage exceeds 700 chars, packs small neighbours without splitting them.
- **`semantic`** — embeds each passage, cuts where cosine similarity between
  consecutive passages drops below a per-document percentile threshold, so topically
  coherent runs merge and topic shifts force boundaries.

**Winner: `semantic`** on R@5, and it does it with the fewest chunks (5.46/doc vs
7.74), meaning a smaller index for better recall. Merging topically-adjacent passages
puts more of the answer inside a single vector, so the gold content is less diluted.
`fixed` wins R@1, which fits: its tighter 512-char units are more precise when they
happen to land well, but they fragment gold passages often enough to lose at k=5.

**Honest caveat: at 120 documents the index is too small for R@5 to discriminate.**
All three are at or near ceiling. These numbers are directionally useful and not
much more. Scoring at a corpus size where R@5 separates the strategies needed
embedding throughput we did not have tonight — see the constraint below.

**The shipped index uses `fixed`, not the winner.** `semantic` has to embed every
passage before it can find boundaries, which roughly doubles ingest cost, and at
5.2 embeds/sec that was unaffordable. This is a deliberate trade recorded here rather
than quietly made.

Reproduce: `make bench-retrieval`

---

## Guardrails

Three gates. Every refusal names its specific reason — a generic error is not a
guardrail, it is a bug report aimed at the user.

**1. Input gate** — empty or near-empty transcript, or unsafe request.

> *"how do I build a pipe bomb at home"*
> → refused at `input` / `unsafe_input`, **0.03ms**, before any retrieval or spend.
> *"I can't help with that one. Ask me something I can look up in the passages instead."*

The unsafe patterns are deliberately narrow. A QA system over MS MARCO will
legitimately be asked about disease, weapons history, drugs and crime; refusing those
would be wrong. The gate targets requests to *produce* harm.

**2. Retrieval gate** — nothing retrieved, top score below 0.55, or a flat top-k
distribution. Flatness is the interesting one: when every candidate scores about
equally, none of them actually answers the question.

> *"what is the capital of Mars and who rules it"*
> → refused at `retrieval` / `retrieval_low_confidence`, top score 0.595, gap 0.011.
> *"I don't know. Several passages matched your question about equally well, which usually means none of them actually answers it."*

**3. Groundedness gate** — the model returns a structured `grounded: bool` plus
citations, so refusal is a first-class output rather than something regexed out of
prose. Also rejects an answer that claims `grounded: true` while citing nothing, and
one whose citations do not match the retrieved chunks.

In the 25-query benchmark run, **4 queries were refused**: 3 at the retrieval gate,
1 at the groundedness gate. The system says "I don't know" and means it.

---

## Stack

| Layer | Choice | Note |
|---|---|---|
| STT | Sarvam `saaras:v3-realtime` over WebSocket | true interim transcripts, VAD endpointing, `mode=translate` |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | local ONNX, in-process, 384-dim, no API hop |
| Vector index | `hnswlib` directly, in-memory | no wrapper library in the hot path |
| Generation | Groq `qwen/qwen3.6-27b`, Cerebras fallback | streaming, persistent HTTP/2 client, pre-warmed |
| Backend | Python 3.12, FastAPI, one WebSocket endpoint | |
| Frontend | single static page, Web Audio API + AudioWorklet | PCM16 @ 16kHz streamed over the same socket |

### Why English passages

MSMARCO-XI rows carry `English_passages` and `Translated_passages` side by side. We
index the English side and run Sarvam in `mode=translate`, so Indic speech arrives as
English text in the same vector space. Reason: `bge-small-en-v1.5` is trained for
exactly this asymmetric MS MARCO retrieval task, whereas fastembed's multilingual
option (`paraphrase-multilingual-MiniLM-L12-v2`) is symmetric-paraphrase trained and
would have cost material recall. Indic in, Indic-dataset content, English retrieval
in the middle.

### Model substitutions, forced not chosen

The brief specified Groq `llama-3.1-8b-instant` with a Cerebras fallback. Neither was
available on the supplied keys:

- **Cerebras returns HTTP 402 Payment Required** on every model. It is wired up behind
  the same interface and the circuit breaker handles it, but it cannot serve traffic.
  The app logs this loudly at startup rather than pretending it has a fallback.
- **Groq has no Llama models** on this account. Of what it does offer,
  `qwen/qwen3.6-27b` was the only model under the latency target — 108ms to first
  token versus 485ms for `gpt-oss-20b` and 814ms for `gpt-oss-120b`.
- **`qwen3.6-27b` needs `reasoning_effort=none`.** Left on, it streams a `<think>`
  block before any answer content: 127ms to a "first token" that is a reasoning
  token, but 602ms to anything usable. Reporting the former as T1 would have been a
  lie. Switched off: 108ms to first token, 149ms to a complete answer.

---

## Running it

```bash
cp .env.example .env    # add SARVAM_API_KEY and GROQ_API_KEY
uv venv && uv pip install -r pyproject.toml
python scripts/fetch_parquet.py hi validation
make ingest
make dev
```

| Target | Does |
|---|---|
| `make smoke-sarvam WAV=x.wav` | proves interim transcripts arrive *during* audio |
| `make smoke-providers` | streaming TTFT for Groq and Cerebras |
| `make ingest` | chunk → embed → HNSW → golden query set |
| `make bench-retrieval` | scores all three chunking strategies |
| `make bench-latency` | P50/P70/P100 for T1, T2 (text mode) |
| `make bench-latency-voice` | same, streaming real WAVs through the WebSocket |
| `make bench-speculation` | prefix-vs-full retrieval agreement |
| `make bench-audio` | synthesize spoken golden queries via Sarvam TTS |

---

## Known constraints

Stated plainly rather than buried.

- **Corpus is 400 documents / 3140 chunks.** The plan was 8k documents / 80k passages.
  At the machine's measured 5.2 embeds/sec that would have taken ~4.6 hours. The index
  is small enough that a judge may notice, and small enough to make the R@5 comparison
  above less discriminative than it should be.
- **T1 P50 is 316ms locally, not under 200ms.** Decomposed above. Deployment near the
  provider is the fix and the measurement is pending.
- **Cerebras is configured but non-functional** (402 on the supplied key).
- **Groq free tier caps 8000 tokens/min**, which is ~9 queries/minute. Generation
  sends the top 3 chunks trimmed to 420 chars to stay inside it; retrieval still
  returns and displays the full top-k.
- **Voice benchmark uses 6 TTS-synthesized clips, not 20-30 human recordings.** The
  audio is real and goes through the real STT path, but Sarvam TTS output is clean:
  no accents, background noise or disfluency. Voice-mode T-STT and the speculation
  hit rate are therefore a best case. Human recordings dropped into `bench/audio/`
  are picked up by the same command and are strictly better evidence.
- **Speculation hit rate is 0.333**, not the ~1.0 the architecture implies, because
  interim transcripts are revised rather than merely extended. Analysed above.

## Cut to fit the clock

Planned, deliberately dropped, not pretended away:

- **Layer 2 speculative generation** — firing the LLM on an unstable interim
  transcript. Cut before work started; too risky to demo the same day.
- **A fourth chunking strategy** and the partial-query-stability ablation.
- **Live animated latency waterfall UI** — the trace panel shows the same events as a
  running log instead.
- **Next.js frontend** — replaced with a single static page served by FastAPI. One
  service, one deploy, same-origin WebSocket.
- **50-query benchmark** → 25 queries.
