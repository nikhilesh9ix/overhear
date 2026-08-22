"""Score the three chunking strategies on recall@5 over MSMARCO-XI's own labels.

Golden set is not synthesized: every row ships a query plus is_selected flags on
its passages, so the gold answer for a query is "any chunk covering one of that
query's selected passages". A retrieval counts as a hit only if a returned chunk
belongs to the right query_id AND covers a gold passage index -- matching the
doc alone would be trivially easy given each doc has ~10 passages.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA  # noqa: E402
from app.core.chunking import (  # noqa: E402
    Chunk,
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    assign_ids,
)
from app.core.corpus import load_rows  # noqa: E402
from app.core.embedder import get_embedder  # noqa: E402
from app.core.index import HnswIndex  # noqa: E402


def build_strategies(embedder):
    return [
        FixedChunker(size=512, overlap=96),
        RecursiveChunker(target_chars=480, max_chars=700),
        SemanticChunker(embed_fn=embedder.embed_documents, percentile=25.0, max_chars=900),
    ]


def evaluate(name: str, chunks: list[Chunk], rows, embedder, ks: tuple[int, ...],
             chunk_wall_s: float) -> dict:
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - t0

    payloads = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
         "passage_idxs": list(c.passage_idxs), "strategy": c.strategy}
        for c in chunks
    ]
    t0 = time.perf_counter()
    index = HnswIndex(dim=embedder.dim)
    index.build(vecs, payloads)
    build_s = time.perf_counter() - t0

    queries = [r.eng_query for r in rows]
    gold = [(r.query_id, set(r.gold_idxs)) for r in rows]
    qvecs = embedder.embed_queries(queries)

    maxk = max(ks)
    hits = {k: 0 for k in ks}
    mrr = 0.0
    lat = []
    for qv, (qid, gidx) in zip(qvecs, gold):
        t0 = time.perf_counter()
        res = index.search(qv, k=maxk)
        lat.append((time.perf_counter() - t0) * 1000)
        rank = None
        for pos, (p, _score) in enumerate(res, start=1):
            if p["doc_id"] == qid and gidx.intersection(p["passage_idxs"]):
                rank = pos
                break
        if rank is not None:
            mrr += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits[k] += 1

    n = len(rows)
    lens = [len(c.text) for c in chunks]
    return {
        "strategy": name,
        "n_chunks": len(chunks),
        "chunks_per_doc": round(len(chunks) / len(rows), 2),
        "chunk_chars_mean": int(np.mean(lens)),
        "chunk_chars_p50": int(np.percentile(lens, 50)),
        "chunk_chars_p95": int(np.percentile(lens, 95)),
        **{f"recall@{k}": round(hits[k] / n, 4) for k in ks},
        "mrr@10": round(mrr / n, 4),
        "search_ms_p50": round(float(np.percentile(lat, 50)), 3),
        "search_ms_p95": round(float(np.percentile(lat, 95)), 3),
        "chunk_wall_s": round(chunk_wall_s, 1),
        "embed_wall_s": round(embed_s, 1),
        "index_build_s": round(build_s, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--docs", type=int, default=1500,
                    help="docs to score over; kept smaller than full ingest so the "
                         "three-way comparison finishes inside its time box")
    ap.add_argument("--ks", default="1,3,5,10")
    ap.add_argument("--out", default=str(DATA / "retrieval_bench.json"))
    args = ap.parse_args()
    ks = tuple(int(x) for x in args.ks.split(","))

    print(f"loading {args.docs} docs from {args.lang} ...", flush=True)
    rows = load_rows(lang=args.lang, limit=args.docs)
    n_pass = sum(len(r.passages) for r in rows)
    print(f"  {len(rows)} docs / {n_pass} passages / "
          f"{sum(len(r.gold_idxs) for r in rows)} gold labels", flush=True)

    embedder = get_embedder()
    print(f"embedder {embedder.model_name} dim={embedder.dim}", flush=True)

    results = []
    for strat in build_strategies(embedder):
        print(f"\n== {strat.name} ==", flush=True)
        t0 = time.perf_counter()
        chunks: list[Chunk] = []
        for r in rows:
            chunks.extend(strat.chunk(r.query_id, r.passages))
        chunks = assign_ids(chunks)
        chunk_s = time.perf_counter() - t0
        print(f"  {len(chunks)} chunks in {chunk_s:.1f}s", flush=True)
        res = evaluate(strat.name, chunks, rows, embedder, ks, chunk_s)
        results.append(res)
        print("  " + json.dumps({k: v for k, v in res.items() if k != "strategy"}), flush=True)

    results.sort(key=lambda r: r["recall@5"], reverse=True)
    winner = results[0]

    print("\n" + "=" * 108, flush=True)
    hdr = f"{'strategy':<11}{'chunks':>8}{'/doc':>7}{'p50 chars':>11}"
    hdr += "".join(f"{'R@'+str(k):>9}" for k in ks) + f"{'MRR@10':>9}{'search p50':>12}{'embed s':>9}"
    print(hdr, flush=True)
    print("-" * 108, flush=True)
    for r in results:
        line = f"{r['strategy']:<11}{r['n_chunks']:>8}{r['chunks_per_doc']:>7}{r['chunk_chars_p50']:>11}"
        line += "".join(f"{r['recall@'+str(k)]:>9.4f}" for k in ks)
        line += f"{r['mrr@10']:>9.4f}{r['search_ms_p50']:>12.3f}{r['embed_wall_s']:>9.1f}"
        print(line, flush=True)
    print("=" * 108, flush=True)
    print(f"WINNER: {winner['strategy']}  recall@5={winner['recall@5']:.4f}", flush=True)

    payload = {
        "lang": args.lang,
        "n_docs": len(rows),
        "n_passages": n_pass,
        "embed_model": embedder.model_name,
        "ks": list(ks),
        "winner": winner["strategy"],
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
