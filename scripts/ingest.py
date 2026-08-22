"""Build the HNSW index from MSMARCO-XI.

Corpus size is deliberately small: CPU embedding on the build machine measured
4.8 docs/sec, so an 80k-passage ingest would take ~4.6 hours. See README for the
honest version of this. The chunker defaults to `fixed` for the same reason --
`semantic` scored best on recall@5 but has to embed every passage first, which
doubles ingest cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA  # noqa: E402
from app.core.chunking import FixedChunker, RecursiveChunker, SemanticChunker, assign_ids  # noqa: E402
from app.core.corpus import load_rows  # noqa: E402
from app.core.embedder import get_embedder  # noqa: E402
from app.core.index import HnswIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--strategy", default="fixed", choices=["fixed", "recursive", "semantic"])
    ap.add_argument("--out", default=str(DATA / "index"))
    args = ap.parse_args()

    t_start = time.perf_counter()
    print(f"loading {args.docs} docs ({args.lang}) ...", flush=True)
    rows = load_rows(lang=args.lang, limit=args.docs)
    print(f"  {len(rows)} docs / {sum(len(r.passages) for r in rows)} passages", flush=True)

    embedder = get_embedder()
    strat = {
        "fixed": lambda: FixedChunker(size=512, overlap=96),
        "recursive": lambda: RecursiveChunker(target_chars=480, max_chars=700),
        "semantic": lambda: SemanticChunker(embed_fn=embedder.embed_documents),
    }[args.strategy]()

    t0 = time.perf_counter()
    chunks = []
    for r in rows:
        chunks.extend(strat.chunk(r.query_id, r.passages))
    chunks = assign_ids(chunks)
    print(f"  {len(chunks)} chunks via {args.strategy} in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"embedding {len(chunks)} chunks (this is the slow part) ...", flush=True)
    t0 = time.perf_counter()
    vecs = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.perf_counter() - t0
    print(f"  embedded in {embed_s:.1f}s ({len(chunks)/embed_s:.1f}/s)", flush=True)

    # Answer text is carried on the payload so groundedness has something to check
    # against and the demo can show the dataset's own reference answer.
    answers = {r.query_id: r.eng_answer for r in rows}
    native = {r.query_id: (r.native_query, r.native_answer) for r in rows}
    payloads = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "text": c.text,
            "passage_idxs": list(c.passage_idxs),
            "strategy": c.strategy,
            "ref_answer": answers.get(c.doc_id, ""),
        }
        for c in chunks
    ]

    t0 = time.perf_counter()
    index = HnswIndex(dim=embedder.dim)
    index.build(vecs, payloads)
    out = Path(args.out)
    index.save(out)
    print(f"  index built+saved in {time.perf_counter()-t0:.1f}s -> {out}", flush=True)

    # Golden query set for the latency benchmark, straight from the dataset.
    golden = [
        {
            "query_id": r.query_id,
            "eng_query": r.eng_query,
            "native_query": r.native_query,
            "eng_answer": r.eng_answer,
            "native_answer": native[r.query_id][1],
            "gold_passage_idxs": list(r.gold_idxs),
        }
        for r in rows
    ]
    (DATA / "golden_queries.json").write_text(
        json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "lang": args.lang,
        "n_docs": len(rows),
        "n_chunks": len(chunks),
        "strategy": args.strategy,
        "embed_model": embedder.model_name,
        "dim": embedder.dim,
        "embed_seconds": round(embed_s, 1),
        "embed_rate_per_s": round(len(chunks) / embed_s, 1),
        "total_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"golden set: {len(golden)} queries -> {DATA / 'golden_queries.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
