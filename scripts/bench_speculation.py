"""Does speculating on a prefix return the same passages as the full query?

The retriever accepts a speculation when the text it was computed from is a word
prefix covering >= speculation_prefix_coverage of the final query. That is only
sound if a prefix retrieves the same neighbourhood. This measures it instead of
assuming it.

For each golden query we retrieve on the full text, then on truncated prefixes,
and report:
  - top-1 agreement : does the prefix return the same best chunk
  - top-5 overlap   : |prefix top5 ∩ full top5| / 5
  - gold recall@5   : does the prefix still surface a gold passage

If prefix retrieval degrades badly, the acceptance rule is wrong and the honest
move is to tighten the coverage threshold, not to keep the flattering hit rate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import DATA  # noqa: E402
from app.core.embedder import get_embedder  # noqa: E402
from app.core.index import HnswIndex  # noqa: E402

COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9]


def prefix_words(text: str, coverage: float, min_words: int = 3) -> str | None:
    w = text.split()
    n = max(min_words, int(round(len(w) * coverage)))
    if n >= len(w) or n < min_words:
        return None
    return " ".join(w[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=str(DATA / "speculation_bench.json"))
    args = ap.parse_args()

    index = HnswIndex.load(DATA / "index")
    emb = get_embedder()
    golden = json.loads((DATA / "golden_queries.json").read_text(encoding="utf-8"))[: args.n]
    queries = [g["eng_query"].strip().lstrip(". ").strip() for g in golden]
    keep = [i for i, q in enumerate(queries) if len(q.split()) >= 5]
    golden = [golden[i] for i in keep]
    queries = [queries[i] for i in keep]
    print(f"{len(queries)} golden queries with >=5 words, k={args.k}", flush=True)

    full_vecs = emb.embed_queries(queries)
    full_res = [index.search(v, k=args.k) for v in full_vecs]
    full_ids = [[p["chunk_id"] for p, _ in r] for r in full_res]

    def gold_hit(res, g) -> bool:
        want = set(g["gold_passage_idxs"])
        return any(p["doc_id"] == g["query_id"] and want.intersection(p["passage_idxs"])
                   for p, _ in res)

    full_gold = [gold_hit(r, g) for r, g in zip(full_res, golden)]

    rows = []
    for cov in COVERAGES:
        pairs = [(i, p) for i, q in enumerate(queries)
                 if (p := prefix_words(q, cov)) is not None]
        if not pairs:
            continue
        idxs = [i for i, _ in pairs]
        vecs = emb.embed_queries([p for _, p in pairs])
        res = [index.search(v, k=args.k) for v in vecs]

        top1 = np.mean([res[j][0][0]["chunk_id"] == full_ids[i][0]
                        for j, i in enumerate(idxs)])
        overlap = np.mean([
            len(set(p["chunk_id"] for p, _ in res[j]) & set(full_ids[i])) / args.k
            for j, i in enumerate(idxs)
        ])
        pg = np.mean([gold_hit(res[j], golden[i]) for j, i in enumerate(idxs)])
        fg = np.mean([full_gold[i] for i in idxs])
        rows.append({
            "coverage": cov, "n": len(idxs),
            "top1_agreement": round(float(top1), 4),
            f"top{args.k}_overlap": round(float(overlap), 4),
            f"prefix_gold_recall@{args.k}": round(float(pg), 4),
            f"full_gold_recall@{args.k}": round(float(fg), 4),
            "recall_delta": round(float(pg - fg), 4),
        })
        print(f"  coverage {cov:.0%}: n={len(idxs)} top1={top1:.3f} "
              f"top{args.k}_overlap={overlap:.3f} gold_recall {fg:.3f} -> {pg:.3f}", flush=True)

    print("\n" + "=" * 92)
    print(f"{'prefix coverage':<18}{'n':>6}{'top-1 agree':>14}{'top-5 overlap':>16}"
          f"{'gold R@5 full':>16}{'gold R@5 prefix':>18}")
    print("-" * 92)
    for r in rows:
        print(f"{r['coverage']:<18.0%}{r['n']:>6}{r['top1_agreement']:>14.3f}"
              f"{r[f'top{args.k}_overlap']:>16.3f}{r[f'full_gold_recall@{args.k}']:>16.3f}"
              f"{r[f'prefix_gold_recall@{args.k}']:>18.3f}")
    print("=" * 92)

    pathlib.Path(args.out).write_text(
        json.dumps({"k": args.k, "n_queries": len(queries), "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
