"""Load MSMARCO-XI rows off the locally downloaded parquet.

Streaming from the Hub read ~200 rows in 4.5 minutes, so ingest works against a
downloaded per-language file instead. We index English_passages (see the Phase 0
commit for why) but keep the Indic query text around for the demo.
"""
from __future__ import annotations

import os
import pathlib

import pyarrow.parquet as pq

from app.config import DATA, ROOT
from app.core.chunking import Passage

os.environ.setdefault("HF_HOME", str(DATA / "hf_cache"))

LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}

COLUMNS = ["query_id", "query_type", "Eng_Query", "query", "Eng_Answer", "Answer", "passages"]


class Row:
    __slots__ = ("query_id", "query_type", "eng_query", "native_query",
                 "eng_answer", "native_answer", "passages")

    def __init__(self, d: dict) -> None:
        self.query_id: int = d["query_id"]
        self.query_type: str = d["query_type"]
        self.eng_query: str = d["Eng_Query"]
        self.native_query: str = d["query"]
        self.eng_answer: str = d["Eng_Answer"]
        self.native_answer: str = d["Answer"]
        p = d["passages"]
        self.passages: list[Passage] = [
            Passage(
                doc_id=self.query_id,
                passage_idx=i,
                text=t,
                is_selected=bool(sel),
            )
            for i, (t, sel) in enumerate(zip(p["English_passages"], p["is_selected"]))
        ]

    @property
    def gold_idxs(self) -> tuple[int, ...]:
        return tuple(p.passage_idx for p in self.passages if p.is_selected)


def parquet_path(lang: str = "hi", split: str = "validation") -> pathlib.Path:
    """Locate the downloaded parquet in the HF cache. Fails loudly if absent."""
    stem = f"{LANG_FILE[lang]}{'val' if split == 'validation' else 'train'}.parquet"
    root = DATA / "hf_cache" / "hub" / "datasets--ai4bharat--MSMARCO-XI"
    hits = sorted(root.glob(f"snapshots/*/{split}/{stem}")) if root.exists() else []
    if not hits:
        raise FileNotFoundError(
            f"{split}/{stem} not downloaded. Run: make fetch LANG={lang} "
            f"(or python scripts/fetch_parquet.py {lang} {split})"
        )
    return hits[0]


def load_rows(lang: str = "hi", split: str = "validation", limit: int = 8000) -> list[Row]:
    """Read the first `limit` usable rows. Rows with no gold passage are dropped:
    they cannot contribute to recall scoring and would only pad the index."""
    path = parquet_path(lang, split)
    pf = pq.ParquetFile(path)
    out: list[Row] = []
    for batch in pf.iter_batches(batch_size=512, columns=COLUMNS):
        for d in batch.to_pylist():
            row = Row(d)
            if not row.gold_idxs or not row.eng_query.strip():
                continue
            out.append(row)
            if len(out) >= limit:
                return out
    return out
