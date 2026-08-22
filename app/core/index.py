"""In-memory HNSW index. hnswlib directly, no wrapper library in the hot path."""
from __future__ import annotations

import json
import pathlib

import hnswlib
import numpy as np


class HnswIndex:
    def __init__(self, dim: int, space: str = "cosine") -> None:
        self.dim = dim
        self.space = space
        self._index: hnswlib.Index | None = None
        self.payloads: list[dict] = []

    def build(self, vecs: np.ndarray, payloads: list[dict],
              m: int = 32, ef_construction: int = 200, ef: int = 96) -> None:
        if len(vecs) != len(payloads):
            raise ValueError(f"{len(vecs)} vectors vs {len(payloads)} payloads")
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        idx.init_index(max_elements=len(vecs), ef_construction=ef_construction, M=m)
        idx.add_items(vecs, np.arange(len(vecs)))
        idx.set_ef(ef)
        self._index = idx
        self.payloads = payloads

    def search(self, vec: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        if self._index is None:
            raise RuntimeError("index not built")
        k = min(k, len(self.payloads))
        labels, dists = self._index.knn_query(vec.reshape(1, -1), k=k)
        # hnswlib returns cosine *distance*; convert to similarity for readability.
        return [(self.payloads[int(i)], 1.0 - float(d)) for i, d in zip(labels[0], dists[0])]

    def set_ef(self, ef: int) -> None:
        if self._index is not None:
            self._index.set_ef(ef)

    def save(self, dirpath: pathlib.Path) -> None:
        if self._index is None:
            raise RuntimeError("index not built")
        dirpath.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(dirpath / "hnsw.bin"))
        (dirpath / "payloads.jsonl").write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in self.payloads),
            encoding="utf-8",
        )
        (dirpath / "meta.json").write_text(
            json.dumps({"dim": self.dim, "space": self.space, "n": len(self.payloads)}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dirpath: pathlib.Path, ef: int = 96) -> "HnswIndex":
        meta = json.loads((dirpath / "meta.json").read_text(encoding="utf-8"))
        obj = cls(dim=meta["dim"], space=meta["space"])
        idx = hnswlib.Index(space=meta["space"], dim=meta["dim"])
        idx.load_index(str(dirpath / "hnsw.bin"), max_elements=meta["n"])
        idx.set_ef(ef)
        obj._index = idx
        obj.payloads = [
            json.loads(line)
            for line in (dirpath / "payloads.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        return obj

    def __len__(self) -> int:
        return len(self.payloads)
