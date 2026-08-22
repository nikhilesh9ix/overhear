"""Local ONNX embeddings via fastembed. In-process, no network hop at query time."""
from __future__ import annotations

import threading
import time

import numpy as np
from fastembed import TextEmbedding

from app.config import settings

# bge-small-en-v1.5 is an asymmetric retrieval model: queries want this prefix,
# documents do not. Skipping it costs real recall.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Thread-safe wrapper with a warm-up so the first real query isn't the cold one."""

    _lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self._model = TextEmbedding(model_name=self.model_name)
        self.dim = len(next(iter(self._model.embed(["warmup"]))))

    def embed_documents(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._model.embed(texts, batch_size=batch_size))
        return _l2(np.asarray(vecs, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        with self._lock:
            vec = next(iter(self._model.query_embed([text])))
        return _l2(np.asarray(vec, dtype=np.float32).reshape(1, -1))[0]

    def embed_queries(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = list(self._model.query_embed(texts, batch_size=batch_size))
        return _l2(np.asarray(vecs, dtype=np.float32))

    def warm(self, n: int = 3) -> float:
        """Run a few throwaway embeds; returns the last one's latency in ms."""
        last = 0.0
        for _ in range(n):
            t0 = time.perf_counter()
            self.embed_query("warm up the onnx graph")
            last = (time.perf_counter() - t0) * 1000
        return last


def _l2(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


_singleton: Embedder | None = None


def get_embedder() -> Embedder:
    global _singleton
    if _singleton is None:
        _singleton = Embedder()
    return _singleton
