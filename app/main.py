from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import DATA, ROOT, settings
from app.core.embedder import get_embedder
from app.core.generation import Generator
from app.core.index import HnswIndex
from app.core.retriever import SpeculativeRetriever
from app.core.types import TranscriptIn
from app.pipeline import answer_query
from app.ws import router as ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("overhear")

START = time.time()
STATIC = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index = None
    app.state.embedder = None
    app.state.generator = None

    t0 = time.perf_counter()
    app.state.embedder = get_embedder()
    warm_ms = app.state.embedder.warm()
    log.info("embedder %s ready in %.1fs (warm query embed %.1fms)",
             settings.embed_model, time.perf_counter() - t0, warm_ms)

    idx_dir = DATA / "index"
    if (idx_dir / "meta.json").exists():
        t0 = time.perf_counter()
        app.state.index = HnswIndex.load(idx_dir)
        log.info("index loaded: %d chunks in %.1fs", len(app.state.index),
                 time.perf_counter() - t0)
    else:
        log.error("NO INDEX at %s -- run `make ingest`. /ws will refuse sessions.", idx_dir)

    app.state.generator = Generator()
    await app.state.generator.start()
    if not app.state.generator.providers:
        log.error("NO LLM PROVIDER configured -- set GROQ_API_KEY or CEREBRAS_API_KEY. "
                  "Retrieval will work; generation will fail loudly.")

    yield

    await app.state.generator.stop()


app = FastAPI(title="Overhear", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.include_router(ws_router)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def root():
    idx = STATIC / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return {"service": "overhear", "hint": "no static UI built"}


@app.get("/health")
def health() -> dict:
    idx = app.state.index if hasattr(app.state, "index") else None
    gen = app.state.generator if hasattr(app.state, "generator") else None
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START, 1),
        "index_chunks": len(idx) if idx else 0,
        "embed_model": settings.embed_model,
        "providers": [p.name for p in gen.providers] if gen else [],
        "provider_rtt_ms": {k: round(v, 1) for k, v in gen.warm_rtt_ms.items()} if gen else {},
        "keys": {
            "sarvam": bool(settings.sarvam_api_key),
            "groq": bool(settings.groq_api_key),
            "cerebras": bool(settings.cerebras_api_key),
        },
    }


class AskIn(BaseModel):
    query: str


@app.post("/ask")
async def ask(body: AskIn) -> dict:
    """Text-in path. Exists so retrieval, guardrails and generation can be exercised
    and benchmarked without a microphone -- the voice path calls the same pipeline."""
    if app.state.index is None:
        return {"error": "no index; run `make ingest`"}
    retriever = SpeculativeRetriever(app.state.index, app.state.embedder)
    t = TranscriptIn(text=body.query, is_final=True)
    events = []
    speech_end = time.perf_counter()
    async for ev in answer_query(t, retriever, app.state.generator,
                                 speech_end_t=speech_end, stt_ms=None):
        events.append(ev)
    return {"query": body.query, "events": events}
