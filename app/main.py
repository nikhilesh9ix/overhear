import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

START = time.time()

app = FastAPI(title="Overhear", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - START, 1),
        "keys": {
            "sarvam": bool(settings.sarvam_api_key),
            "groq": bool(settings.groq_api_key),
            "cerebras": bool(settings.cerebras_api_key),
        },
        "embed_model": settings.embed_model,
    }
