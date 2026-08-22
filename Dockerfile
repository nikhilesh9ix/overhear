FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    HF_HOME=/app/data/hf_cache \
    OMP_NUM_THREADS=4

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY data/index ./data/index
COPY data/golden_queries.json ./data/golden_queries.json

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
