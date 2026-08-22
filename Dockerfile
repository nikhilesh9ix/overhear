FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/hf_cache \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY pyproject.toml ./

# hnswlib 0.8.0 is published as an sdist only -- there is no manylinux wheel -- so
# it is compiled here. The toolchain is installed and removed inside one layer so
# it does not end up in the shipped image.
#
# HNSWLIB_NO_NATIVE and the explicit -march matter: hnswlib's setup.py compiles
# with -march=native, baking in whatever instruction set the *build* machine has.
# On Render that produced a binary the *runtime* instance could not execute --
# "Illegal instruction (core dumped)", exit 132, immediately on import. Targeting
# a portable baseline costs a little search throughput; HNSW query time was
# 0.067ms, so there is room.
ENV HNSWLIB_NO_NATIVE=1 \
    CFLAGS="-march=x86-64 -mtune=generic" \
    CXXFLAGS="-march=x86-64 -mtune=generic"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential g++; \
    pip install --no-cache-dir uv; \
    uv pip install --system --no-cache --no-binary hnswlib -r pyproject.toml; \
    apt-get purge -y --auto-remove build-essential g++; \
    rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY static ./static
COPY scripts ./scripts
COPY data/index ./data/index
COPY data/golden_queries.json ./data/golden_queries.json

# Bake the embedding model into the image. Without this the first request after a
# cold start pays a ~67MB download, which is exactly when a judge is watching.
RUN python -c "from fastembed import TextEmbedding; \
    TextEmbedding(model_name='BAAI/bge-small-en-v1.5').embed(['warm'])" \
    && python -c "import numpy as np, hnswlib, fastembed, fastapi; i = hnswlib.Index(space='cosine', dim=8); i.init_index(max_elements=4, ef_construction=8, M=4); i.add_items(np.random.rand(4, 8).astype('float32'), np.arange(4)); i.knn_query(np.random.rand(1, 8).astype('float32'), k=2); print('imports ok, hnswlib executes')"

EXPOSE 8000

# Must stay in shell form: Railway injects PORT at runtime and an exec-form CMD
# would pass "$PORT" through literally, which uvicorn rejects with
# "invalid int value: '$PORT'". Do not add a startCommand override in
# railway.json either -- that is run without a shell and reintroduces the bug.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info
