PY := ./.venv/Scripts/python.exe

.PHONY: smoke smoke-dataset smoke-sarvam smoke-providers ingest bench-retrieval bench-latency dev health

smoke: smoke-dataset smoke-providers

smoke-dataset:
	$(PY) scripts/smoke_dataset.py

smoke-sarvam:
	$(PY) scripts/smoke_sarvam.py $(WAV)

smoke-providers:
	$(PY) scripts/smoke_providers.py

ingest:
	$(PY) scripts/ingest.py

bench-retrieval:
	$(PY) scripts/bench_retrieval.py

bench-latency:
	$(PY) scripts/bench_latency.py

dev:
	$(PY) -m uvicorn app.main:app --reload --port 8000

health:
	curl -s http://localhost:8000/health
