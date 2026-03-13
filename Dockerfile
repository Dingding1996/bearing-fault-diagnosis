# =============================================================
# Bearing Fault Diagnosis — Inference Service
# =============================================================
# Serves the trained ML model as a REST API via FastAPI.
# The model artefacts (mlruns/) are mounted at runtime — not
# baked into the image — so the same image works with any
# trained checkpoint.
#
# Build:
#   docker build -t bearing-fault-api .
#
# Run (standalone):
#   docker run -p 8000:8000 \
#     -v "$(pwd)/mlruns:/app/mlruns:ro" \
#     bearing-fault-api
#
# Run via Compose (recommended):
#   docker compose up
# =============================================================

FROM python:3.11-slim

# --- System dependencies ---
# gcc is required to compile some Python C extensions (e.g. scipy wheels fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies ---
# Copy requirements first to leverage Docker layer caching.
# Re-installing packages only happens when this file changes.
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# --- Application code ---
# Only copy the utils/ modules needed for inference; notebooks and
# training artefacts stay on the host.
COPY utils/ ./utils/

# --- Runtime ---
EXPOSE 8000

# mlruns/ is NOT copied — it must be mounted as a volume at runtime.
# This keeps the image model-agnostic and avoids bloating it with
# experiment artefacts that change after every training run.

CMD ["uvicorn", "utils.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
