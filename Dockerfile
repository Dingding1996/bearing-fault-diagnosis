# =============================================================
# Bearing Fault Diagnosis — Inference Service
# =============================================================
# Serves the trained ML model as a REST API via FastAPI.
#
# Two build modes:
#
#   CI / AWS (model baked in):
#     GitHub Actions downloads mlruns/ from S3 before building,
#     so COPY mlruns/ below embeds the model directly in the image.
#     docker build -t bearing-fault-api .
#
#   Local dev (model mounted at runtime):
#     docker compose up          ← uses docker-compose.yml volume mount
#     docker run -p 8000:8000 \
#       -v "$(pwd)/mlruns:/app/mlruns:ro" \
#       bearing-fault-api
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

# --- Model artefacts ---
# In CI builds, GitHub Actions syncs mlruns/ from S3 before `docker build`,
# so this COPY embeds the trained model in the image (fast cold starts, no
# runtime S3 dependency).
# For local dev without mlruns/, use the docker-compose.yml volume mount instead.
COPY mlruns/ ./mlruns/

# --- Runtime ---
EXPOSE 8000

CMD ["uvicorn", "utils.inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
