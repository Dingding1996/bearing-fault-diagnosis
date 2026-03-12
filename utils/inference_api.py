"""
Bearing Fault Diagnosis — FastAPI Inference Service
====================================================
Loads the best model from MLflow and exposes a /predict endpoint.

Usage:
    python utils/inference_api.py
    # or
    uvicorn utils.inference_api:app --reload --port 8000
"""

import sys
from pathlib import Path

import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config — edit these to match your latest MLflow run
# ---------------------------------------------------------------------------
_BASE_DIR    = Path(__file__).parent.parent
MLRUNS_URI   = f"file:///{_BASE_DIR / 'mlruns'}"
EXPERIMENT   = "paderborn-bearing-fault"
CLASS_NAMES  = ["Healthy", "OR_damage", "IR_damage"]

# ---------------------------------------------------------------------------
# Load model at startup (once)
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLRUNS_URI)

_runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    order_by=["start_time DESC"],
)
if _runs.empty:
    raise RuntimeError(f"No runs found in experiment '{EXPERIMENT}'.")

_run_id  = _runs.iloc[0]["run_id"]
selector = mlflow.sklearn.load_model(f"runs:/{_run_id}/feature_selector")
model    = mlflow.sklearn.load_model(f"runs:/{_run_id}/best_model")

print(f"Model loaded from run: {_run_id[:8]}...")
print(f"Best model: {_runs.iloc[0].get('params.best_model', 'unknown')}")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bearing Fault Diagnosis API",
    description="3-class fault classification: Healthy / OR_damage / IR_damage",
    version="1.0.0",
)


class FeaturesInput(BaseModel):
    """Input schema: list of raw DSP feature vectors (171 features each)."""
    features: list[list[float]]


class PredictionOutput(BaseModel):
    """Output schema: predicted class index and label for each input."""
    predictions: list[int]
    labels: list[str]
    run_id: str


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "run_id": _run_id[:8]}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: FeaturesInput):
    """
    Predict bearing fault class from raw DSP feature vectors.

    Args:
        payload: JSON with key 'features' — list of feature vectors (171 values each).

    Returns:
        Predicted class indices and human-readable labels.

    Example:
        POST /predict
        {"features": [[f1, f2, ..., f171], [f1, f2, ..., f171]]}
    """
    try:
        X = np.array(payload.features, dtype=np.float32)
        X_fs = selector.transform(X)       # 171 -> ~85 features
        y_pred = model.predict(X_fs)       # scaler + classifier inside
        labels = [CLASS_NAMES[i] for i in y_pred]
        return PredictionOutput(
            predictions=y_pred.tolist(),
            labels=labels,
            run_id=_run_id[:8],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
