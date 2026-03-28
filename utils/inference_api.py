"""
Bearing Fault Diagnosis — FastAPI Inference Service
====================================================
Loads the best model from MLflow and exposes one endpoint:

  POST /predict_mat  — accepts a raw .mat file; DSP feature extraction is done
                       server-side so the caller only needs to supply the file

Usage:
    python utils/inference_api.py
    # or
    uvicorn utils.inference_api:app --reload --port 8000
"""

import dataclasses
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

# Ensure the project root is on sys.path so utils sub-modules are importable
# whether the file is run directly or via `uvicorn utils.inference_api:app`.
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import (  # noqa: E402
    OPERATING_CONDITIONS,
    calc_characteristic_frequencies,
    load_mat_file,
)
from utils.dsp_features import extract_features_from_bearing  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR      = Path(__file__).parent.parent
MLRUNS_URI     = f"file:///{_BASE_DIR / 'mlruns'}"
CLASS_NAMES    = ["Healthy", "OR_damage", "IR_damage"]
MODEL_NAME     = "bearing_fault_xgb"       # registered model name in MLflow Registry
SELECTOR_NAME  = "bearing_fault_selector"  # registered selector name in MLflow Registry

# ---------------------------------------------------------------------------
# Load model at startup (once)
# ---------------------------------------------------------------------------
mlflow.set_tracking_uri(MLRUNS_URI)

_client = mlflow.MlflowClient()


def _load_registered(name: str) -> object:
    """Load the latest version of a registered sklearn model.

    Reads storage_location from the version meta.yaml directly — bypasses the
    MLflow client API which no longer exposes storage_location in MLflow 3.x,
    and also avoids cross-OS path issues (Windows paths in Linux containers).
    """
    import yaml  # bundled with mlflow

    models_dir = _BASE_DIR / "mlruns" / "models" / name
    version_dirs = sorted(models_dir.glob("version-*"), key=lambda p: int(p.name.split("-")[1]))
    if not version_dirs:
        raise RuntimeError(f"No versions found for registered model '{name}'")

    meta = yaml.safe_load((version_dirs[-1] / "meta.yaml").read_text())
    storage_location: str = meta["storage_location"]

    # Convert Windows absolute URI (file:///C:\...\mlruns/...) to container path
    normalised = storage_location.replace("\\", "/")
    idx = normalised.find("mlruns/")
    if idx == -1:
        raise RuntimeError(f"Cannot find 'mlruns/' in storage_location: {storage_location}")
    relative = normalised[idx + len("mlruns/"):]
    artifact_path = _BASE_DIR / "mlruns" / relative

    return mlflow.sklearn.load_model(str(artifact_path))


selector = _load_registered(SELECTOR_NAME)
model    = _load_registered(MODEL_NAME)

_latest_version = _client.get_registered_model(MODEL_NAME).latest_versions[-1]
_run_id = _latest_version.run_id or "registry"

print(f"Selector loaded : models:/{SELECTOR_NAME}/latest")
print(f"Model loaded    : models:/{MODEL_NAME}/latest  (run {_run_id[:8]}...)")

# ---------------------------------------------------------------------------
# Load per-condition signal normalisation stats (saved by MLflow logging cell)
# ---------------------------------------------------------------------------
_sig_stats_path = _BASE_DIR / "mlruns" / "cond_signal_stats.pkl"
if _sig_stats_path.exists():
    with open(_sig_stats_path, "rb") as _f:
        _COND_SIGNAL_STATS: dict = pickle.load(_f)
    print(f"Signal stats loaded from {_sig_stats_path.name}  "
          f"(conditions: {list(_COND_SIGNAL_STATS.keys())})")
else:
    _COND_SIGNAL_STATS = {}
    print("WARNING: cond_signal_stats.pkl not found — signal normalisation skipped")


# ---------------------------------------------------------------------------
# Preprocessing helpers (must mirror the notebook training pipeline exactly)
# ---------------------------------------------------------------------------

def _normalize_freq_features(feats: Dict[str, float], f_shaft: float) -> Dict[str, float]:
    """Convert absolute-frequency features to speed-invariant shaft orders.

    Mirrors the identical function in the training notebook (Section 3f-ii).
    Must stay in sync with any changes made there.
    """
    hz_suffixes = ('_spectral_centroid', '_spectral_std', '_peak_frequency')
    for k in list(feats):
        if any(k.endswith(s) for s in hz_suffixes):
            feats[k] = feats[k] / (f_shaft + 1e-10)
    for k in list(feats):
        if k.endswith('_spectral_variance'):
            feats[k] = feats[k] / (f_shaft ** 2 + 1e-10)
    return feats

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bearing Fault Diagnosis API",
    description=(
        "3-class fault classification: Healthy / OR_damage / IR_damage\n\n"
        "Upload a raw `.mat` file; the full DSP feature extraction pipeline runs server-side."
    ),
    version="1.2.0",
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionOutput(BaseModel):
    """Output schema: predicted class index and label for each input."""
    predictions: list[int]
    labels: list[str]
    run_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "run_id": _run_id[:8]}


@app.post("/predict_mat", response_model=PredictionOutput)
async def predict_mat(file: UploadFile = File(...)):
    """
    Predict bearing fault class from a raw Paderborn .mat file.

    The full pipeline runs server-side:
      .mat file → load signals → DSP feature extraction (171 features)
               → feature selection → model prediction → fault label

    The filename must follow Paderborn naming convention so the operating
    condition (speed / torque) can be parsed automatically, e.g.:
      N15_M07_F10_K001_1.mat

    Returns:
        Predicted class index, label, and serving run ID.
    """
    if not file.filename.endswith(".mat"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .mat file.")

    tmp_path = None
    try:
        # Write upload to a named temp file so load_mat_file() can parse it.
        # The suffix preserves the original filename for parse_filename().
        with tempfile.NamedTemporaryFile(
            suffix=f"_{file.filename}", delete=False
        ) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        sig = load_mat_file(tmp_path)

        if sig.setting not in OPERATING_CONDITIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown operating condition '{sig.setting}'. "
                       f"Expected one of: {list(OPERATING_CONDITIONS.keys())}",
            )

        rpm        = OPERATING_CONDITIONS[sig.setting]["speed_rpm"]
        char_freqs = calc_characteristic_frequencies(rpm)

        # Apply per-condition signal normalisation (mirrors Section 3f of training notebook).
        _st = _COND_SIGNAL_STATS.get(sig.setting)
        if _st:
            sig = dataclasses.replace(
                sig,
                vibration       = (sig.vibration       - _st["vib_mean"])  / _st["vib_std"],
                phase_current_1 = (sig.phase_current_1 - _st["cur1_mean"]) / _st["cur1_std"],
                phase_current_2 = (sig.phase_current_2 - _st["cur2_mean"]) / _st["cur2_std"],
            )

        feats      = extract_features_from_bearing(
            sig,
            use_current=True,
            use_vibration=True,
            characteristic_freqs=char_freqs,
        )

        X      = np.array([list(feats.values())], dtype=np.float32)
        X_fs   = selector.transform(X)
        y_pred = model.predict(X_fs)
        labels = [CLASS_NAMES[i] for i in y_pred]

        return PredictionOutput(
            predictions=y_pred.tolist(),
            labels=labels,
            run_id=_run_id[:8],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
