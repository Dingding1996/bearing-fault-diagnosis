# Bearing Fault Diagnosis via Motor Current Signals

Multi-class bearing fault classification using the Paderborn University benchmark dataset.
Sensorless fault detection in electric drive systems via DSP feature engineering + classical ML.

---

## Project Structure

```
bearing-fault-diagnosis/
├── BearingFault_Training.ipynb   # End-to-end training pipeline (DSP → ML → MLflow)
├── requirements.txt              # Pinned dependencies (training environment)
├── requirements-inference.txt    # Minimal dependencies for the Docker inference service
├── Dockerfile                    # Container image for the FastAPI inference service
├── Dockerrun.aws.json            # AWS Elastic Beanstalk deployment configuration
├── docker-compose.yml            # Compose config — mounts mlruns/, exposes port 8000
├── README.md                     # This file
├── .github/workflows/
│   ├── ci.yml                    # Run unit tests on every push
│   └── deploy.yml                # Build image → push ECR → deploy Elastic Beanstalk
├── tests/
│   └── test_features.py          # Unit tests for DSP feature extraction
├── scripts/
│   └── upload_model_to_s3.py     # Upload mlruns/ to S3 after training
├── utils/
│   ├── download_dataset.py       # Dataset downloader (library + CLI)
│   ├── data_loader.py            # Data loading, label mapping, characteristic frequency calculation
│   ├── dsp_features.py           # DSP feature extraction (183 features: time / freq / TF / envelope)
│   ├── ml_classification.py      # ML pipeline (RF, GBT, XGBoost) with sklearn Pipeline + StratifiedGroupKFold
│   ├── inference_api.py          # FastAPI inference service (loaded by Docker)
│   └── plot_style.py             # Portfolio-wide figure styling
├── mlruns/                       # MLflow experiment tracking + model registry (gitignored)
└── paderborn_data/               # Downloaded dataset (gitignored)
    ├── rar/                      # Temporary .rar archives (deleted after extraction)
    └── mat/                      # Extracted .mat files, one folder per bearing
```

## Dataset

- **Source**: Paderborn University KAt-DataCenter — [Zenodo DOI 10.5281/zenodo.15845309](https://zenodo.org/records/15845309)
- **Paper**: Lessmeier et al., PHME 2016
- **Signals**: Dual-phase current (64 kHz) + vibration acceleration (64 kHz) + operating parameters
- **Bearings**: 32 experiments — 6 healthy, 12 artificially damaged, 14 real damage
- **Operating conditions**: 4 combinations of speed / torque / radial force

## Quick Start

### Training
```bash
# 1. Create and activate the environment
conda activate ds-py311
pip install -r requirements.txt

# 2. Open and run the training notebook (downloads data automatically if missing)
jupyter lab BearingFault_Training.ipynb
```

The notebook calls `ensure_data()` at startup — if the dataset is already on disk it
skips the download and runs immediately.

To download data separately:
```bash
python utils/download_dataset.py --minimal    # ~2.4 GB, 15 bearings (recommended)
python utils/download_dataset.py              # full dataset, all 32 bearings
```

### Inference API (Docker)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Run the training notebook first to populate `mlruns/` with the trained model and
per-condition signal statistics (`cond_signal_stats.pkl`).

```bash
# Start the API (builds image on first run)
docker compose up --build

# API is now running at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

**Endpoints:**

| Endpoint | Method | Input | Description |
|---|---|---|---|
| `/health` | GET | — | Health check + serving run ID |
| `/predict_mat` | POST | `.mat` file upload | Full pipeline: raw file → signal norm → DSP features → prediction |

**Example — upload a raw `.mat` file:**
```bash
curl -X POST http://localhost:8000/predict_mat \
  -F "file=@paderborn_data/mat/KA01/N15_M07_F10_KA01_1.mat"
```

**Response:**
```json
{
  "predictions": [1],
  "labels": ["OR_damage"],
  "run_id": "b1710f7b"
}
```

## Pipeline Overview

| Step | Detail |
|---|---|
| Signal normalisation | Per-condition z-score on raw vibration / current (training stats only) |
| DSP feature extraction | 183 features — time, frequency, WPD, envelope |
| Frequency normalisation | Hz → shaft orders for spectral features |
| Feature selection | `SelectFromModel` (RF, median threshold): 183 → ~92 features |
| CV strategy | `StratifiedGroupKFold` — bearing-aware, no cross-bearing leakage |
| Hyperparameter tuning | `RandomizedSearchCV` (30 iters, 3-fold inner CV) per model |
| ML classification | RF, GBT, XGBoost + Stacking meta-ensemble |
| Leakage prevention | sklearn `Pipeline` wraps scaler + model, re-fit inside every fold |
| Experiment tracking | MLflow registry — best model served by Docker API |

## Roadmap

### Phase 1 — DSP Signal Processing
- [x] Data loading and parsing
- [x] Time-domain features (RMS, peak, kurtosis, crest factor, etc.)
- [x] Frequency-domain features (FFT, PSD, spectral centroid, etc.)
- [x] Time-frequency analysis (STFT, CWT, wavelet packet decomposition)
- [x] Envelope analysis (Hilbert transform)
- [x] Characteristic frequency calculation (BPFO, BPFI, BSF, FTF)

### Phase 2 — Traditional ML Classification
- [x] Feature extraction pipeline (183 features per signal)
- [x] Feature selection (`SelectFromModel`, 183 → ~92 features)
- [x] 3 classifiers (RF, GBT, XGBoost) with RandomizedSearchCV tuning
- [x] Bearing-aware cross-validation (StratifiedGroupKFold — no leakage)
- [x] Per-condition signal normalisation (training stats only)
- [x] MLflow experiment tracking + model registry
- [x] FastAPI inference service + Docker deployment
- [x] CI/CD pipeline (GitHub Actions — test → build → push ECR → deploy)
- [x] Cloud deployment on AWS Elastic Beanstalk (eu-west-1)
- [ ] Reproduce paper baseline results
- [ ] Additional harmonic features (4x/5x BPFI/BPFO) or ICS2 cyclostationarity

### Phase 3 — Deep Learning
- [x] 1D-CNN model architecture
- [x] 2D-CNN model architecture (STFT / CWT image input)
- [ ] Training and tuning
- [ ] Comparison with traditional ML

### Phase 4 — Advanced Experiments
- [ ] Cross-condition generalisation (train on one operating condition, test on another)
- [ ] Artificial → real damage domain adaptation
- [ ] Current + vibration multimodal fusion
- [ ] Damage severity regression

## Key DSP Techniques

| Technique | Purpose | Function |
|-----------|---------|----------|
| FFT / PSD | Spectrum analysis, identify dominant frequencies | `frequency_domain_features()` |
| STFT | Time-frequency map, observe frequency drift | `stft_features()`, `signal_to_stft_image()` |
| CWT | Wavelet time-frequency map, multi-scale analysis | `cwt_features()`, `signal_to_cwt_image()` |
| WPD | Wavelet packet decomposition, sub-band energy | `wavelet_packet_features()` |
| Hilbert transform | Envelope analysis, extract fault characteristic frequencies | `envelope_analysis()` |
| Bandpass filter | Signal preprocessing, isolate frequency band of interest | Butterworth in `envelope_analysis()` |

## Bearing Characteristic Frequencies (6203 @ 1500 rpm)

| Name | Frequency | Meaning |
|------|-----------|---------|
| Shaft | 25.0 Hz | Shaft rotation frequency |
| BPFO | 76.1 Hz | Ball pass frequency, outer race |
| BPFI | 123.9 Hz | Ball pass frequency, inner race |
| BSF | 31.9 Hz | Ball spin frequency |
| FTF | 9.5 Hz | Fundamental train (cage) frequency |

## References

1. Lessmeier, C., et al. (2016). "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification" PHME 2016.
