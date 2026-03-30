# Bearing Fault Diagnosis via Vibration and Motor Current Signals

End-to-end predictive maintenance case study following the **CRISP-DM** methodology.
Sensorless bearing fault detection in electric drive systems using DSP feature engineering,
classical ML, and unsupervised anomaly detection — from raw `.mat` files to a deployed REST API.

---

## Business Understanding

**What are we detecting?**
Bearing faults produce impulsive spikes in vibration and current signals — elevated kurtosis,
crest factor, and energy peaks at characteristic defect frequencies (BPFO, BPFI, BSF, FTF).
The goal is to classify signals as Healthy, Outer Race damage, or Inner Race damage.

**Why this matters.**
Undetected bearing faults cause unplanned downtime. The cost of a missed fault (False Negative)
far exceeds the cost of a false alarm (False Positive), so **Recall is the primary metric**.
The binary healthy-vs-fault task (Section 6b–6c) directly maps to the deployment question:
*"Does this bearing need attention?"*

**Success metrics defined upfront:**
- Multi-class: F1-macro ≥ 0.85 on held-out bearings
- Binary fault detection: Recall ≥ 0.95 on held-out bearings
- Unsupervised baseline: Isolation Forest as a label-free comparator

---

## Data Understanding

- **Source**: Paderborn University KAt-DataCenter — [Zenodo DOI 10.5281/zenodo.15845309](https://zenodo.org/records/15845309)
- **Paper**: Lessmeier et al., PHME 2016
- **Signals**: Dual-phase current (64 kHz) + vibration acceleration (64 kHz)
- **Bearings**: 32 experiments — 6 healthy, 12 artificially damaged, 14 real damage
- **Operating conditions**: 4 combinations of speed / torque / radial force
- **Class distribution**: ~19% Healthy, ~40% OR damage, ~41% IR damage

**Key data challenges addressed:**
- Same bearing at different operating conditions produces different signal distributions even when healthy → per-condition z-score normalisation applied before feature extraction
- Labels are available (supervised task) but the unsupervised IF comparator validates that fault signatures are genuine anomalies, not learned class boundary artefacts

---

## Data Preparation

**Working condition normalisation** :
Per-condition z-score on raw signals using training statistics only. Ensures high RMS reflects a
genuine anomaly, not a high-load period.

**DSP feature extraction** — 183 features per signal across four domains:

| Domain | Features | What they capture |
|---|---|---|
| Time | RMS (root mean square energy), peak, kurtosis (impulsiveness), crest factor (peak-to-RMS ratio), skewness, shape factor, impulse factor | Overall signal health; kurtosis is the earliest indicator of bearing damage |
| Frequency | Spectral centroid, PSD (power spectral density), band energies, dominant frequency | Fault-specific energy at known defect frequencies; band ratios are load-invariant |
| Time-frequency | WPD (wavelet packet decomposition) — 8 sub-band energies at level-3 decomposition | Localises fault energy in both time and frequency simultaneously |
| Envelope | Amplitudes at BPFO, BPFI, BSF, FTF and their 1×–3× harmonics + inter-frequency ratios | Demodulated fault impulse rates; ratios discriminate outer-race from inner-race damage |

**Feature selection**: `SelectFromModel` with a Random Forest and median importance threshold — reduces 183 → ~92 features, keeping only those that contribute to class separation.

**Key DSP techniques:**

| Technique | Full name | Purpose |
|---|---|---|
| FFT / PSD | Fast Fourier Transform / Power Spectral Density | Converts time signal to frequency spectrum; identifies dominant fault frequencies |
| STFT | Short-Time Fourier Transform | Sliding FFT window — tracks how the frequency content changes over time |
| CWT | Continuous Wavelet Transform | Multi-resolution time-frequency map; resolves both fast transients and slow trends |
| WPD | Wavelet Packet Decomposition | Splits signal into 8 equal-width frequency sub-bands for energy analysis |
| Hilbert transform | — | Extracts the signal envelope; exposes the repetition rate of fault impulses |

**Bearing defect frequencies (SKF 6203 @ 1500 rpm, 25 Hz shaft):**

| Abbreviation | Full name | Frequency | Physical meaning |
|---|---|---|---|
| BPFO | Ball Pass Frequency, Outer race | 76.1 Hz | How often a rolling element strikes an outer-race defect |
| BPFI | Ball Pass Frequency, Inner race | 123.9 Hz | How often a rolling element strikes an inner-race defect |
| BSF | Ball Spin Frequency | 31.9 Hz | Rotation rate of an individual rolling element (ball) |
| FTF | Fundamental Train Frequency | 9.5 Hz | Rotation rate of the ball cage |

---

## Modeling

**Split strategy** — bearing-aware to prevent identity leakage:
`StratifiedGroupKFold` ensures no bearing spans train and test. A model that sees bearing K001
in training must not see K001 signals in validation — otherwise it memorises bearing-specific
noise rather than learning fault patterns that generalise to unseen assets.

**Model selection** :

| Label availability | Model | Rationale |
|---|---|---|
| Labels available | RF, GBT, XGBoost, Stacking | Supervised, full 3-class and binary |
| No labels | Isolation Forest + PCA | One-class, trained on healthy only; tests whether fault signatures are genuine anomalies |

**sklearn Pipeline**:
All preprocessing (StandardScaler) and model steps are wrapped in a single `Pipeline` object.
The scaler is fit only on training data inside each CV fold — no leakage possible.

**Hyperparameter tuning**: `RandomizedSearchCV` (30 iterations, 3-fold inner CV) for RF, GBT,
XGBoost. Manual `ParameterGrid` for Isolation Forest (unsupervised estimators do not accept `y`
in `fit`, so `GridSearchCV` cannot be used directly).

---

## Evaluation

**Primary metrics**:
Accuracy is not reported as a primary metric — on this dataset a trivial classifier achieves
high accuracy. F1-macro (equal weight per class) and binary Precision / Recall are used instead.

**Multi-class results** (3-class, held-out test bearings):

| Model | Accuracy | F1-macro |
|---|---|---|
| RF | — | — |
| Stacking | — | — |
| XGB | — | — |
| GBT | — | — |

*(Results populated at runtime — see Section 6 of the notebook.)*

**Binary results — Healthy vs Fault** (Section 6b–6c):

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Supervised (best) | ~0.97 | ~0.95 | ~0.96 |
| IsolationForest (unsupervised) | ~0.71 | ~0.98 | ~0.82 |

The Isolation Forest achieves Recall ≈ 0.98 with no label supervision, confirming that fault
signatures are geometrically anomalous relative to the healthy distribution. The gap in Precision
reflects the fundamental limit of unsupervised detection on a dataset where faults are not rare.

**Threshold selection** :
The IF decision threshold is tuned by sweeping percentiles of healthy training scores across
3-fold CV folds — decoupling model capacity from the operating point. Lowering the threshold
increases Recall (fewer missed faults) at the cost of Precision (more false alarms).

---

## Deployment

**MLflow** — experiment tracking and model registry. Every training run logs parameters,
F1-macro, and the best pipeline. The registered model is loaded by the inference service at startup.

**FastAPI + Docker** — the sklearn Pipeline is served as a REST API. New `.mat` files are
passed raw; the pipeline handles signal normalisation, DSP feature extraction, and prediction internally.

**CI/CD** — GitHub Actions: unit tests on every push, Docker image build → ECR push →
Elastic Beanstalk deploy on merge to `main`.

### Quick Start

```bash
# Training
conda activate ds-py311
pip install -r requirements.txt
jupyter lab BearingFault_Training.ipynb   # downloads data automatically if missing
```

**Live API — deployed on AWS Elastic Beanstalk (eu-west-1, always on):**

```bash
curl -X POST http://bearing-fault-env.eba-qprqprfs.eu-west-1.elasticbeanstalk.com/predict_mat \
  -F "file=@paderborn_data/mat/KA01/N15_M07_F10_KA01_1.mat"
# {"predictions": [1], "labels": ["OR_damage"], "run_id": "b1710f7b"}
```

Interactive docs: `http://bearing-fault-env.eba-qprqprfs.eu-west-1.elasticbeanstalk.com/docs`

```bash
# Local inference (requires training notebook to have been run first)
docker compose up --build
# → http://localhost:8000/docs
```

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Service status + serving run ID |
| `POST` | `/predict_mat` | Raw `.mat` upload → fault class prediction |

---

## Project Structure

```
bearing-fault-diagnosis/
├── BearingFault_Training.ipynb   # End-to-end CRISP-DM pipeline (DSP → ML → MLflow)
├── requirements.txt              # Pinned training dependencies
├── requirements-inference.txt    # Minimal inference dependencies (Docker)
├── Dockerfile                    # FastAPI inference service container
├── docker-compose.yml            # Local deployment (mounts mlruns/, port 8000)
├── Dockerrun.aws.json            # AWS Elastic Beanstalk configuration
├── .github/workflows/
│   ├── ci.yml                    # Unit tests on every push
│   └── deploy.yml                # Build → ECR → Elastic Beanstalk
├── tests/
│   └── test_features.py          # DSP feature extraction unit tests
├── utils/
│   ├── download_dataset.py       # Zenodo dataset downloader
│   ├── data_loader.py            # Signal loading, label mapping, characteristic frequencies
│   ├── dsp_features.py           # 183-feature DSP extraction pipeline
│   ├── ml_classification.py      # sklearn Pipeline + StratifiedGroupKFold training
│   ├── inference_api.py          # FastAPI service
│   └── plot_style.py             # Consistent figure styling
└── mlruns/                       # MLflow tracking + model registry (gitignored)
```

---

## Next Steps

- **Distribution shift monitoring**: track PSI on rolling feature windows in production; trigger retraining when PSI > 0.2 
- **Cross-condition generalisation**: train on one operating condition, evaluate on another — the harder and more realistic deployment test
- **Deep learning**: train the 1D-CNN and 2D-CNN architectures (defined, not yet trained) and compare against the classical ML baseline

---

## References

1. Lessmeier, C., et al. (2016). "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set for Data-Driven Classification." *PHME 2016*.
