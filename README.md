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

**Operating conditions** — the dataset covers 4 fixed settings, each a combination of shaft speed,
load torque, and radial force:

| Setting code | Speed | Torque | Radial force |
|---|---|---|---|
| N15_M07_F10 | 1500 rpm | 0.7 Nm | 1000 N |
| N09_M07_F10 | 900 rpm | 0.7 Nm | 1000 N |
| N15_M01_F10 | 1500 rpm | 0.1 Nm | 1000 N |
| N15_M07_F04 | 1500 rpm | 0.7 Nm | 400 N |

The same bearing under different conditions produces different signal amplitudes and spectral content
even when healthy — a model trained on N15 without normalisation would misclassify N09 signals as anomalies.
A two-stage pipeline addresses this before any feature is computed.

**Stage 1 — Signal-level z-score normalisation (before feature extraction)**

Each raw signal is standardised using the mean and standard deviation computed from
*training bearings only*, grouped by operating condition:

```
sig_norm(t) = ( sig(t) − μ_cond ) / σ_cond
```

`μ_cond` and `σ_cond` are estimated on training data only — no test-set statistics leak in.
Because fault energy contributes < 10 % of total signal power, the baseline is essentially
fault-free and generalises well to unseen bearings. After this step, a high RMS value
reflects a genuine mechanical anomaly rather than a high-load operating point.

**Stage 2 — Shaft-order conversion (after feature extraction)**

Spectral features (spectral centroid, peak frequency, spectral variance) are expressed in Hz
and scale linearly with shaft speed. Dividing by the shaft frequency `f_shaft = rpm / 60`
converts them to dimensionless shaft orders, so the same fault pattern produces the same
feature value regardless of operating speed. BPFO/BPFI/BSF/FTF envelope amplitudes are
also computed at the speed-corrected characteristic frequency for each file individually.

| Feature group | Raw unit | After Stage 2 |
|---|---|---|
| Spectral centroid, peak frequency, spectral std | Hz | shaft orders (÷ f_shaft) |
| Spectral variance | Hz² | orders² (÷ f_shaft²) |
| Envelope amplitudes, time-domain stats, WPD ratios | dimensionless | unchanged |

**DSP feature extraction** — 183 features per signal across four domains:

| Domain | Feature | What it measures |
|---|---|---|
| Time | RMS | Overall signal energy — first metric to rise as a bearing degrades |
| Time | Peak | Maximum absolute amplitude — sensitive to sudden large transients |
| Time | Kurtosis | Impulsiveness of the signal — healthy bearing ≈ 3, faulty bearing > 10; best early fault indicator |
| Time | Crest factor | Peak ÷ RMS — rises when isolated spikes appear against a low background |
| Time | Skewness | Asymmetry of the amplitude distribution — healthy signals are near-symmetric |
| Time | Shape factor | RMS ÷ mean absolute value — measures how "spiky" the waveform is relative to its average |
| Time | Impulse factor | Peak ÷ mean absolute value — amplifies single large impulses; sensitive to early pitting faults |
| Frequency | Spectral centroid | Weighted average frequency — shifts upward as fault energy moves to higher frequencies |
| Frequency | PSD band energies | Signal power in specific frequency bands — load-invariant when expressed as ratios |
| Frequency | Dominant frequency | Frequency of maximum energy — detects new periodic components introduced by a fault |
| Frequency | WPD sub-band energies | Signal energy in each of 8 equal-width frequency bands (level-3 decomposition, 2³ = 8 bands) — captures how fault energy is distributed across the spectrum without needing exact defect frequencies |
| Envelope | BPFO / BPFI / BSF / FTF amplitudes (1×–3× harmonics) | Strength of each defect frequency in the demodulated envelope — directly quantifies fault impulse repetition rate |
| Envelope | Inter-frequency ratios (BPFO/BPFI) | Ratio of outer-race to inner-race energy — primary discriminator between OR and IR damage types |

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

**AWS infrastructure** — three services work together:

| Service | Role |
|---|---|
| S3 | Stores MLflow model artifacts (`mlruns/`) and EB deployment bundles; persists the model outside the container so it survives re-deploys |
| ECR (Elastic Container Registry) | Private Docker registry — each push to `main` builds a new image tagged with the commit SHA and pushes it here |
| Elastic Beanstalk (eu-west-1) | Runs the FastAPI container behind an Nginx reverse proxy; EB pulls the new image from ECR on each deploy and routes port 80 → container port 8000 |

**Deploy flow on merge to `main`:**
1. Unit tests pass (pytest)
2. `mlruns/` synced from S3 → baked into the Docker image at build time
3. Image pushed to ECR with tag `<commit-sha>` and `latest`
4. `Dockerrun.aws.json` updated with the new image URI → zipped with `.platform/` Nginx config → uploaded to S3
5. EB creates a new application version from the S3 bundle and rolls it out to the environment

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
