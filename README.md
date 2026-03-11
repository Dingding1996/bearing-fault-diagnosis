# Bearing Fault Diagnosis via Motor Current Signals

Multi-class bearing fault classification using the Paderborn University benchmark dataset.
Sensorless fault detection in electric drive systems via DSP feature engineering + classical ML.

---

## Project Structure

```
ds_projects/
├── 00_download_dataset.py   # Dataset downloader (library + CLI)
├── 01_data_loader.py        # Data loading, label mapping, characteristic frequency calculation
├── 02_dsp_features.py       # DSP feature extraction (time / frequency / time-frequency / envelope)
├── 03_ml_classification.py  # ML classifiers (traditional ML + 1D-CNN + 2D-CNN)
├── 04_main_pipeline.py      # Entry point — run this
├── requirements.txt         # Pinned dependencies
├── README.md                # This file
└── paderborn_data/          # Downloaded dataset (gitignored)
    ├── rar/                 # Temporary .rar archives (deleted after extraction)
    └── mat/                 # Extracted .mat files, one folder per bearing
```

## Dataset

- **Source**: Paderborn University KAt-DataCenter — [Zenodo DOI 10.5281/zenodo.15845309](https://zenodo.org/records/15845309)
- **Paper**: Lessmeier et al., PHME 2016
- **Signals**: Dual-phase current (64 kHz) + vibration acceleration (64 kHz) + operating parameters
- **Bearings**: 32 experiments — 6 healthy, 12 artificially damaged, 14 real damage
- **Operating conditions**: 4 combinations of speed / torque / radial force

## Quick Start

```bash
# 1. Create and activate the environment
conda activate ds-py311
pip install -r requirements.txt

# 2. Run the full pipeline (downloads data automatically if missing)
python 04_main_pipeline.py
```

The pipeline calls `ensure_data()` at startup — if the dataset is already on disk it
skips the download and runs immediately.

To download data separately:
```bash
python 00_download_dataset.py --minimal    # ~2.4 GB, 15 bearings (recommended)
python 00_download_dataset.py             # full dataset, all 32 bearings
```

## Roadmap

### Phase 1 — DSP Signal Processing
- [x] Data loading and parsing
- [x] Time-domain features (RMS, peak, kurtosis, crest factor, etc.)
- [x] Frequency-domain features (FFT, PSD, spectral centroid, etc.)
- [x] Time-frequency analysis (STFT, CWT, wavelet packet decomposition)
- [x] Envelope analysis (Hilbert transform)
- [x] Characteristic frequency calculation (BPFO, BPFI, BSF, FTF)

### Phase 2 — Traditional ML Classification
- [x] Feature extraction pipeline
- [x] 7 classifiers (CART, RF, GBT, SVM, kNN, MLP, Ensemble)
- [ ] Reproduce paper baseline results
- [ ] Feature selection and optimisation

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

1. Lessmeier, C., et al. (2016). "Condition Monitoring of Bearing Damage in Electromechanical Drive Systems by Using Motor Current Signals." PHME 2016.
2. Randall, R.B. (2011). *Vibration-based Condition Monitoring.* Wiley.
3. Smith, W.A. & Randall, R.B. (2015). "Rolling element bearing diagnostics using the CWRU data." MSSP.
