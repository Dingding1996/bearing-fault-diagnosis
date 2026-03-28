"""
Unit tests for DSP feature extraction and bearing characteristic frequencies.

All tests use synthetic numpy signals — no dataset download required.
These run in CI without any model artifacts or AWS credentials.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import BearingSignal, calc_characteristic_frequencies
from utils.dsp_features import extract_features_from_bearing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS        = 64_000    # Paderborn sampling rate (Hz)
N_SAMPLES = 256_000   # 4 seconds @ 64 kHz
N_4K      = 16_000    # 4 seconds @ 4 kHz
RPM       = 1500


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_signal() -> BearingSignal:
    """Synthetic bearing signal: 50 Hz sine + white noise.

    Uses a fixed seed for full determinism across test runs.
    All required BearingSignal fields are populated with plausible values.
    """
    rng  = np.random.default_rng(42)
    t64k = np.arange(N_SAMPLES, dtype=np.float64) / FS
    base = np.sin(2 * np.pi * 50 * t64k) + 0.1 * rng.standard_normal(N_SAMPLES)

    return BearingSignal(
        bearing_code    = "K001",
        setting         = "N15_M07_F10",
        measurement_id  = 0,
        label_3class    = 0,
        label_name      = "Healthy",
        damage_origin   = "healthy",
        phase_current_1 = base.astype(np.float32),
        phase_current_2 = (base * 0.5).astype(np.float32),
        vibration       = (base * 0.8).astype(np.float32),
        time_64k        = t64k.astype(np.float32),
        speed           = np.full(N_4K, RPM,   dtype=np.float32),
        torque          = np.full(N_4K, 0.7,   dtype=np.float32),
        force           = np.full(N_4K, 1000., dtype=np.float32),
        time_4k         = np.arange(N_4K, dtype=np.float32) / 4000,
        temperature     = np.full(4, 25., dtype=np.float32),
    )


@pytest.fixture(scope="module")
def char_freqs() -> dict:
    return calc_characteristic_frequencies(RPM)


@pytest.fixture(scope="module")
def full_features(synthetic_signal, char_freqs) -> dict:
    """Cache the full feature dict so it is only computed once per test session."""
    return extract_features_from_bearing(
        synthetic_signal,
        use_current=True,
        use_vibration=True,
        characteristic_freqs=char_freqs,
    )


# ---------------------------------------------------------------------------
# Tests — characteristic frequencies
# ---------------------------------------------------------------------------
class TestCharacteristicFrequencies:
    def test_returns_all_expected_keys(self, char_freqs):
        for key in ("shaft_freq", "BPFO", "BPFI", "BSF", "FTF"):
            assert key in char_freqs, f"Missing key: {key}"

    def test_all_frequencies_positive(self, char_freqs):
        for k, v in char_freqs.items():
            assert v > 0, f"Frequency '{k}' = {v} must be positive"

    def test_bpfi_greater_than_bpfo(self, char_freqs):
        """For 6203 geometry inner-ring passes the balls faster than outer-ring."""
        assert char_freqs["BPFI"] > char_freqs["BPFO"]

    def test_shaft_freq_matches_rpm(self, char_freqs):
        expected = RPM / 60
        assert abs(char_freqs["shaft_freq"] - expected) < 1e-9

    @pytest.mark.parametrize("rpm", [900, 1500])
    def test_all_frequencies_scale_linearly_with_rpm(self, rpm):
        """All bearing frequencies are proportional to shaft speed."""
        base  = calc_characteristic_frequencies(1500)
        scaled = calc_characteristic_frequencies(rpm)
        ratio = rpm / 1500
        for key in ("BPFO", "BPFI", "BSF", "FTF", "shaft_freq"):
            np.testing.assert_allclose(
                scaled[key], base[key] * ratio, rtol=1e-9,
                err_msg=f"{key} did not scale correctly",
            )


# ---------------------------------------------------------------------------
# Tests — feature extraction (full pipeline)
# ---------------------------------------------------------------------------
class TestExtractFeaturesFull:
    def test_returns_nonempty_dict(self, full_features):
        assert isinstance(full_features, dict)
        assert len(full_features) > 0

    def test_all_values_finite(self, full_features):
        bad = {k: v for k, v in full_features.items() if not np.isfinite(v)}
        assert not bad, f"Non-finite features: {list(bad.keys())}"

    def test_all_values_are_floats(self, full_features):
        non_float = {k: type(v) for k, v in full_features.items()
                     if not isinstance(v, (float, int, np.floating, np.integer))}
        assert not non_float, f"Non-numeric features: {non_float}"

    def test_feature_count_above_minimum(self, full_features):
        """Extraction produces at least 100 features for 3-channel input."""
        assert len(full_features) >= 100

    def test_expected_channel_prefixes_present(self, full_features):
        keys = set(full_features.keys())
        assert any(k.startswith("vib_") for k in keys), "Vibration features missing"
        assert any(k.startswith("c1_")  for k in keys), "Current-1 features missing"
        assert any(k.startswith("c2_")  for k in keys), "Current-2 features missing"

    def test_time_domain_features_present(self, full_features):
        keys = set(full_features.keys())
        for suffix in ("rms", "kurtosis", "skewness", "crest_factor"):
            assert any(suffix in k for k in keys), f"Time-domain feature '{suffix}' missing"

    def test_frequency_domain_features_present(self, full_features):
        keys = set(full_features.keys())
        for suffix in ("spectral_centroid", "spectral_variance", "peak_frequency"):
            assert any(suffix in k for k in keys), f"Freq-domain feature '{suffix}' missing"

    def test_deterministic_across_two_calls(self, synthetic_signal, char_freqs):
        feats1 = extract_features_from_bearing(
            synthetic_signal, use_current=True, use_vibration=True,
            characteristic_freqs=char_freqs,
        )
        feats2 = extract_features_from_bearing(
            synthetic_signal, use_current=True, use_vibration=True,
            characteristic_freqs=char_freqs,
        )
        assert feats1 == feats2, "Feature extraction is not deterministic"

    def test_feature_count_stable(self, synthetic_signal, char_freqs, full_features):
        """Count must not drift — a change would silently break trained models."""
        feats2 = extract_features_from_bearing(
            synthetic_signal, use_current=True, use_vibration=True,
            characteristic_freqs=char_freqs,
        )
        assert len(feats2) == len(full_features)


# ---------------------------------------------------------------------------
# Tests — channel toggle flags
# ---------------------------------------------------------------------------
class TestChannelFlags:
    def test_vibration_only_no_current_keys(self, synthetic_signal, char_freqs):
        feats = extract_features_from_bearing(
            synthetic_signal, use_current=False, use_vibration=True,
            characteristic_freqs=char_freqs,
        )
        assert any(k.startswith("vib_") for k in feats), "Vibration features expected"
        assert not any(k.startswith("c1_") for k in feats), "c1_ should be absent"
        assert not any(k.startswith("c2_") for k in feats), "c2_ should be absent"

    def test_current_only_no_vibration_keys(self, synthetic_signal, char_freqs):
        feats = extract_features_from_bearing(
            synthetic_signal, use_current=True, use_vibration=False,
            characteristic_freqs=char_freqs,
        )
        assert any(k.startswith("c1_") for k in feats), "c1_ features expected"
        assert not any(k.startswith("vib_") for k in feats), "vib_ should be absent"

    def test_fewer_features_with_vibration_only(self, synthetic_signal, char_freqs,
                                                 full_features):
        """Disabling current channels must strictly reduce feature count."""
        feats_vib = extract_features_from_bearing(
            synthetic_signal, use_current=False, use_vibration=True,
            characteristic_freqs=char_freqs,
        )
        assert len(feats_vib) < len(full_features)
