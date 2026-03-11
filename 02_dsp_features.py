"""
Paderborn Bearing Dataset - DSP Analysis & Feature Extraction
==============================================================
Signal processing pipeline for bearing fault diagnosis:
  1. Time-domain features
  2. Frequency-domain features (FFT, PSD)
  3. Time-frequency features (STFT, CWT, WPD)
  4. Envelope analysis (for bearing characteristic frequencies)

Author: [Your Name]
"""

import numpy as np
from scipy.signal import stft, welch, hilbert, butter, filtfilt, sosfilt
from scipy.signal import butter, sosfiltfilt
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, List, Optional

# Try to import pywt; if unavailable, use FFT-based alternatives
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _simple_cwt(signal, scales):
    """Simple CWT using convolution with Morlet wavelet (fallback when pywt unavailable)."""
    N = len(signal)
    coeffs = np.zeros((len(scales), N))
    for i, s in enumerate(scales):
        # Morlet wavelet
        t_wav = np.arange(-4*s, 4*s+1) / s
        wavelet = np.exp(1j * 2 * np.pi * t_wav) * np.exp(-t_wav**2 / 2)
        wavelet = wavelet.real / np.sqrt(s)
        conv = np.convolve(signal, wavelet, mode='same')
        coeffs[i] = conv
    return coeffs


# ============================================================
# 1. TIME-DOMAIN FEATURES
# ============================================================

def time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract standard time-domain statistical features.
    These are the most basic features used in CM.
    
    Args:
        signal: 1D time-series signal
        
    Returns:
        Dictionary of feature name -> value
    """
    n = len(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    
    features = {
        # Basic statistics
        'mean': mean,
        'std': std,
        'rms': np.sqrt(np.mean(signal**2)),
        'peak': np.max(np.abs(signal)),
        'peak_to_peak': np.max(signal) - np.min(signal),
        
        # Shape descriptors
        'skewness': float(np.mean(((signal - mean) / (std + 1e-10))**3)),
        'kurtosis': float(np.mean(((signal - mean) / (std + 1e-10))**4)),
        
        # Impulse indicators (sensitive to bearing faults)
        'crest_factor': np.max(np.abs(signal)) / (np.sqrt(np.mean(signal**2)) + 1e-10),
        'shape_factor': np.sqrt(np.mean(signal**2)) / (np.mean(np.abs(signal)) + 1e-10),
        'impulse_factor': np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-10),
        'clearance_factor': np.max(np.abs(signal)) / (np.mean(np.sqrt(np.abs(signal)))**2 + 1e-10),
        
        # Energy
        'energy': np.sum(signal**2),
        'entropy': _signal_entropy(signal),
    }
    
    return features


def _signal_entropy(signal: np.ndarray, n_bins: int = 100) -> float:
    """Calculate Shannon entropy of signal amplitude distribution."""
    hist, _ = np.histogram(signal, bins=n_bins, density=True)
    hist = hist[hist > 0]
    bin_width = (signal.max() - signal.min()) / n_bins
    return -np.sum(hist * np.log2(hist + 1e-10) * bin_width)


# ============================================================
# 2. FREQUENCY-DOMAIN FEATURES
# ============================================================

def frequency_domain_features(signal: np.ndarray, fs: int) -> Dict[str, float]:
    """
    Extract frequency-domain features using FFT and PSD.
    
    Args:
        signal: 1D time-series signal
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary of feature name -> value
    """
    N = len(signal)
    
    # FFT
    freqs = fftfreq(N, 1/fs)[:N//2]
    fft_mag = np.abs(fft(signal))[:N//2] * 2 / N
    
    # PSD via Welch's method
    f_psd, psd = welch(signal, fs=fs, nperseg=min(4096, N//4))
    
    # Spectral features
    total_power = np.sum(psd)
    f_mean = np.sum(f_psd * psd) / (total_power + 1e-10)
    f_var = np.sum((f_psd - f_mean)**2 * psd) / (total_power + 1e-10)
    
    features = {
        # Spectral center and spread
        'spectral_centroid': f_mean,
        'spectral_variance': f_var,
        'spectral_std': np.sqrt(f_var),
        
        # Spectral shape
        'spectral_skewness': np.sum((f_psd - f_mean)**3 * psd) / ((f_var**1.5 + 1e-10) * total_power),
        'spectral_kurtosis': np.sum((f_psd - f_mean)**4 * psd) / ((f_var**2 + 1e-10) * total_power),
        
        # Energy in different bands
        'total_spectral_energy': total_power,
        'peak_frequency': f_psd[np.argmax(psd)],
        'peak_amplitude': np.max(fft_mag),
        
        # Band energy ratios (useful for bearing diagnosis)
        'energy_0_500Hz': _band_energy(f_psd, psd, 0, 500),
        'energy_500_2000Hz': _band_energy(f_psd, psd, 500, 2000),
        'energy_2000_8000Hz': _band_energy(f_psd, psd, 2000, 8000),
        'energy_8000_16000Hz': _band_energy(f_psd, psd, 8000, 16000),
    }
    
    return features


def _band_energy(freqs: np.ndarray, psd: np.ndarray, 
                  f_low: float, f_high: float) -> float:
    """Calculate energy in a frequency band."""
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(psd[mask]))


# ============================================================
# 3. TIME-FREQUENCY FEATURES (STFT & Wavelet)
# ============================================================

def stft_features(signal: np.ndarray, fs: int, 
                  nperseg: int = 2048) -> Dict[str, float]:
    """
    Extract features from Short-Time Fourier Transform.
    
    Args:
        signal: 1D time-series signal
        fs: Sampling frequency
        nperseg: Window size for STFT
        
    Returns:
        Dictionary of features
    """
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg*3//4)
    mag = np.abs(Zxx)
    
    # Time-averaged spectrum statistics
    mean_spectrum = np.mean(mag, axis=1)
    std_spectrum = np.std(mag, axis=1)
    
    features = {
        'stft_mean_energy': float(np.mean(mag**2)),
        'stft_std_energy': float(np.std(mag**2)),
        'stft_max_energy': float(np.max(mag**2)),
        'stft_spectral_flatness': float(
            np.exp(np.mean(np.log(mean_spectrum + 1e-10))) / 
            (np.mean(mean_spectrum) + 1e-10)
        ),
    }
    
    return features


def wavelet_packet_features(signal: np.ndarray, wavelet: str = 'db4', 
                             level: int = 3) -> Dict[str, float]:
    """
    Extract features using Wavelet Packet Decomposition (WPD).
    Falls back to frequency-band energy if pywt is not available.
    
    Args:
        signal: 1D time-series signal
        wavelet: Wavelet type
        level: Decomposition level
        
    Returns:
        Dictionary with wavelet/band energy features
    """
    features = {}
    
    if HAS_PYWT:
        wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=level)
        nodes = [node.path for node in wp.get_level(level, 'freq')]
        
        energies = []
        total_energy = 0
        for i, node_path in enumerate(nodes):
            coeffs = wp[node_path].data
            energy = np.sum(coeffs**2)
            energies.append(energy)
            total_energy += energy
            features[f'energy_band_{i}'] = float(energy)
    else:
        # Fallback: split FFT into 2^level equal frequency bands
        n_bands = 2 ** level
        fft_mag = np.abs(fft(signal))[:len(signal)//2]
        band_size = len(fft_mag) // n_bands
        
        energies = []
        total_energy = 0
        for i in range(n_bands):
            start = i * band_size
            end = (i + 1) * band_size if i < n_bands - 1 else len(fft_mag)
            energy = float(np.sum(fft_mag[start:end]**2))
            energies.append(energy)
            total_energy += energy
            features[f'energy_band_{i}'] = energy
    
    # Normalized energies
    for i, e in enumerate(energies):
        features[f'energy_ratio_{i}'] = float(e / (total_energy + 1e-10))
    
    # Entropy
    probs = np.array(energies) / (total_energy + 1e-10)
    features['entropy'] = float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    return features


def cwt_features(signal: np.ndarray, fs: int, 
                  scales: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Extract features using Continuous Wavelet Transform (CWT).
    
    Args:
        signal: 1D time-series signal
        fs: Sampling frequency
        scales: CWT scales (if None, auto-generated)
        
    Returns:
        Dictionary of CWT features
    """
    if scales is None:
        scales = np.arange(1, 128)
    
    if HAS_PYWT:
        coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
    else:
        # Use scipy's cwt with morlet wavelet
        widths = scales
        coeffs = _simple_cwt(signal, widths)
    
    power = np.abs(coeffs)**2
    
    features = {
        'cwt_mean_power': float(np.mean(power)),
        'cwt_std_power': float(np.std(power)),
        'cwt_max_power': float(np.max(power)),
        'cwt_energy': float(np.sum(power)),
    }
    
    # Energy per scale band
    n_bands = 4
    band_size = len(scales) // n_bands
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else len(scales)
        features[f'cwt_band_{i}_energy'] = float(np.sum(power[start:end]))
    
    return features


# ============================================================
# 4. ENVELOPE ANALYSIS (Key technique for bearing diagnosis)
# ============================================================

def envelope_analysis(signal: np.ndarray, fs: int, 
                      band: Tuple[float, float] = (500, 10000),
                      n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform envelope analysis - the gold standard for bearing fault detection.
    
    Steps:
    1. Bandpass filter around resonance frequency
    2. Hilbert transform to get analytic signal
    3. Extract envelope (magnitude of analytic signal)
    4. FFT of envelope to get envelope spectrum
    
    Args:
        signal: 1D vibration or current signal
        fs: Sampling frequency
        band: Bandpass filter range (f_low, f_high) in Hz
        n_fft: FFT length for envelope spectrum
        
    Returns:
        (frequencies, envelope_spectrum_magnitude)
    """
    # Step 1: Bandpass filter
    sos = butter(5, [band[0], band[1]], btype='band', fs=fs, output='sos')
    filtered = sosfiltfilt(sos, signal)
    
    # Step 2: Hilbert transform -> analytic signal
    analytic = hilbert(filtered)
    
    # Step 3: Envelope (magnitude)
    envelope = np.abs(analytic)
    
    # Remove DC component
    envelope = envelope - np.mean(envelope)
    
    # Step 4: FFT of envelope
    if n_fft is None:
        n_fft = len(envelope)
    
    freqs = fftfreq(n_fft, 1/fs)[:n_fft//2]
    env_fft = np.abs(fft(envelope, n=n_fft))[:n_fft//2] * 2 / n_fft
    
    return freqs, env_fft


def envelope_features(signal: np.ndarray, fs: int,
                      characteristic_freqs: Dict[str, float],
                      band: Tuple[float, float] = (500, 10000),
                      tolerance: float = 3.0) -> Dict[str, float]:
    """
    Extract features from envelope spectrum at characteristic frequencies.
    
    Args:
        signal: 1D signal
        fs: Sampling frequency
        characteristic_freqs: Dict with BPFO, BPFI, BSF, FTF frequencies
        band: Bandpass filter range
        tolerance: Frequency tolerance in Hz for peak search
        
    Returns:
        Dictionary of envelope features
    """
    freqs, env_spectrum = envelope_analysis(signal, fs, band)
    
    features = {}
    
    for name, f_char in characteristic_freqs.items():
        if f_char <= 0:
            continue
            
        # Search for peak near characteristic frequency and its harmonics
        for harmonic in [1, 2, 3]:
            f_target = f_char * harmonic
            mask = (freqs >= f_target - tolerance) & (freqs <= f_target + tolerance)
            
            if np.any(mask):
                peak_amp = np.max(env_spectrum[mask])
                features[f'env_{name}_{harmonic}x'] = float(peak_amp)
            else:
                features[f'env_{name}_{harmonic}x'] = 0.0
    
    return features


# ============================================================
# 5. COMBINED FEATURE EXTRACTION
# ============================================================

def extract_all_features(signal: np.ndarray, fs: int, 
                         signal_type: str = 'current',
                         characteristic_freqs: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract all features from a single signal.
    
    Args:
        signal: 1D signal array
        fs: Sampling frequency
        signal_type: 'current' or 'vibration' (affects some parameters)
        characteristic_freqs: Bearing characteristic frequencies
        
    Returns:
        Combined feature dictionary
    """
    features = {}
    
    # Time-domain
    td = time_domain_features(signal)
    features.update({f'td_{k}': v for k, v in td.items()})
    
    # Frequency-domain
    fd = frequency_domain_features(signal, fs)
    features.update({f'fd_{k}': v for k, v in fd.items()})
    
    # Wavelet packet decomposition
    wpd = wavelet_packet_features(signal, wavelet='db4', level=3)
    features.update({f'wpd_{k}': v for k, v in wpd.items()})
    
    # Envelope analysis (if characteristic frequencies provided)
    if characteristic_freqs is not None:
        if signal_type == 'vibration':
            band = (500, 10000)
        else:  # current
            band = (50, 5000)
        
        env = envelope_features(signal, fs, characteristic_freqs, band)
        features.update({f'env_{k}': v for k, v in env.items()})
    
    return features


def extract_features_from_bearing(bearing_signal, 
                                   use_current: bool = True,
                                   use_vibration: bool = True,
                                   characteristic_freqs: Optional[Dict] = None) -> Dict[str, float]:
    """
    Extract features from a BearingSignal object (both channels).
    
    Args:
        bearing_signal: BearingSignal dataclass instance
        use_current: Include current signal features
        use_vibration: Include vibration signal features
        characteristic_freqs: Bearing characteristic frequencies
        
    Returns:
        Combined feature dictionary from all channels
    """
    features = {}
    fs = bearing_signal.fs
    
    if use_current:
        c1_feats = extract_all_features(
            bearing_signal.phase_current_1, fs, 'current', characteristic_freqs)
        features.update({f'c1_{k}': v for k, v in c1_feats.items()})
        
        c2_feats = extract_all_features(
            bearing_signal.phase_current_2, fs, 'current', characteristic_freqs)
        features.update({f'c2_{k}': v for k, v in c2_feats.items()})
    
    if use_vibration:
        vib_feats = extract_all_features(
            bearing_signal.vibration, fs, 'vibration', characteristic_freqs)
        features.update({f'vib_{k}': v for k, v in vib_feats.items()})
    
    return features


# ============================================================
# 6. GENERATE 2D IMAGES FOR CNN INPUT
# ============================================================

def signal_to_stft_image(signal: np.ndarray, fs: int,
                          nperseg: int = 256, 
                          max_freq: float = 5000,
                          target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Convert signal to STFT spectrogram image for CNN input.
    
    Args:
        signal: 1D signal
        fs: Sampling frequency
        nperseg: STFT window size
        max_freq: Maximum frequency to include
        target_size: Output image size (height, width)
        
    Returns:
        2D numpy array (spectrogram image), normalized to [0, 1]
    """
    from scipy.ndimage import zoom
    
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=nperseg*3//4)
    mag = np.abs(Zxx)
    
    # Limit frequency range
    freq_mask = f <= max_freq
    mag = mag[freq_mask, :]
    
    # Convert to dB scale
    mag_db = 20 * np.log10(mag + 1e-10)
    
    # Resize to target
    zoom_factors = (target_size[0] / mag_db.shape[0], 
                    target_size[1] / mag_db.shape[1])
    img = zoom(mag_db, zoom_factors)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    
    return img


def signal_to_cwt_image(signal: np.ndarray, fs: int,
                         n_scales: int = 128,
                         target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Convert signal to CWT scalogram image for CNN input.
    
    Args:
        signal: 1D signal
        fs: Sampling frequency
        n_scales: Number of wavelet scales
        target_size: Output image size
        
    Returns:
        2D numpy array (scalogram image), normalized to [0, 1]
    """
    from scipy.ndimage import zoom
    
    scales = np.arange(1, n_scales + 1)
    if HAS_PYWT:
        coeffs, freqs = pywt.cwt(signal, scales, 'morl', sampling_period=1/fs)
    else:
        coeffs = _simple_cwt(signal, scales)
    power = np.abs(coeffs)**2
    
    # Log scale for better visualization
    power_db = 10 * np.log10(power + 1e-10)
    
    # Resize
    zoom_factors = (target_size[0] / power_db.shape[0],
                    target_size[1] / power_db.shape[1])
    img = zoom(power_db, zoom_factors)
    
    # Normalize
    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
    
    return img


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    # Generate a test signal
    fs = 64000
    t = np.arange(0, 1, 1/fs)
    
    # Simulate: 100 Hz supply + bearing fault at 76 Hz (BPFO)
    signal = 2.0 * np.sin(2 * np.pi * 100 * t)  # supply frequency
    signal += 0.05 * np.sin(2 * np.pi * 76 * t)  # BPFO
    signal += 0.1 * np.random.randn(len(t))       # noise
    
    print("=== Testing Feature Extraction ===")
    
    td = time_domain_features(signal)
    print(f"\nTime-domain features ({len(td)}):")
    for k, v in td.items():
        print(f"  {k}: {v:.6f}")
    
    fd = frequency_domain_features(signal, fs)
    print(f"\nFrequency-domain features ({len(fd)}):")
    for k, v in fd.items():
        print(f"  {k}: {v:.6f}")
    
    wpd = wavelet_packet_features(signal, level=3)
    print(f"\nWPD features ({len(wpd)}):")
    for k, v in list(wpd.items())[:5]:
        print(f"  {k}: {v:.6f}")
    print(f"  ... ({len(wpd)} total)")
    
    print(f"\nSTFT image shape: {signal_to_stft_image(signal, fs).shape}")
    print(f"CWT image shape: {signal_to_cwt_image(signal[:fs//4], fs).shape}")  # shorter for speed
    
    print("\nAll tests passed!")
