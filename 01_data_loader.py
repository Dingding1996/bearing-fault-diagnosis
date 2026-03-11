"""
Paderborn Bearing Dataset - Data Loader
========================================
Loads .mat files from the Paderborn University bearing dataset
and provides structured access to signals and metadata.

Author: [Your Name]
Project: Bearing Fault Diagnosis via Motor Current Signals
"""

import os
import numpy as np
import scipy.io as sio
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


# ============================================================
# Label Mapping (from paper Tables 4, 5, 7)
# ============================================================

# Healthy bearings
HEALTHY_BEARINGS = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006']

# Artificial damage - Outer Ring
ARTIFICIAL_OR = {
    'KA01': {'method': 'EDM',              'extent': 1, 'component': 'OR'},
    'KA03': {'method': 'electric_engraver', 'extent': 2, 'component': 'OR'},
    'KA05': {'method': 'electric_engraver', 'extent': 1, 'component': 'OR'},
    'KA06': {'method': 'electric_engraver', 'extent': 2, 'component': 'OR'},
    'KA07': {'method': 'drilling',          'extent': 1, 'component': 'OR'},
    'KA08': {'method': 'drilling',          'extent': 2, 'component': 'OR'},
    'KA09': {'method': 'drilling',          'extent': 2, 'component': 'OR'},
}

# Artificial damage - Inner Ring
ARTIFICIAL_IR = {
    'KI01': {'method': 'EDM',              'extent': 1, 'component': 'IR'},
    'KI03': {'method': 'electric_engraver', 'extent': 1, 'component': 'IR'},
    'KI05': {'method': 'electric_engraver', 'extent': 1, 'component': 'IR'},
    'KI07': {'method': 'electric_engraver', 'extent': 2, 'component': 'IR'},
    'KI08': {'method': 'electric_engraver', 'extent': 2, 'component': 'IR'},
}

# Real damage - Outer Ring
REAL_OR = {
    'KA04': {'mode': 'fatigue_pitting',    'extent': 1, 'combination': 'S', 'characteristic': 'single_point'},
    'KA15': {'mode': 'plastic_deform',     'extent': 1, 'combination': 'S', 'characteristic': 'single_point'},
    'KA16': {'mode': 'fatigue_pitting',    'extent': 2, 'combination': 'R', 'characteristic': 'single_point'},
    'KA22': {'mode': 'fatigue_pitting',    'extent': 1, 'combination': 'S', 'characteristic': 'single_point'},
    'KA30': {'mode': 'plastic_deform',     'extent': 1, 'combination': 'R', 'characteristic': 'distributed'},
}

# Real damage - Inner Ring
REAL_IR = {
    'KI04': {'mode': 'fatigue_pitting', 'extent': 1, 'combination': 'M', 'characteristic': 'single_point'},
    'KI14': {'mode': 'fatigue_pitting', 'extent': 1, 'combination': 'M', 'characteristic': 'single_point'},
    'KI16': {'mode': 'fatigue_pitting', 'extent': 3, 'combination': 'S', 'characteristic': 'single_point'},
    'KI17': {'mode': 'fatigue_pitting', 'extent': 1, 'combination': 'R', 'characteristic': 'single_point'},
    'KI18': {'mode': 'fatigue_pitting', 'extent': 2, 'combination': 'S', 'characteristic': 'single_point'},
    'KI21': {'mode': 'fatigue_pitting', 'extent': 1, 'combination': 'S', 'characteristic': 'single_point'},
}

# Real damage - Both rings (multiple damage)
REAL_BOTH = {
    'KB23': {'mode': 'fatigue_pitting',  'extent': 2, 'combination': 'M', 'primary': 'IR'},
    'KB24': {'mode': 'fatigue_pitting',  'extent': 3, 'combination': 'M', 'primary': 'IR'},
    'KB27': {'mode': 'plastic_deform',   'extent': 1, 'combination': 'M', 'primary': 'OR'},
}

# Operating conditions
OPERATING_CONDITIONS = {
    'N15_M07_F10': {'speed_rpm': 1500, 'torque_Nm': 0.7, 'force_N': 1000},
    'N09_M07_F10': {'speed_rpm': 900,  'torque_Nm': 0.7, 'force_N': 1000},
    'N15_M01_F10': {'speed_rpm': 1500, 'torque_Nm': 0.1, 'force_N': 1000},
    'N15_M07_F04': {'speed_rpm': 1500, 'torque_Nm': 0.7, 'force_N': 400},
}

# Bearing 6203 geometry (for characteristic frequency calculation)
BEARING_6203 = {
    'n_balls': 8,
    'd_inner': 24.0,    # mm
    'd_outer': 33.1,    # mm
    'd_pitch': 28.55,   # mm
    'd_ball': 6.75,     # mm
    'contact_angle': 0, # degrees
}


@dataclass
class BearingSignal:
    """Container for a single bearing measurement."""
    # Metadata
    bearing_code: str
    setting: str
    measurement_id: int
    
    # Labels
    label_3class: int          # 0=Healthy, 1=OR, 2=IR
    label_name: str            # 'Healthy', 'OR_damage', 'IR_damage'
    damage_origin: str         # 'healthy', 'artificial', 'real'
    
    # High-rate signals (64 kHz)
    phase_current_1: np.ndarray
    phase_current_2: np.ndarray
    vibration: np.ndarray
    time_64k: np.ndarray
    
    # Low-rate signals (4 kHz)
    speed: np.ndarray
    torque: np.ndarray
    force: np.ndarray
    time_4k: np.ndarray
    
    # Temperature (1 Hz)
    temperature: np.ndarray
    
    # Constants
    fs: int = 64000
    fs_low: int = 4000
    duration: float = 4.0


def get_label(bearing_code: str) -> Tuple[int, str, str]:
    """
    Determine the 3-class label from bearing code.
    
    Returns:
        (label_int, label_name, damage_origin)
    """
    if bearing_code in HEALTHY_BEARINGS:
        return 0, 'Healthy', 'healthy'
    
    if bearing_code in ARTIFICIAL_OR or bearing_code in REAL_OR:
        origin = 'artificial' if bearing_code in ARTIFICIAL_OR else 'real'
        return 1, 'OR_damage', origin
    
    if bearing_code in ARTIFICIAL_IR or bearing_code in REAL_IR:
        origin = 'artificial' if bearing_code in ARTIFICIAL_IR else 'real'
        return 2, 'IR_damage', origin
    
    if bearing_code in REAL_BOTH:
        primary = REAL_BOTH[bearing_code]['primary']
        if primary == 'IR':
            return 2, 'IR_damage', 'real'
        else:
            return 1, 'OR_damage', 'real'
    
    raise ValueError(f"Unknown bearing code: {bearing_code}")


def parse_filename(filename: str) -> Tuple[str, str, int]:
    """
    Parse filename like 'N15_M07_F10_K001_1.mat'
    
    Returns:
        (setting, bearing_code, measurement_id)
    """
    name = os.path.splitext(filename)[0]
    # Handle prefix like '1773221478402_'
    parts = name.split('_')
    
    # Find the setting pattern (NXX_MXX_FXX)
    for i, p in enumerate(parts):
        if p.startswith('N') and p[1:].isdigit():
            setting = f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
            bearing_code = parts[i+3]
            measurement_id = int(parts[i+4])
            return setting, bearing_code, measurement_id
    
    raise ValueError(f"Cannot parse filename: {filename}")


def load_mat_file(filepath: str) -> BearingSignal:
    """
    Load a single .mat file and return a BearingSignal object.
    
    Args:
        filepath: Path to .mat file
        
    Returns:
        BearingSignal with all signals and metadata
    """
    filename = os.path.basename(filepath)
    setting, bearing_code, measurement_id = parse_filename(filename)
    label_int, label_name, damage_origin = get_label(bearing_code)
    
    mat = sio.loadmat(filepath)
    
    # Find the data key (skip __header__, __version__, __globals__)
    data_key = [k for k in mat.keys() if not k.startswith('__')][0]
    d = mat[data_key][0, 0]
    
    Y = d['Y']
    X = d['X']
    
    return BearingSignal(
        bearing_code=bearing_code,
        setting=setting,
        measurement_id=measurement_id,
        label_3class=label_int,
        label_name=label_name,
        damage_origin=damage_origin,
        phase_current_1=Y['Data'][0, 1].flatten(),
        phase_current_2=Y['Data'][0, 2].flatten(),
        vibration=Y['Data'][0, 6].flatten(),
        time_64k=X['Data'][0, 1].flatten(),
        speed=Y['Data'][0, 3].flatten(),
        torque=Y['Data'][0, 5].flatten(),
        force=Y['Data'][0, 0].flatten(),
        time_4k=X['Data'][0, 0].flatten(),
        temperature=Y['Data'][0, 4].flatten(),
    )


def load_dataset(data_dir: str, setting_filter: Optional[str] = None) -> List[BearingSignal]:
    """
    Load all .mat files from a directory.
    
    Args:
        data_dir: Directory containing .mat files
        setting_filter: If provided, only load files matching this setting
                       (e.g., 'N15_M07_F10')
    
    Returns:
        List of BearingSignal objects
    """
    signals = []
    mat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])
    
    for f in mat_files:
        if setting_filter and setting_filter not in f:
            continue
        try:
            sig = load_mat_file(os.path.join(data_dir, f))
            signals.append(sig)
            print(f"  Loaded {f} -> {sig.label_name} ({sig.bearing_code})")
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    
    print(f"\nTotal loaded: {len(signals)} signals")
    return signals


def calc_characteristic_frequencies(rpm: float) -> Dict[str, float]:
    """
    Calculate bearing characteristic frequencies for type 6203.
    
    Args:
        rpm: Rotational speed in RPM
        
    Returns:
        Dictionary with characteristic frequencies in Hz
    """
    b = BEARING_6203
    fr = rpm / 60  # Shaft rotation frequency
    d = b['d_ball']
    D = b['d_pitch']
    n = b['n_balls']
    alpha = np.radians(b['contact_angle'])
    
    # Ball Pass Frequency Outer (BPFO)
    bpfo = (n / 2) * fr * (1 - (d / D) * np.cos(alpha))
    
    # Ball Pass Frequency Inner (BPFI)
    bpfi = (n / 2) * fr * (1 + (d / D) * np.cos(alpha))
    
    # Ball Spin Frequency (BSF)
    bsf = (D / (2 * d)) * fr * (1 - (d / D)**2 * np.cos(alpha)**2)
    
    # Fundamental Train Frequency (FTF)
    ftf = (fr / 2) * (1 - (d / D) * np.cos(alpha))
    
    return {
        'shaft_freq': fr,
        'BPFO': bpfo,
        'BPFI': bpfi,
        'BSF': bsf,
        'FTF': ftf,
    }


# ============================================================
# Quick test
# ============================================================
if __name__ == '__main__':
    # Test with single file
    import glob
    
    test_files = glob.glob('/mnt/user-data/uploads/*K001*.mat')
    if test_files:
        sig = load_mat_file(test_files[0])
        print(f"Bearing: {sig.bearing_code}")
        print(f"Label: {sig.label_name} (class {sig.label_3class})")
        print(f"Setting: {sig.setting}")
        print(f"Current 1 shape: {sig.phase_current_1.shape}")
        print(f"Vibration shape: {sig.vibration.shape}")
        print(f"Speed mean: {sig.speed.mean():.1f} rpm")
        print(f"Temperature mean: {sig.temperature.mean():.1f} °C")
        
        freqs = calc_characteristic_frequencies(sig.speed.mean())
        print(f"\nCharacteristic frequencies at {sig.speed.mean():.0f} rpm:")
        for name, f in freqs.items():
            print(f"  {name}: {f:.2f} Hz")
    else:
        print("No test files found. Place .mat files in the uploads directory.")
