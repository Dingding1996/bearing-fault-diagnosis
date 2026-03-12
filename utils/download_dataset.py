"""
Paderborn Bearing Dataset - Download & Extract
===============================================
Can be used as a library or run directly as a script.

As a library:
    from 00_download_dataset import ensure_data, MINIMAL_SET
    mat_dir = ensure_data(MINIMAL_SET)

As a script:
    python 00_download_dataset.py                   # download all bearings
    python 00_download_dataset.py --minimal         # download subset only
    python 00_download_dataset.py --bearings K001 KA01 KI03
    python 00_download_dataset.py --output_dir /path/to/save

Requirements:
    pip install requests tqdm
    7-Zip must be installed (Windows: https://www.7-zip.org)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Run: pip install requests tqdm")
    sys.exit(1)


# Dataset hosted on Zenodo (DOI: 10.5281/zenodo.15845309)
BASE_URL = "https://zenodo.org/records/15845309/files"

# Default data directory relative to this file
DEFAULT_DATA_DIR = Path(__file__).parent / 'paderborn_data'

ALL_BEARINGS = [
    # Healthy
    'K001', 'K002', 'K003', 'K004', 'K005', 'K006',
    # Artificial damage - Outer Ring
    'KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09',
    # Real damage - Outer Ring
    'KA04', 'KA15', 'KA16', 'KA22', 'KA30',
    # Both rings damaged
    'KB23', 'KB24', 'KB27',
    # Artificial damage - Inner Ring
    'KI01', 'KI03', 'KI05', 'KI07', 'KI08',
    # Real damage - Inner Ring
    'KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21',
]

# Minimal subset — real damage only, balanced classes (~2.4 GB, 15 bearings)
MINIMAL_SET = [
    'K001', 'K002', 'K003', 'K004', 'K005',  # Healthy
    'KA04', 'KA15', 'KA16', 'KA22', 'KA30',  # Outer ring real damage
    'KI04', 'KI14', 'KI16', 'KI18', 'KI21',  # Inner ring real damage
]

# Full set — all bearings including artificial damage (~5 GB, 32 bearings)
FULL_SET = ALL_BEARINGS


def download_file(url: str, dest: str) -> bool:
    """Download a single file with progress bar and resume support.

    Args:
        url: Direct download URL.
        dest: Local destination file path.

    Returns:
        True if download succeeded, False otherwise.
    """
    existing_size = os.path.getsize(dest) if os.path.exists(dest) else 0

    headers = {}
    if existing_size > 0:
        headers['Range'] = f'bytes={existing_size}-'

    response = requests.get(url, headers=headers, stream=True)

    if response.status_code == 200:
        existing_size = 0  # Server sent full file; start fresh
        mode = 'wb'
    elif response.status_code == 206:
        mode = 'ab'  # Partial content — append to existing file
    elif response.status_code == 416:
        print("  Already complete, skipping.")
        return True
    else:
        print(f"  Download failed: HTTP {response.status_code}")
        return False

    total = int(response.headers.get('content-length', 0)) + existing_size

    with open(dest, mode) as f:
        with tqdm(total=total, initial=existing_size, unit='B',
                  unit_scale=True, desc=os.path.basename(dest)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    return True


def extract_rar(rar_path: str, output_dir: str) -> bool:
    """Extract a .rar archive, trying multiple extractors in order.

    Tries 7-Zip first (recommended on Windows), then unrar, then
    the Python rarfile module as a last resort.

    Args:
        rar_path: Path to the .rar file.
        output_dir: Directory to extract into.

    Returns:
        True if extraction succeeded, False otherwise.
    """
    # 7-Zip: preferred on Windows — no separate unrar binary needed
    seven_zip_candidates = [
        '7z',
        r'C:\Program Files\7-Zip\7z.exe',
        r'C:\Program Files (x86)\7-Zip\7z.exe',
    ]
    for cmd in seven_zip_candidates:
        try:
            result = subprocess.run(
                [cmd, 'x', '-y', f'-o{output_dir}', rar_path],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print("  Extracted successfully with 7-Zip.")
                return True
            else:
                print(f"  7-Zip error: {result.stderr.strip() or result.stdout.strip()}")
        except FileNotFoundError:
            continue

    # unrar command-line tool
    try:
        result = subprocess.run(
            ['unrar', 'x', '-o+', rar_path, output_dir],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass

    # Python rarfile module (still requires unrar binary internally)
    try:
        import rarfile
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(output_dir)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  rarfile extraction failed: {e}")

    print(f"  Could not extract {rar_path}.")
    print("  Install 7-Zip from https://www.7-zip.org (Windows)")
    print("  or run: brew install unrar / apt install unrar")
    return False


def ensure_data(
    bearings: list[str] = MINIMAL_SET,
    output_dir: Path = DEFAULT_DATA_DIR,
    keep_rar: bool = False,
) -> Path:
    """Check if bearing data exists; download and extract any that are missing.

    This is the main library entry point. Call this at the top of any script
    that needs the dataset — it is a no-op for bearings already on disk.

    Args:
        bearings: List of bearing IDs to ensure are present.
        output_dir: Root directory for downloaded data.
        keep_rar: If True, keep .rar files after extraction.

    Returns:
        Path to the mat/ directory containing the extracted .mat files.
    """
    output_dir = Path(output_dir)
    rar_dir = output_dir / 'rar'
    mat_dir = output_dir / 'mat'
    rar_dir.mkdir(parents=True, exist_ok=True)
    mat_dir.mkdir(parents=True, exist_ok=True)

    missing = [b for b in bearings if not any((mat_dir / b).rglob('*.mat'))]

    if not missing:
        print(f"Dataset ready: all {len(bearings)} bearings found in {mat_dir}")
        return mat_dir

    print(f"\n{'='*50}")
    print("Paderborn Bearing Dataset Downloader")
    print(f"{'='*50}")
    print(f"Already on disk : {len(bearings) - len(missing)}/{len(bearings)} bearings")
    print(f"To download     : {len(missing)} bearings (~{len(missing) * 160} MB)")
    print(f"Output          : {output_dir.absolute()}")
    print(f"{'='*50}\n")

    failed = []
    for i, bearing in enumerate(missing, 1):
        url = f"{BASE_URL}/{bearing}.rar?download=1"
        dest = rar_dir / f"{bearing}.rar"

        print(f"[{i}/{len(missing)}] {bearing}.rar")

        success = download_file(url, str(dest))
        if not success:
            failed.append(bearing)
            continue

        print("  Extracting...")
        (mat_dir / bearing).mkdir(exist_ok=True)
        success = extract_rar(str(dest), str(mat_dir))
        if success and not keep_rar:
            dest.unlink()
            print("  Extracted and .rar removed.")
        elif success:
            print("  Extracted.")
        else:
            failed.append(bearing)

    if failed:
        print(f"\nWarning: failed to download/extract: {failed}")
    else:
        print(f"\nAll {len(missing)} bearings downloaded successfully.")

    return mat_dir


def main():
    parser = argparse.ArgumentParser(description='Download the Paderborn Bearing Dataset')
    parser.add_argument('--output_dir', default=str(DEFAULT_DATA_DIR),
                        help=f'Directory to save data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--bearings', nargs='+', default=None,
                        help='Specific bearing IDs to download (default: all)')
    parser.add_argument('--minimal', action='store_true',
                        help='Download minimal subset only (15 bearings, ~2.4 GB)')
    parser.add_argument('--keep_rar', action='store_true',
                        help='Keep .rar files after extraction')
    args = parser.parse_args()

    if args.bearings:
        bearings = args.bearings
    elif args.minimal:
        bearings = MINIMAL_SET
    else:
        bearings = ALL_BEARINGS

    mat_dir = ensure_data(bearings, args.output_dir, keep_rar=args.keep_rar)

    # Final summary
    mat_files = list(mat_dir.rglob('*.mat'))
    print(f"\nTotal .mat files : {len(mat_files)}")
    print(f"Data path        : {mat_dir.absolute()}")


if __name__ == '__main__':
    main()
