#!/usr/bin/env python3
"""
Upload local MLflow model artifacts to S3.

Run this on your laptop after training to make the model available
for the GitHub Actions CI/CD pipeline.

Usage:
    pip install boto3
    python scripts/upload_model_to_s3.py --bucket bearing-fault-mlruns-yourname

What it uploads:
    mlruns/  →  s3://<bucket>/mlruns/

The script skips files that are already up-to-date (same size),
so re-running after a fresh training only uploads changed files.
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR   = Path(__file__).parent.parent
_MLRUNS_DIR = _BASE_DIR / "mlruns"


def _try_import_boto3():
    try:
        import boto3
        from botocore.exceptions import ClientError
        return boto3, ClientError
    except ImportError:
        print("ERROR: boto3 is not installed.", file=sys.stderr)
        print("  Run:  pip install boto3", file=sys.stderr)
        sys.exit(1)


def upload_dir_to_s3(local_dir: Path, bucket: str, s3_prefix: str) -> tuple[int, int]:
    """Upload every file in local_dir to S3 under s3_prefix.

    Skips files where the remote object already has the same byte size
    (cheap fast-path — avoids re-uploading unchanged model versions).

    Args:
        local_dir:  Local directory to upload recursively.
        bucket:     S3 bucket name.
        s3_prefix:  Key prefix inside the bucket (no trailing slash).

    Returns:
        (uploaded, skipped) counts.
    """
    boto3, ClientError = _try_import_boto3()
    s3 = boto3.client("s3")

    uploaded = skipped = 0

    all_files = sorted(p for p in local_dir.rglob("*") if p.is_file())
    print(f"  {len(all_files)} local files found")

    for local_path in all_files:
        relative  = local_path.relative_to(local_dir).as_posix()
        s3_key    = f"{s3_prefix}/{relative}"
        local_size = local_path.stat().st_size

        # Check if already uploaded with same size
        try:
            head = s3.head_object(Bucket=bucket, Key=s3_key)
            if head["ContentLength"] == local_size:
                skipped += 1
                continue
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "404":
                raise  # unexpected error — propagate

        print(f"  uploading  {s3_key}  ({local_size:,} bytes)")
        s3.upload_file(str(local_path), bucket, s3_key)
        uploaded += 1

    return uploaded, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload mlruns/ model artifacts to S3 for CI/CD builds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/upload_model_to_s3.py --bucket bearing-fault-mlruns-alice
  python scripts/upload_model_to_s3.py --bucket bearing-fault-mlruns-alice --prefix mlruns
        """,
    )
    parser.add_argument(
        "--bucket", required=True,
        help="S3 bucket name (e.g. bearing-fault-mlruns-yourname)",
    )
    parser.add_argument(
        "--prefix", default="mlruns",
        help="S3 key prefix — defaults to 'mlruns'",
    )
    args = parser.parse_args()

    if not _MLRUNS_DIR.exists():
        print(
            f"ERROR: {_MLRUNS_DIR} not found.\n"
            "  Train the model first by running the notebook end-to-end.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nUploading  {_MLRUNS_DIR}")
    print(f"     →  s3://{args.bucket}/{args.prefix}/\n")

    uploaded, skipped = upload_dir_to_s3(_MLRUNS_DIR, args.bucket, args.prefix)

    print(f"\nDone.  {uploaded} file(s) uploaded,  {skipped} already up-to-date.")
    if uploaded > 0:
        print(
            "\nNext step: push to GitHub (main branch) to trigger a deploy.\n"
            "  git add -A && git commit -m 'chore: retrain model' && git push"
        )


if __name__ == "__main__":
    main()
