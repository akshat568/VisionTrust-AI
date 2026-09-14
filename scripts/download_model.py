"""Model Download Utility for VisionTrust AI.

Utility script to verify or download the pre-trained ResNet-18 baseline checkpoint 
('outputs/models/baseline_resnet18_best.pth', ~128MB) for fresh repository clones.
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request

# Placeholder URL - Replace with actual GitHub Release asset URL or S3/GCS bucket link upon release
DEFAULT_MODEL_URL = os.environ.get(
    "VISIONTRUST_MODEL_URL",
    "<INSERT_HOSTED_RELEASE_URL_HERE>",
)

DEFAULT_DESTINATION = Path("outputs/models/baseline_resnet18_best.pth")


def download_file(url: str, dest_path: Path):
    """Download file with progress report."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pre-trained ResNet-18 model weights from:\n  {url}")
    print(f"Destination: {dest_path}")

    def progress_callback(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, (downloaded / total_size) * 100.0)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(
                f"\rProgress: [{percent:6.2f}%] {mb_downloaded:6.2f} MB / {mb_total:6.2f} MB"
            )
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {mb_downloaded:6.2f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nFailed to download model weights: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Verify or download VisionTrust AI ResNet-18 pre-trained model weights."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_DESTINATION),
        help="Destination file path for baseline weights checkpoint.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_MODEL_URL,
        help="URL to download model checkpoint from (or set VISIONTRUST_MODEL_URL env var).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if destination file already exists.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check if model weights exist on disk without downloading.",
    )
    args = parser.parse_args()

    dest_path = Path(args.output).resolve()

    if dest_path.exists() and not args.force:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"Model checkpoint already present at '{dest_path}' ({size_mb:.2f} MB).")
        print("Use --force to re-download if necessary.")
        return

    if args.check_only:
        print(f"Model checkpoint NOT found at '{dest_path}'.")
        sys.exit(1)

    url = args.url
    if not url or "<INSERT_HOSTED_RELEASE_URL_HERE>" in url:
        print("\n" + "=" * 70)
        print("VISIONTRUST AI MODEL DOWNLOAD GUIDE")
        print("=" * 70)
        print(f"The baseline model weights file ('{dest_path.name}') is ~128 MB and excluded")
        print("from git history to stay within repository size limits.\n")
        print("To obtain the model checkpoint:")
        print("  1. Download 'baseline_resnet18_best.pth' from GitHub Releases or hosted storage.")
        print(f"  2. Place the file at: {dest_path}")
        print("  OR")
        print("  3. Pass a direct URL: python scripts/download_model.py --url <DIRECT_DOWNLOAD_URL>")
        print("  OR set environment variable: VISIONTRUST_MODEL_URL=<DIRECT_DOWNLOAD_URL>")
        print("=" * 70 + "\n")
        sys.exit(0)

    download_file(url, dest_path)


if __name__ == "__main__":
    main()
