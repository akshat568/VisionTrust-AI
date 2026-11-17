"""Model Download Utility for VisionTrust AI.

Utility script to verify or download the pre-trained ResNet-18 baseline checkpoint 
('outputs/models/baseline_resnet18_best.pth', ~128MB) for fresh repository clones.
"""

import argparse
import os
from pathlib import Path
import sys
import urllib.request

# Default GitHub Release asset URL for pre-trained ResNet-18 model checkpoint (~128 MB)
DEFAULT_MODEL_URL = os.environ.get(
    "VISIONTRUST_MODEL_URL",
    "https://github.com/akshat568/VisionTrust-AI/releases/download/v1.0.0/baseline_resnet18_best.pth",
)
DEFAULT_ONNX_MODEL_URL = os.environ.get(
    "VISIONTRUST_ONNX_MODEL_URL",
    "",
)

DEFAULT_PYTORCH_DESTINATION = Path("outputs/models/baseline_resnet18_best.pth")
DEFAULT_ONNX_DESTINATION = Path("outputs/models/baseline_resnet18.onnx")
DEFAULT_DESTINATION = DEFAULT_PYTORCH_DESTINATION


def download_file(url: str, dest_path: Path):
    """Download file with progress report."""
    dest_path = Path(dest_path).resolve()
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
        req = urllib.request.Request(url, headers={"User-Agent": "VisionTrust-AI/1.0"})
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            block_size = 8192
            block_num = 0
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                block_num += 1
                progress_callback(block_num, block_size, total_size)
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nFailed to download model weights: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download model weights from '{url}': {e}") from e


def main():
    parser = argparse.ArgumentParser(
        description="Verify or download VisionTrust AI ResNet-18 model weights (PyTorch or ONNX)."
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Target ONNX runtime model ('outputs/models/baseline_resnet18.onnx') instead of PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination file path for model weights checkpoint/artifact.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to download model checkpoint from (or set VISIONTRUST_MODEL_URL / VISIONTRUST_ONNX_MODEL_URL).",
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

    # Determine destination path and download URL based on target format
    if args.onnx:
        dest_path = Path(args.output or DEFAULT_ONNX_DESTINATION).resolve()
        url = args.url or DEFAULT_ONNX_MODEL_URL
        model_name = "ONNX ResNet-18 model"
    else:
        dest_path = Path(args.output or DEFAULT_PYTORCH_DESTINATION).resolve()
        url = args.url or DEFAULT_MODEL_URL
        model_name = "PyTorch ResNet-18 checkpoint"

    if dest_path.exists() and not args.force:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"{model_name} already present at '{dest_path}' ({size_mb:.2f} MB).")
        print("Use --force to re-download if necessary.")
        return

    if args.check_only:
        print(f"{model_name} NOT found at '{dest_path}'.")
        sys.exit(1)

    if not url or "<INSERT_HOSTED_RELEASE_URL_HERE>" in url:
        print("\n" + "=" * 70)
        print("VISIONTRUST AI MODEL DOWNLOAD GUIDE")
        print("=" * 70)
        print(f"The model weights file ('{dest_path.name}') is not yet available at: {dest_path}\n")
        if args.onnx:
            print("To obtain the ONNX model:")
            print("  1. Run: python scripts/export_onnx.py (generates outputs/models/baseline_resnet18.onnx)")
            print("  OR")
            print("  2. Pass a direct URL: python scripts/download_model.py --onnx --url <DIRECT_DOWNLOAD_URL>")
            print("  OR set environment variable: VISIONTRUST_ONNX_MODEL_URL=<DIRECT_DOWNLOAD_URL>")
        else:
            print("To obtain the PyTorch model checkpoint:")
            print("  1. Download 'baseline_resnet18_best.pth' from GitHub Releases or hosted storage.")
            print(f"  2. Place the file at: {dest_path}")
            print("  OR")
            print("  3. Pass a direct URL: python scripts/download_model.py --url <DIRECT_DOWNLOAD_URL>")
            print("  OR set environment variable: VISIONTRUST_MODEL_URL=<DIRECT_DOWNLOAD_URL>")
        print("=" * 70 + "\n")
        sys.exit(0)

    try:
        download_file(url, dest_path)
    except Exception:
        sys.exit(1)



if __name__ == "__main__":
    main()
