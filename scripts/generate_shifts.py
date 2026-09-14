from pathlib import Path
import sys
from tqdm import tqdm
import torch
from torchvision.datasets import CIFAR10

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import SEVERITY_PARAMS, apply_corruption_to_tensor_batch
from src.utils.seed import set_seed


def generate_all_shifted_test_sets(
    data_dir: str = "data/raw",
    output_dir: str = "data/shifted",
    seed: int = 42,
):
    """Generate and save all 25 controlled distribution-shifted versions of the 10,000 CIFAR-10 test images.

    Args:
        data_dir: Directory where raw CIFAR-10 data is stored.
        output_dir: Directory to save generated shifted dataset `.pt` files.
        seed: Random seed for reproducibility.
    """
    set_seed(seed)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VISIONTRUST AI — GENERATING CONTROLLED DISTRIBUTION SHIFTS (PHASE 4)")
    print("=" * 65)

    # Load raw PIL CIFAR-10 test dataset without transform
    raw_test_ds = CIFAR10(root=data_dir, train=False, download=True, transform=None)
    num_samples = len(raw_test_ds)
    assert num_samples == 10000, f"Expected 10,000 test samples, got {num_samples}"

    print(f"\nLoaded {num_samples:,} raw CIFAR-10 test set images.")
    print("Generating 5 Corruptions x 5 Severities = 25 Shifted Datasets...\n")

    # Convert all raw images to [10000, 3, 32, 32] float tensor in [0, 1]
    raw_tensors = torch.from_numpy(raw_test_ds.data).permute(0, 3, 1, 2).float() / 255.0
    labels_tensor = torch.tensor(raw_test_ds.targets, dtype=torch.long)

    corruptions = list(SEVERITY_PARAMS.keys())
    severities = [1, 2, 3, 4, 5]

    total_datasets = len(corruptions) * len(severities)
    count = 0

    for corruption in corruptions:
        for severity in severities:
            count += 1
            save_dir = out_root / corruption
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"severity_{severity}.pt"

            print(
                f"[{count:02d}/{total_datasets:02d}] Generating: {corruption:12s} | Severity {severity} -> {save_path}"
            )

            images_tensor = apply_corruption_to_tensor_batch(
                raw_tensors, corruption_type=corruption, severity=severity, seed=seed
            )

            assert images_tensor.shape == (
                10000,
                3,
                32,
                32,
            ), f"Invalid shape: {images_tensor.shape}"
            assert len(labels_tensor) == 10000, f"Invalid label count: {len(labels_tensor)}"

            payload = {
                "corruption": corruption,
                "severity": severity,
                "images": images_tensor,
                "labels": labels_tensor,
            }
            torch.save(payload, save_path)

    print("\n" + "=" * 65)
    print("ALL 25 SHIFTED TEST DATASETS GENERATED SUCCESSFULLY!")
    print(f"Saved to directory: {out_root}")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_shifted_test_sets()
