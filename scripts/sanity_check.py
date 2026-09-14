import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data import (
    CIFAR10_CLASSES,
    get_cifar10_datasets,
    get_cifar10_dataloaders,
    generate_dataset_metadata,
    save_dataset_metadata,
)
from src.utils.seed import set_seed


def run_sanity_checks():
    print("=" * 60)
    print("VISIONTRUST AI — CIFAR-10 DATASET PIPELINE SANITY CHECKS")
    print("=" * 60)

    seed = 42
    set_seed(seed)
    data_dir = "data/raw"

    print("\n[1/10] Loading CIFAR-10 datasets and generating stratified split...")
    train_ds, val_ds, test_ds, split_indices = get_cifar10_datasets(
        data_dir=data_dir, val_split=0.1, seed=seed, download=True
    )

    # Check 1: Training dataset size = 45,000
    assert (
        len(train_ds) == 45000
    ), f"Expected 45,000 training samples, got {len(train_ds)}"
    print(f"✓ Check 1 Passed: Training set size = {len(train_ds):,}")

    # Check 2: Validation dataset size = 5,000
    assert (
        len(val_ds) == 5000
    ), f"Expected 5,000 validation samples, got {len(val_ds)}"
    print(f"✓ Check 2 Passed: Validation set size = {len(val_ds):,}")

    # Check 3: Test dataset size = 10,000
    assert len(test_ds) == 10000, f"Expected 10,000 test samples, got {len(test_ds)}"
    print(f"✓ Check 3 Passed: Test set size = {len(test_ds):,}")

    # Check 4: Exactly 10 classes
    assert (
        len(CIFAR10_CLASSES) == 10
    ), f"Expected 10 classes, got {len(CIFAR10_CLASSES)}"
    print(
        f"✓ Check 4 Passed: Exactly {len(CIFAR10_CLASSES)} classes detected ({', '.join(CIFAR10_CLASSES)})"
    )

    # Check 5: Compute & report class distributions
    metadata = generate_dataset_metadata(
        train_ds, val_ds, test_ds, seed=seed, val_split=0.1
    )
    print("\n✓ Check 5 Passed: Class distributions computed:")
    print(f"  Class Names: {metadata['class_names']}")
    print("  Train distribution per class:")
    for cls, count in metadata["class_distributions"]["train"].items():
        print(f"    - {cls:12s}: {count:5d} ({count/len(train_ds)*100:.2f}%)")
    print("  Val distribution per class:")
    for cls, count in metadata["class_distributions"]["val"].items():
        print(f"    - {cls:12s}: {count:5d} ({count/len(val_ds)*100:.2f}%)")

    # Check 6 & 7 & 8: DataLoaders batch execution, tensor shapes, label ranges
    batch_size = 64
    dataloaders = get_cifar10_dataloaders(
        data_dir=data_dir, batch_size=batch_size, num_workers=0, seed=seed
    )

    for split_name, loader in dataloaders.items():
        images, labels = next(iter(loader))

        # Check 6: Batch loaded
        assert images is not None and labels is not None, f"Failed to load batch from {split_name} loader!"
        assert images.size(0) == batch_size, f"Expected batch size {batch_size}, got {images.size(0)}"

        # Check 7: Image tensor shape (B, 3, 32, 32)
        expected_shape = (batch_size, 3, 32, 32)
        assert (
            images.shape == expected_shape
        ), f"Expected image shape {expected_shape}, got {images.shape}"

        # Check 8: Label range [0, 9]
        min_label, max_label = labels.min().item(), labels.max().item()
        assert (
            0 <= min_label and max_label < 10
        ), f"{split_name} label out of bounds: min={min_label}, max={max_label}"

    print(
        f"\n✓ Check 6 Passed: Batch of size {batch_size} loaded successfully from all DataLoaders."
    )
    print(
        f"✓ Check 7 Passed: Image tensor shape verified as (B, 3, 32, 32) for CIFAR-10."
    )
    print(f"✓ Check 8 Passed: All batch labels fall strictly in range [0, 9].")

    # Check 9: Original test set untouched
    assert hasattr(test_ds, "targets"), "Test set is missing raw targets attribute!"
    assert (
        len(test_ds.targets) == 10000
    ), "Test set targets count does not match original CIFAR-10 test set!"
    print(
        "✓ Check 9 Passed: Original test set is untouched and contains full 10,000 test images."
    )

    # Check 10: Disjoint sample indices between train and val
    train_idx = set(split_indices["train"])
    val_idx = set(split_indices["val"])
    intersection = train_idx.intersection(val_idx)
    assert (
        len(intersection) == 0
    ), f"Data leakage detected! {len(intersection)} duplicate indices exist between train and val subsets."
    print(
        f"✓ Check 10 Passed: Zero overlap between training and validation sample indices (strictly disjoint)."
    )

    # Save observed metadata
    meta_path = "outputs/metrics/cifar10_metadata.json"
    save_dataset_metadata(metadata, filepath=meta_path)
    print(f"\nObserved dataset metadata saved to: {meta_path}")

    print("\n" + "=" * 60)
    print("ALL 10 SANITY CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()
