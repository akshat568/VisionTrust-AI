import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from torch.utils.data import Dataset, Subset

from src.data.cifar10 import CIFAR10_CLASSES
from src.data.transforms import CIFAR10_MEAN, CIFAR10_STD


def compute_class_distribution(
    dataset: Dataset, class_names: List[str] = CIFAR10_CLASSES
) -> Dict[str, int]:
    """Compute the actual observed label count per class in a dataset or subset.

    Args:
        dataset: The Dataset or Subset instance.
        class_names: List of class name strings.

    Returns:
        Dictionary mapping class name to observed sample count.
    """
    if isinstance(dataset, Subset):
        base_ds = dataset.dataset
        indices = dataset.indices
        targets = [base_ds.targets[i] for i in indices]
    else:
        targets = dataset.targets

    counts = np.bincount(targets, minlength=len(class_names))
    return {
        class_name: int(count) for class_name, count in zip(class_names, counts)
    }


def generate_dataset_metadata(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    seed: int = 42,
    val_split: float = 0.1,
) -> Dict[str, Any]:
    """Generate comprehensive, observed metadata for the CIFAR-10 dataset split.

    Args:
        train_ds: Training dataset split.
        val_ds: Validation dataset split.
        test_ds: Test dataset split.
        seed: Random seed used for the split.
        val_split: Fraction reserved for validation.

    Returns:
        Metadata dictionary containing exact observed sizes and class counts.
    """
    train_counts = compute_class_distribution(train_ds)
    val_counts = compute_class_distribution(val_ds)
    test_counts = compute_class_distribution(test_ds)

    metadata = {
        "dataset_name": "CIFAR-10",
        "random_seed": seed,
        "validation_split_ratio": val_split,
        "num_classes": len(CIFAR10_CLASSES),
        "class_names": list(CIFAR10_CLASSES),
        "split_sizes": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
            "total": len(train_ds) + len(val_ds) + len(test_ds),
        },
        "normalization": {
            "mean": list(CIFAR10_MEAN),
            "std": list(CIFAR10_STD),
        },
        "class_distributions": {
            "train": train_counts,
            "validation": val_counts,
            "test": test_counts,
        },
    }

    return metadata


def save_dataset_metadata(
    metadata: Dict[str, Any],
    filepath: str = "outputs/metrics/cifar10_metadata.json",
) -> None:
    """Save dataset metadata to a JSON file.

    Args:
        metadata: Metadata dictionary.
        filepath: Path to output JSON file.
    """
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
