from typing import Tuple, List, Dict
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from src.data.transforms import get_cifar10_transforms
from src.utils.seed import set_seed

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def get_cifar10_datasets(
    data_dir: str = "data/raw",
    val_split: float = 0.1,
    seed: int = 42,
    augment_training: bool = True,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset, Dict[str, List[int]]]:
    """Load CIFAR-10 dataset and perform a reproducible, stratified train/validation split.

    The original 50,000 training images are split into 45,000 training and 5,000 validation samples.
    The original 10,000 test images remain completely untouched.

    Args:
        data_dir: Directory where raw CIFAR-10 data is stored/downloaded.
        val_split: Fraction of training dataset to set aside for validation (default 0.1 -> 5,000 / 50,000).
        seed: Random seed for reproducible stratified splitting.
        augment_training: Whether to apply standard training data augmentation.
        download: Whether to download CIFAR-10 if not present locally.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, split_indices_dict).
    """
    set_seed(seed)

    train_transform, eval_transform = get_cifar10_transforms(
        augment_training=augment_training
    )

    # Base dataset for training (with training transform)
    raw_train_ds = CIFAR10(
        root=data_dir, train=True, download=download, transform=train_transform
    )

    # Base dataset for validation (with eval transform, deterministic)
    raw_val_ds = CIFAR10(
        root=data_dir, train=True, download=download, transform=eval_transform
    )

    # Test dataset (with eval transform, untouched original test set)
    test_ds = CIFAR10(
        root=data_dir, train=False, download=download, transform=eval_transform
    )

    # Stratified split on 50,000 train labels
    targets = np.array(raw_train_ds.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)

    train_indices, val_indices = next(sss.split(np.zeros(len(targets)), targets))

    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    # Verify indices are strictly disjoint
    assert set(train_indices).isdisjoint(
        set(val_indices)
    ), "Training and validation index sets must be strictly disjoint!"

    # Create subset datasets
    train_ds = Subset(raw_train_ds, train_indices)
    val_ds = Subset(raw_val_ds, val_indices)

    split_indices = {"train": train_indices, "val": val_indices}

    return train_ds, val_ds, test_ds, split_indices
