from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.cifar10 import get_cifar10_datasets


def create_dataloaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, DataLoader]:
    """Construct PyTorch DataLoaders for train, validation, and test datasets.

    Args:
        train_ds: Training dataset split.
        val_ds: Validation dataset split.
        test_ds: Test dataset split.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.
        pin_memory: If True, pin memory for faster GPU transfer.
        generator: PyTorch random number generator for worker reproducibility.

    Returns:
        Dictionary mapping keys ('train', 'val', 'test') to their respective DataLoaders.
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def get_cifar10_dataloaders(
    data_dir: str = "data/raw",
    batch_size: int = 128,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
    augment_training: bool = True,
    download: bool = True,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    """High-level utility to get CIFAR-10 datasets and construct configured DataLoaders.

    Args:
        data_dir: Path to raw data directory.
        batch_size: Batch size for all data loaders.
        num_workers: Data loader worker count.
        val_split: Validation split fraction.
        seed: Random seed for split reproducibility.
        augment_training: Apply training augmentation if True.
        download: Download dataset if missing.
        pin_memory: Enable memory pinning for CUDA.

    Returns:
        Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    train_ds, val_ds, test_ds, _ = get_cifar10_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        augment_training=augment_training,
        download=download,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    return create_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
    )
