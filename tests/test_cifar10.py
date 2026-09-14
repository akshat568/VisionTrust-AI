import pytest
import torch
from torch.utils.data import DataLoader
from src.data import (
    CIFAR10_CLASSES,
    get_cifar10_datasets,
    get_cifar10_dataloaders,
)


@pytest.fixture(scope="module")
def dataset_splits():
    train_ds, val_ds, test_ds, split_indices = get_cifar10_datasets(
        data_dir="data/raw", val_split=0.1, seed=42, download=True
    )
    return train_ds, val_ds, test_ds, split_indices


def test_dataset_sizes(dataset_splits):
    train_ds, val_ds, test_ds, _ = dataset_splits
    assert len(train_ds) == 45000, f"Expected 45000 train samples, got {len(train_ds)}"
    assert len(val_ds) == 5000, f"Expected 5000 val samples, got {len(val_ds)}"
    assert len(test_ds) == 10000, f"Expected 10000 test samples, got {len(test_ds)}"


def test_class_count():
    assert len(CIFAR10_CLASSES) == 10
    assert "airplane" in CIFAR10_CLASSES
    assert "truck" in CIFAR10_CLASSES


def test_reproducible_split():
    _, _, _, split_indices1 = get_cifar10_datasets(
        data_dir="data/raw", val_split=0.1, seed=42, download=False
    )
    _, _, _, split_indices2 = get_cifar10_datasets(
        data_dir="data/raw", val_split=0.1, seed=42, download=False
    )
    assert (
        split_indices1["train"] == split_indices2["train"]
    ), "Reproducible train split failed for seed=42"
    assert (
        split_indices1["val"] == split_indices2["val"]
    ), "Reproducible val split failed for seed=42"


def test_index_disjointness(dataset_splits):
    _, _, _, split_indices = dataset_splits
    train_indices = set(split_indices["train"])
    val_indices = set(split_indices["val"])
    assert train_indices.isdisjoint(
        val_indices
    ), "Training and validation index sets overlap!"


def test_tensor_shape_and_label_validity(dataset_splits):
    train_ds, val_ds, test_ds, _ = dataset_splits
    batch_size = 16
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(loader))
    assert images.shape == (batch_size, 3, 32, 32)
    assert torch.is_floating_point(images)
    assert labels.min() >= 0 and labels.max() < 10
