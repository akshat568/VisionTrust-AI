from pathlib import Path
from typing import Tuple, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.data.transforms import CIFAR10_MEAN, CIFAR10_STD
from src.utils.seed import set_seed

# Explicit, documented severity parameter maps for 32x32 CIFAR-10 images
SEVERITY_PARAMS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "blur": {
        1: {"kernel_size": 3, "sigma": 0.5},
        2: {"kernel_size": 3, "sigma": 1.0},
        3: {"kernel_size": 5, "sigma": 1.5},
        4: {"kernel_size": 5, "sigma": 2.0},
        5: {"kernel_size": 5, "sigma": 2.5},
    },
    "noise": {
        1: {"std": 0.05},
        2: {"std": 0.10},
        3: {"std": 0.15},
        4: {"std": 0.20},
        5: {"std": 0.25},
    },
    "brightness": {
        1: {"factor": 0.85},
        2: {"factor": 0.70},
        3: {"factor": 0.55},
        4: {"factor": 0.40},
        5: {"factor": 0.25},
    },
    "contrast": {
        1: {"factor": 0.85},
        2: {"factor": 0.70},
        3: {"factor": 0.55},
        4: {"factor": 0.40},
        5: {"factor": 0.25},
    },
    "rotation": {
        1: {"angle": 5.0},
        2: {"angle": 10.0},
        3: {"angle": 15.0},
        4: {"angle": 25.0},
        5: {"angle": 35.0},
    },
}


def apply_gaussian_blur(img: Image.Image, severity: int) -> Image.Image:
    """Apply Gaussian Blur to PIL Image for a given severity level (1-5)."""
    params = SEVERITY_PARAMS["blur"][severity]
    return TF.gaussian_blur(
        img, kernel_size=params["kernel_size"], sigma=params["sigma"]
    )


def apply_gaussian_noise(
    tensor_img: torch.Tensor, severity: int, seed: int = 42
) -> torch.Tensor:
    """Add Gaussian noise to a unnormalized [0, 1] image Tensor.

    Args:
        tensor_img: Float tensor of shape [3, H, W] in range [0, 1].
        severity: Severity level (1-5).
        seed: Random seed for reproducible noise generation.

    Returns:
        Noisy image tensor clipped to [0, 1].
    """
    params = SEVERITY_PARAMS["noise"][severity]
    std = params["std"]

    # Generate reproducible noise
    g = torch.Generator()
    g.manual_seed(seed + severity * 1000)
    noise = torch.randn(tensor_img.shape, generator=g) * std

    noisy_img = torch.clamp(tensor_img + noise, 0.0, 1.0)
    return noisy_img


def apply_brightness(img: Image.Image, severity: int) -> Image.Image:
    """Adjust brightness of PIL Image for a given severity level (1-5)."""
    params = SEVERITY_PARAMS["brightness"][severity]
    return TF.adjust_brightness(img, brightness_factor=params["factor"])


def apply_contrast(img: Image.Image, severity: int) -> Image.Image:
    """Adjust contrast of PIL Image for a given severity level (1-5)."""
    params = SEVERITY_PARAMS["contrast"][severity]
    return TF.adjust_contrast(img, contrast_factor=params["factor"])


def apply_rotation(img: Image.Image, severity: int) -> Image.Image:
    """Rotate PIL Image for a given severity level (1-5)."""
    params = SEVERITY_PARAMS["rotation"][severity]
    return TF.rotate(img, angle=params["angle"])


def apply_corruption_to_pil(
    img: Image.Image, corruption_type: str, severity: int, seed: int = 42
) -> torch.Tensor:
    """Apply specified corruption to a raw PIL image and return normalized PyTorch tensor [3, 32, 32].

    Args:
        img: Input PIL RGB image (32x32).
        corruption_type: One of ('blur', 'noise', 'brightness', 'contrast', 'rotation').
        severity: Severity level (1-5).
        seed: Seed for reproducible noise generation.

    Returns:
        Normalized tensor [3, 32, 32] ready for ResNet-18 model inference.
    """
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)

    if corruption_type == "blur":
        corrupted_pil = apply_gaussian_blur(img, severity)
        tensor_img = to_tensor(corrupted_pil)
    elif corruption_type == "brightness":
        corrupted_pil = apply_brightness(img, severity)
        tensor_img = to_tensor(corrupted_pil)
    elif corruption_type == "contrast":
        corrupted_pil = apply_contrast(img, severity)
        tensor_img = to_tensor(corrupted_pil)
    elif corruption_type == "rotation":
        corrupted_pil = apply_rotation(img, severity)
        tensor_img = to_tensor(corrupted_pil)
    elif corruption_type == "noise":
        raw_tensor = to_tensor(img)
        tensor_img = apply_gaussian_noise(raw_tensor, severity, seed=seed)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    return normalize(tensor_img)


def apply_corruption_to_tensor_batch(
    raw_tensors: torch.Tensor, corruption_type: str, severity: int, seed: int = 42
) -> torch.Tensor:
    """Apply specified corruption to a batch of unnormalized [0, 1] image tensors [N, 3, 32, 32]
    and return normalized PyTorch tensors [N, 3, 32, 32].

    Args:
        raw_tensors: Unnormalized float tensor [N, 3, 32, 32] in range [0, 1].
        corruption_type: One of ('blur', 'noise', 'brightness', 'contrast', 'rotation').
        severity: Severity level (1-5).
        seed: Seed for reproducible noise generation.

    Returns:
        Normalized tensor [N, 3, 32, 32] ready for model inference.
    """
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    params = SEVERITY_PARAMS[corruption_type][severity]

    if corruption_type == "blur":
        corrupted = TF.gaussian_blur(
            raw_tensors, kernel_size=params["kernel_size"], sigma=params["sigma"]
        )
    elif corruption_type == "brightness":
        corrupted = TF.adjust_brightness(raw_tensors, brightness_factor=params["factor"])
    elif corruption_type == "contrast":
        corrupted = TF.adjust_contrast(raw_tensors, contrast_factor=params["factor"])
    elif corruption_type == "rotation":
        corrupted = TF.rotate(raw_tensors, angle=params["angle"])
    elif corruption_type == "noise":
        std = params["std"]
        g = torch.Generator()
        g.manual_seed(seed + severity * 1000)
        noise = torch.randn(raw_tensors.shape, generator=g) * std
        corrupted = torch.clamp(raw_tensors + noise, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown corruption type: {corruption_type}")

    return normalize(corrupted)


class ShiftedCIFAR10Dataset(Dataset):
    """PyTorch Dataset loading pre-generated shifted CIFAR-10 image tensors and labels."""

    def __init__(self, filepath: Union[str, Path]):
        """Initialize ShiftedCIFAR10Dataset.

        Args:
            filepath: Path to .pt file containing 'images' and 'labels'.
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Shifted dataset file not found: {self.filepath}")

        data = torch.load(self.filepath, map_location="cpu")
        self.images = data["images"]  # [N, 3, 32, 32]
        self.labels = data["labels"]  # [N]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx].item())
