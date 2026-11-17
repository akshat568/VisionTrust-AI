from typing import Tuple, Any

try:
    from torchvision import transforms
except ImportError:
    transforms = None

from src.inference.constants import CIFAR10_MEAN, CIFAR10_STD


def get_cifar10_transforms(
    augment_training: bool = True,
) -> Tuple[Any, Any]:
    """Get preprocessing transformations for training, validation, and testing on CIFAR-10.

    Args:
        augment_training: If True, apply standard training augmentation (RandomCrop, RandomHorizontalFlip).
                          If False, use deterministic transforms for training as well.

    Returns:
        Tuple of (train_transform, eval_transform).
        eval_transform is strictly deterministic (ToTensor + Normalize) without any random augmentation.
    """
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    if augment_training:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    else:
        train_transform = eval_transform

    return train_transform, eval_transform
