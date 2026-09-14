from typing import Tuple
from torchvision import transforms

# Official CIFAR-10 channel mean and standard deviation
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(
    augment_training: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
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
