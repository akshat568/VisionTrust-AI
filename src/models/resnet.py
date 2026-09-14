from typing import Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18CIFAR(nn.Module):
    """ResNet-18 vision classifier adapted for CIFAR-10.

    Supports dual output: standard classification logits and internal 512-dimensional
    penultimate feature representation for downstream reliability/uncertainty analysis.
    """

    def __init__(
        self,
        num_classes: int = 10,
        cifar_stem: bool = True,
        pretrained: bool = False,
    ):
        """Initialize ResNet18CIFAR.

        Args:
            num_classes: Number of output classification classes (default 10).
            cifar_stem: If True, replace 7x7 conv & maxpool with 3x3 conv stride 1 for 32x32 images.
            pretrained: If True, load ImageNet weights (default False: train from scratch).
        """
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        if cifar_stem:
            # Replace 7x7 stride 2 conv with 3x3 stride 1 conv for 32x32 images
            backbone.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # Remove aggressive max pooling for 32x32 inputs
            backbone.maxpool = nn.Identity()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.feature_dim: int = 512
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass extracting both classification logits and penultimate feature vectors.

        Args:
            x: Input image tensor of shape [B, 3, H, W].

        Returns:
            Tuple of (logits [B, num_classes], features [B, 512]).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # [B, 512]
        logits = self.fc(features)  # [B, num_classes]

        return logits, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning classification logits.

        Args:
            x: Input image tensor [B, 3, H, W].

        Returns:
            Logits tensor [B, num_classes].
        """
        logits, _ = self.extract_features(x)
        return logits


def get_baseline_model(
    num_classes: int = 10,
    cifar_stem: bool = True,
    pretrained: bool = False,
) -> ResNet18CIFAR:
    """Factory function to instantiate the baseline ResNet-18 vision model.

    Args:
        num_classes: Number of target classes.
        cifar_stem: Whether to use 32x32 CIFAR-optimized stem.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Instantiated ResNet18CIFAR PyTorch model.
    """
    return ResNet18CIFAR(
        num_classes=num_classes, cifar_stem=cifar_stem, pretrained=pretrained
    )
