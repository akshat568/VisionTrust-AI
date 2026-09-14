from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def compute_confidence(probabilities: np.ndarray) -> np.ndarray:
    """Compute maximum class probability confidence score.

    Args:
        probabilities: Array of class probabilities of shape [N, C].

    Returns:
        Confidence array of shape [N] in range [0, 1].
    """
    return np.max(probabilities, axis=-1)


def compute_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Shannon entropy from class probability vectors.

    Args:
        probabilities: Array of class probabilities of shape [N, C].
        eps: Small epsilon to prevent log(0).

    Returns:
        Entropy array of shape [N], non-negative.
    """
    p_clipped = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(p_clipped * np.log(p_clipped), axis=-1)
    return np.maximum(entropy, 0.0)


def build_feature_centroids(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 10,
) -> np.ndarray:
    """Compute class-conditional feature centroids from training set features.

    Must be constructed strictly from training data and labels.

    Args:
        train_features: Training set feature vectors of shape [N_train, D].
        train_labels: Training set target class labels of shape [N_train].
        num_classes: Number of classification classes (default 10).

    Returns:
        Array of shape [num_classes, D] containing mean centroid for each class.
    """
    feature_dim = train_features.shape[1]
    centroids = np.zeros((num_classes, feature_dim), dtype=np.float32)

    for k in range(num_classes):
        mask = train_labels == k
        if np.any(mask):
            centroids[k] = np.mean(train_features[mask], axis=0)
        else:
            centroids[k] = np.zeros(feature_dim, dtype=np.float32)

    return centroids


def compute_feature_distance(
    features: np.ndarray,
    predictions: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute Euclidean distance in feature space to the predicted class centroid.

    Args:
        features: Test/shifted feature vectors of shape [N, D].
        predictions: Predicted class indices of shape [N].
        centroids: Pre-computed training class centroids of shape [C, D].

    Returns:
        Distance array of shape [N], non-negative and finite.
    """
    pred_centroids = centroids[predictions]  # [N, D]
    diff = features - pred_centroids
    distances = np.linalg.norm(diff, axis=-1)
    return distances


def compute_ood_energy_score(
    logits: np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Compute energy-based score from unnormalized classification logits.

    Energy S_energy = T * log(sum_k exp(z_k / T)).
    Higher energy score indicates in-distribution / higher model certainty.

    Args:
        logits: Unnormalized class logits of shape [N, C].
        temperature: Temperature scaling factor T (default 1.0).

    Returns:
        Energy score array of shape [N].
    """
    scaled_logits = logits / temperature
    # Use logsumexp for numerical stability
    max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
    energy = temperature * (
        max_logits.squeeze(-1)
        + np.log(np.sum(np.exp(scaled_logits - max_logits), axis=-1))
    )
    return energy


@torch.no_grad()
def compute_augmentation_consistency(
    model: nn.Module,
    images: torch.Tensor,
    original_preds: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute prediction consistency under deterministic augmentations.

    Applies 3 light deterministic augmentations (horizontal flip, shift right-down, shift left-up)
    and measures agreement fraction with original prediction in range [0, 1].

    Args:
        model: PyTorch baseline model.
        images: Image tensor batch of shape [N, 3, 32, 32].
        original_preds: Original prediction array of shape [N].
        device: PyTorch device.
        batch_size: Batch size for model evaluation.

    Returns:
        Consistency score array of shape [N] in range [0, 1].
    """
    model.eval()
    model.to(device)
    n_samples = len(images)

    # 1. Horizontal Flip
    flip_images = torch.flip(images, dims=[-1])

    # 2. Shift Right-Down (pad 2, crop right-down)
    padded = F.pad(images, (2, 2, 2, 2), mode="reflect")
    shift1_images = padded[:, :, 4:36, 4:36]

    # 3. Shift Left-Up (pad 2, crop left-up)
    shift2_images = padded[:, :, 0:32, 0:32]

    all_augs = torch.cat([flip_images, shift1_images, shift2_images], dim=0)

    aug_preds_list = []
    for idx in range(0, len(all_augs), batch_size):
        batch = all_augs[idx : idx + batch_size].to(device)
        logits = model(batch)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        aug_preds_list.append(preds)

    all_aug_preds = np.concatenate(aug_preds_list, axis=0)
    flip_preds = all_aug_preds[:n_samples]
    shift1_preds = all_aug_preds[n_samples : 2 * n_samples]
    shift2_preds = all_aug_preds[2 * n_samples :]

    matches = (
        (flip_preds == original_preds).astype(np.float32)
        + (shift1_preds == original_preds).astype(np.float32)
        + (shift2_preds == original_preds).astype(np.float32)
    )

    consistency = matches / 3.0
    return consistency


def compute_image_quality(images: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute measurable image quality signals on an image tensor batch.

    Args:
        images: Image tensor batch of shape [N, 3, 32, 32] in range [0, 1].

    Returns:
        Dictionary mapping keys ('brightness', 'contrast', 'sharpness', 'composite_quality')
        to NumPy float arrays of shape [N].
    """
    # 1. Brightness: mean intensity across channels & pixels
    brightness = torch.mean(images, dim=(1, 2, 3)).numpy()

    # 2. Contrast: std intensity across channels & pixels
    contrast = torch.std(images, dim=(1, 2, 3)).numpy()

    # 3. Sharpness: Variance of 2D Laplacian operator on grayscale image
    # Grayscale conversion: 0.2989 R + 0.5870 G + 0.1140 B
    r, g, b = images[:, 0:1, :, :], images[:, 1:2, :, :], images[:, 2:3, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b  # [N, 1, 32, 32]

    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
    ).view(1, 1, 3, 3)

    lap_res = F.conv2d(gray, laplacian_kernel, padding=1)  # [N, 1, 32, 32]
    sharpness = torch.var(lap_res, dim=(1, 2, 3)).numpy()

    # Composite Quality: Sharpness normalized by contrast
    composite_quality = sharpness / (contrast + 1e-6)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "composite_quality": composite_quality,
    }
