from typing import Dict, Any, Tuple, Optional, Union
import numpy as np


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


def compute_augmentation_consistency(
    model: Any,
    images: Any,
    original_preds: np.ndarray,
    device: Any,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute prediction consistency under deterministic augmentations using PyTorch.

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
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Use compute_augmentation_consistency_onnx instead.")

    with torch.no_grad():
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


def compute_augmentation_consistency_onnx(
    session: Any,
    images: np.ndarray,
    original_preds: np.ndarray,
) -> np.ndarray:
    """Compute prediction consistency under deterministic augmentations using ONNX Runtime.

    Applies 3 light deterministic augmentations (horizontal flip, shift right-down, shift left-up)
    and measures agreement fraction with original prediction in range [0, 1].

    Args:
        session: onnxruntime.InferenceSession.
        images: Image array of shape [N, 3, 32, 32].
        original_preds: Original prediction array of shape [N].

    Returns:
        Consistency score array of shape [N] in range [0, 1].
    """
    n_samples = len(images)

    # 1. Horizontal Flip
    flip_images = np.flip(images, axis=-1)

    # 2. Shift Right-Down (pad 2, crop right-down)
    padded = np.pad(images, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="reflect")
    shift1_images = padded[:, :, 4:36, 4:36]

    # 3. Shift Left-Up (pad 2, crop left-up)
    shift2_images = padded[:, :, 0:32, 0:32]

    all_augs = np.concatenate([flip_images, shift1_images, shift2_images], axis=0).astype(np.float32)

    # Run inference with ONNX Runtime session
    logits = session.run(["logits"], {"input": all_augs})[0]
    all_aug_preds = np.argmax(logits, axis=-1)

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


def compute_image_quality_numpy(images: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute measurable image quality signals on an image NumPy array batch.

    Args:
        images: Image array batch of shape [N, 3, 32, 32] in range [0, 1].

    Returns:
        Dictionary mapping keys ('brightness', 'contrast', 'sharpness', 'composite_quality')
        to NumPy float arrays of shape [N].
    """
    if images.ndim == 3:
        images = images[np.newaxis, ...]

    # 1. Brightness: mean intensity across channels & pixels
    brightness = np.mean(images, axis=(1, 2, 3)).astype(np.float32)

    # 2. Contrast: std intensity across channels & pixels (ddof=1 matches PyTorch unbiased std)
    contrast = np.std(images, axis=(1, 2, 3), ddof=1).astype(np.float32)

    # 3. Sharpness: Variance of 2D Laplacian operator on grayscale image
    # Grayscale conversion: 0.2989 R + 0.5870 G + 0.1140 B
    gray = (
        0.2989 * images[:, 0:1, :, :]
        + 0.5870 * images[:, 1:2, :, :]
        + 0.1140 * images[:, 2:3, :, :]
    )  # [N, 1, 32, 32]

    # Pad with constant 0 on spatial dimensions (axes 2 and 3)
    padded = np.pad(gray, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    lap_res = (
        padded[:, :, 0:32, 1:33]
        + padded[:, :, 2:34, 1:33]
        + padded[:, :, 1:33, 0:32]
        + padded[:, :, 1:33, 2:34]
        - 4.0 * padded[:, :, 1:33, 1:33]
    )
    sharpness = np.var(lap_res, axis=(1, 2, 3), ddof=1).astype(np.float32)

    # Composite Quality: Sharpness normalized by contrast
    composite_quality = (sharpness / (contrast + 1e-6)).astype(np.float32)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "composite_quality": composite_quality,
    }


def compute_image_quality(images: Any) -> Dict[str, np.ndarray]:
    """Compute measurable image quality signals on an image tensor or NumPy array batch.

    Args:
        images: Image tensor or array batch of shape [N, 3, 32, 32] in range [0, 1].

    Returns:
        Dictionary mapping keys ('brightness', 'contrast', 'sharpness', 'composite_quality')
        to NumPy float arrays of shape [N].
    """
    if hasattr(images, "detach"):
        images_np = images.detach().cpu().numpy()
    else:
        images_np = np.asarray(images)

    return compute_image_quality_numpy(images_np)
