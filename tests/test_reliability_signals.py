from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.reliability.signals import (
    compute_confidence,
    compute_entropy,
    build_feature_centroids,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency,
    compute_image_quality,
)
from src.reliability.evaluator import evaluate_single_signal, analyze_high_confidence_failures_signals
from src.models import get_baseline_model


def test_confidence_calculation():
    """Verify confidence is max softmax probability in range [0, 1]."""
    probs = np.array([[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]], dtype=np.float32)
    conf = compute_confidence(probs)
    assert conf.shape == (2,)
    np.testing.assert_allclose(conf, [0.7, 0.4])
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)


def test_entropy_calculation():
    """Verify Shannon entropy calculation, non-negativity, and output dimensions."""
    probs = np.array([[0.5, 0.5], [1.0, 0.0]], dtype=np.float32)
    ent = compute_entropy(probs)

    assert ent.shape == (2,)
    assert np.all(ent >= 0.0), "Entropy contains negative values!"
    assert np.isclose(ent[1], 0.0, atol=1e-5), "Entropy of deterministic distribution should be zero!"
    assert ent[0] > 0.0, "Entropy of uniform distribution should be positive!"


def test_feature_distance_output_shape_and_validity():
    """Verify feature centroid distance output shape, non-negativity, and absence of NaNs/Infs."""
    features = np.random.randn(20, 512).astype(np.float32)
    predictions = np.random.randint(0, 10, size=20)
    centroids = np.random.randn(10, 512).astype(np.float32)

    dist = compute_feature_distance(features, predictions, centroids)

    assert dist.shape == (20,), f"Expected shape (20,), got {dist.shape}"
    assert not np.isnan(dist).any(), "Feature distances contain NaNs!"
    assert not np.isinf(dist).any(), "Feature distances contain Infs!"
    assert np.all(dist >= 0.0), "Feature distances contain negative values!"


def test_ood_energy_score_shape_and_validity():
    """Verify OOD logit energy score output shape and finiteness."""
    logits = np.random.randn(15, 10).astype(np.float32)
    ood_score = compute_ood_energy_score(logits, temperature=1.0)

    assert ood_score.shape == (15,), f"Expected shape (15,), got {ood_score.shape}"
    assert not np.isnan(ood_score).any(), "OOD scores contain NaNs!"
    assert not np.isinf(ood_score).any(), "OOD scores contain Infs!"


def test_augmentation_consistency_bounds():
    """Verify augmentation consistency lies strictly in [0, 1]."""
    model = get_baseline_model(num_classes=10, cifar_stem=True, pretrained=False)
    images = torch.randn(8, 3, 32, 32)
    original_preds = np.random.randint(0, 10, size=8)
    device = torch.device("cpu")

    aug_cons = compute_augmentation_consistency(model, images, original_preds, device, batch_size=4)

    assert aug_cons.shape == (8,)
    assert np.all(aug_cons >= 0.0) and np.all(aug_cons <= 1.0), "Augmentation consistency out of [0, 1]!"


def test_image_quality_metrics_finiteness():
    """Verify image quality signals return finite values."""
    images = torch.rand(10, 3, 32, 32)
    qual = compute_image_quality(images)

    expected_keys = {"brightness", "contrast", "sharpness", "composite_quality"}
    assert set(qual.keys()) == expected_keys

    for k, arr in qual.items():
        assert arr.shape == (10,), f"Key {k} shape mismatch: {arr.shape}"
        assert not np.isnan(arr).any(), f"Quality signal {k} contains NaNs!"
        assert not np.isinf(arr).any(), f"Quality signal {k} contains Infs!"


def test_consolidated_clean_signal_table():
    """Verify clean signal table contains 10,000 samples and all expected signals if file exists."""
    clean_csv = Path("outputs/metrics/reliability_signals_clean.csv")
    if not clean_csv.exists():
        pytest.skip("reliability_signals_clean.csv not yet generated")

    df_clean = pd.read_csv(clean_csv)
    assert len(df_clean) == 10000, f"Expected 10,000 clean test samples, got {len(df_clean)}"

    required_cols = [
        "sample_index",
        "true_label",
        "predicted_label",
        "correctness",
        "confidence",
        "entropy",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
        "brightness",
        "contrast",
        "sharpness",
        "composite_quality",
    ]
    for col in required_cols:
        assert col in df_clean.columns, f"Missing column {col} in clean signal table!"
