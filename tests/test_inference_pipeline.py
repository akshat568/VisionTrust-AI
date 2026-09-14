import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import torch

from src.inference.pipeline import VisionTrustPipeline, InferenceResult
from src.data.cifar10 import CIFAR10_CLASSES

MODEL_PATH = "outputs/models/baseline_resnet18_best.pth"
CENTROIDS_PATH = "outputs/metrics/train_feature_centroids.npy"
TRUST_MODEL_PATH = "outputs/models/trust_model.pkl"


def test_pipeline_missing_files():
    """Verify VisionTrustPipeline raises FileNotFoundError when required artifact paths are missing."""
    with pytest.raises(FileNotFoundError):
        VisionTrustPipeline(model_path="non_existent_model.pth")

    with pytest.raises(FileNotFoundError):
        VisionTrustPipeline(
            model_path=MODEL_PATH,
            centroids_path="non_existent_centroids.npy",
        )

    with pytest.raises(FileNotFoundError):
        VisionTrustPipeline(
            model_path=MODEL_PATH,
            centroids_path=CENTROIDS_PATH,
            trust_model_path="non_existent_trust_model.pkl",
        )


def test_pipeline_single_prediction_pil():
    """Test VisionTrustPipeline.predict_image with a PIL Image input."""
    pipeline = VisionTrustPipeline(
        model_path=MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
        device="cpu",
    )

    # Create dummy 32x32 RGB PIL image
    img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(img_array)

    result = pipeline.predict_image(pil_img)

    assert isinstance(result, InferenceResult)
    assert 0 <= result.predicted_class_id < 10
    assert result.predicted_class_name in CIFAR10_CLASSES
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.probabilities) == 10
    assert pytest.approx(sum(result.probabilities), abs=1e-4) == 1.0
    assert len(result.predicted_logits) == 10

    # Verify reliability signals
    sigs = result.reliability_signals
    assert "confidence" in sigs
    assert "entropy" in sigs
    assert "feature_distance" in sigs
    assert "ood_score" in sigs
    assert "augmentation_consistency" in sigs
    assert "sharpness" in sigs
    assert "brightness" in sigs
    assert "contrast" in sigs
    assert "composite_quality" in sigs

    assert 0.0 <= sigs["confidence"] <= 1.0
    assert sigs["entropy"] >= 0.0
    assert sigs["feature_distance"] >= 0.0
    assert 0.0 <= sigs["augmentation_consistency"] <= 1.0

    # Verify Trust evaluation
    assert 0.0 <= result.trust_score <= 1.0
    assert pytest.approx(result.trust_score + result.failure_probability, abs=1e-5) == 1.0
    assert result.reliability_status in {"TRUSTED", "UNTRUSTED"}


def test_pipeline_single_prediction_filepath():
    """Test VisionTrustPipeline.predict_image with a temporary image file path."""
    pipeline = VisionTrustPipeline(
        model_path=MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
        device="cpu",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_img.png"
        img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(img_path)

        result = pipeline.predict_image(str(img_path))
        assert isinstance(result, InferenceResult)
        assert result.reliability_status in {"TRUSTED", "UNTRUSTED"}


def test_pipeline_batch_prediction():
    """Test VisionTrustPipeline.predict_batch with multiple inputs."""
    pipeline = VisionTrustPipeline(
        model_path=MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
        device="cpu",
    )

    img1 = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    img2 = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))

    results = pipeline.predict_batch([img1, img2])

    assert len(results) == 2
    for r in results:
        assert isinstance(r, InferenceResult)
        assert 0.0 <= r.trust_score <= 1.0


def test_inference_result_to_dict():
    """Verify InferenceResult.to_dict produces valid JSON-serializable dictionary."""
    pipeline = VisionTrustPipeline(
        model_path=MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
        device="cpu",
    )

    img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    result = pipeline.predict_image(img)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert "predicted_class_id" in d
    assert "predicted_class_name" in d
    assert "reliability_signals" in d
    assert "trust_score" in d

    # JSON serialization check
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
