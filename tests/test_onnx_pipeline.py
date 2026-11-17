import io
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.inference.onnx_pipeline import ONNXVisionTrustPipeline
from src.inference.schema import InferenceResult
from src.data.cifar10 import CIFAR10_CLASSES
from src.api.main import app
from src.api.dependencies import init_pipeline

ONNX_MODEL_PATH = "outputs/models/baseline_resnet18.onnx"
CENTROIDS_PATH = "outputs/metrics/train_feature_centroids.npy"
TRUST_MODEL_PATH = "outputs/models/trust_model.pkl"


def test_onnx_pipeline_missing_files():
    """Verify ONNXVisionTrustPipeline raises FileNotFoundError when artifacts are missing."""
    with pytest.raises(FileNotFoundError):
        ONNXVisionTrustPipeline(model_path="non_existent_model.onnx")

    with pytest.raises(FileNotFoundError):
        ONNXVisionTrustPipeline(
            model_path=ONNX_MODEL_PATH,
            centroids_path="non_existent_centroids.npy",
        )

    with pytest.raises(FileNotFoundError):
        ONNXVisionTrustPipeline(
            model_path=ONNX_MODEL_PATH,
            centroids_path=CENTROIDS_PATH,
            trust_model_path="non_existent_trust_model.pkl",
        )


def test_onnx_pipeline_single_prediction():
    """Test ONNXVisionTrustPipeline.predict_image with a synthetic PIL image."""
    pipeline = ONNXVisionTrustPipeline(
        model_path=ONNX_MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
    )

    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)

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


def test_onnx_pipeline_batch_prediction():
    """Test ONNXVisionTrustPipeline.predict_batch."""
    pipeline = ONNXVisionTrustPipeline(
        model_path=ONNX_MODEL_PATH,
        centroids_path=CENTROIDS_PATH,
        trust_model_path=TRUST_MODEL_PATH,
    )

    images = [
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
        Image.fromarray(np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)),
    ]

    results = pipeline.predict_batch(images)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, InferenceResult)
        assert r.reliability_status in {"TRUSTED", "UNTRUSTED"}


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient with ONNX engine loaded."""
    init_pipeline(engine="onnx", force_reload=True)
    with TestClient(app) as test_client:
        yield test_client


def test_api_with_onnx_engine(client: TestClient):
    """Test full FastAPI application endpoints with ONNX inference engine."""
    # 1. Health endpoint
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

    # 2. Predict endpoint
    arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    img_bytes = buf.getvalue()

    resp = client.post("/predict", files={"image": ("test.png", img_bytes, "image/png")})
    assert resp.status_code == 200
    res_data = resp.json()
    assert "prediction" in res_data
    assert "vision" in res_data
    assert "reliability" in res_data
    assert "trust" in res_data
    assert res_data["trust"]["status"] in {"TRUSTED", "UNTRUSTED"}
    assert res_data["latency_ms"] is not None

    # 3. Batch predict endpoint
    files = [
        ("images", ("img1.png", img_bytes, "image/png")),
        ("images", ("img2.png", img_bytes, "image/png")),
    ]
    resp_batch = client.post("/predict/batch", files=files)
    assert resp_batch.status_code == 200
    batch_data = resp_batch.json()
    assert batch_data["total_processed"] == 2
    assert len(batch_data["results"]) == 2
