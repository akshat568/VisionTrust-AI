import io
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.config import MAX_UPLOAD_SIZE_BYTES
from src.data.cifar10 import CIFAR10_CLASSES


@pytest.fixture(scope="module")
def client():
    """Module-level TestClient triggering lifespan startup (loading pipeline once)."""
    with TestClient(app) as test_client:
        yield test_client


def create_dummy_image_bytes(width: int = 32, height: int = 32, format: str = "PNG") -> bytes:
    """Generate dummy image binary bytes for test uploads."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_health_endpoint(client: TestClient):
    """Test GET /health returns status ok and model_loaded true without running inference."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_single_valid_image(client: TestClient):
    """Test POST /predict with a real valid image upload."""
    img_bytes = create_dummy_image_bytes(32, 32, "PNG")

    files = {"image": ("test_frog.png", img_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()

    # 1. Prediction schema check
    assert "prediction" in data
    pred = data["prediction"]
    assert 0 <= pred["class_id"] < 10
    assert pred["class_name"] in CIFAR10_CLASSES

    # 2. Vision schema check
    assert "vision" in data
    vis = data["vision"]
    assert 0.0 <= vis["confidence"] <= 1.0
    assert vis["entropy"] >= 0.0
    assert len(vis["probabilities"]) == 10
    assert pytest.approx(sum(vis["probabilities"]), abs=1e-4) == 1.0

    # 3. Reliability schema check
    assert "reliability" in data
    rel = data["reliability"]
    assert rel["feature_distance"] >= 0.0
    assert "ood_score" in rel
    assert 0.0 <= rel["augmentation_consistency"] <= 1.0
    assert "sharpness" in rel
    assert "brightness" in rel
    assert "contrast" in rel
    assert "composite_quality" in rel

    # 4. Trust schema check
    assert "trust" in data
    trust = data["trust"]
    assert 0.0 <= trust["trust_probability"] <= 1.0
    assert 0.0 <= trust["failure_probability"] <= 1.0
    assert pytest.approx(trust["trust_probability"] + trust["failure_probability"], abs=1e-5) == 1.0
    assert trust["status"] in {"TRUSTED", "UNTRUSTED"}

    # 5. Latency check
    assert "latency_ms" in data
    assert data["latency_ms"] is not None and data["latency_ms"] > 0.0


def test_predict_batch_images(client: TestClient):
    """Test POST /predict/batch with multiple image uploads."""
    img1 = create_dummy_image_bytes(32, 32, "PNG")
    img2 = create_dummy_image_bytes(64, 64, "JPEG")

    files = [
        ("images", ("img1.png", img1, "image/png")),
        ("images", ("img2.jpg", img2, "image/jpeg")),
    ]

    response = client.post("/predict/batch", files=files)
    assert response.status_code == 200
    data = response.json()

    assert data["total_processed"] == 2
    assert len(data["results"]) == 2

    for res in data["results"]:
        assert 0 <= res["prediction"]["class_id"] < 10
        assert res["trust"]["status"] in {"TRUSTED", "UNTRUSTED"}


def test_predict_empty_upload(client: TestClient):
    """Test POST /predict with empty bytes raises 400 Bad Request."""
    files = {"image": ("empty.png", b"", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Empty image file" in response.json()["detail"]


def test_predict_unsupported_file_extension(client: TestClient):
    """Test POST /predict with invalid file extension raises 400 Bad Request."""
    files = {"image": ("script.py", b"print('hello')", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_predict_corrupt_image_bytes(client: TestClient):
    """Test POST /predict with invalid image content raises 400 Bad Request."""
    files = {"image": ("corrupt.png", b"not an image data string", "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "not a valid or readable image" in response.json()["detail"]


def test_predict_file_size_exceeded(client: TestClient):
    """Test POST /predict with oversized payload raises 413 Payload Too Large."""
    oversized_bytes = b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024)
    files = {"image": ("huge.png", oversized_bytes, "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "exceeds maximum allowable limit" in response.json()["detail"]
