import pytest
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.corruptions import (
    SEVERITY_PARAMS,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_brightness,
    apply_contrast,
    apply_rotation,
    apply_corruption_to_pil,
    ShiftedCIFAR10Dataset,
)
from src.models import get_baseline_model, run_inference


@pytest.fixture
def sample_pil_image():
    # 32x32 RGB image
    return Image.new("RGB", (32, 32), color=(128, 64, 200))


def test_corruptions_preserve_shape(sample_pil_image):
    for corruption in ["blur", "brightness", "contrast", "rotation", "noise"]:
        for severity in [1, 3, 5]:
            tensor_out = apply_corruption_to_pil(
                sample_pil_image, corruption_type=corruption, severity=severity
            )
            assert tensor_out.shape == (
                3,
                32,
                32,
            ), f"{corruption} severity {severity} altered image dimensions to {tensor_out.shape}"


def test_severity_parameters_validity():
    for c_name, s_dict in SEVERITY_PARAMS.items():
        assert len(s_dict) == 5
        # Verify severity 1..5 exist
        for sev in range(1, 6):
            assert sev in s_dict


def test_deterministic_gaussian_noise():
    raw_tensor = torch.rand(3, 32, 32)
    noisy1 = apply_gaussian_noise(raw_tensor, severity=3, seed=42)
    noisy2 = apply_gaussian_noise(raw_tensor, severity=3, seed=42)
    noisy3 = apply_gaussian_noise(raw_tensor, severity=3, seed=999)

    assert torch.equal(noisy1, noisy2), "Noise with same seed should be identical!"
    assert not torch.equal(noisy1, noisy3), "Noise with different seed should differ!"


def test_model_inference_on_shifted_batch():
    model = get_baseline_model(num_classes=10, cifar_stem=True)
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 10, (8,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4)

    out = run_inference(model, loader, device=torch.device("cpu"))
    assert out.logits.shape == (8, 10)
    assert out.features.shape == (8, 512)
    assert out.predictions.shape == (8,)
    assert out.correctness.shape == (8,)
