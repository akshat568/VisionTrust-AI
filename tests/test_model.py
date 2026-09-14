import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models import ResNet18CIFAR, get_baseline_model, run_inference, Trainer


@pytest.fixture
def dummy_model():
    return get_baseline_model(num_classes=10, cifar_stem=True, pretrained=False)


@pytest.fixture
def dummy_dataloader():
    # 16 dummy 32x32 RGB images
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4)


def test_model_output_shape(dummy_model):
    x = torch.randn(4, 3, 32, 32)
    logits = dummy_model(x)
    assert logits.shape == (4, 10), f"Expected shape (4, 10), got {logits.shape}"


def test_probabilities_and_confidence(dummy_model, dummy_dataloader):
    out = run_inference(dummy_model, dummy_dataloader, device=torch.device("cpu"))

    assert out.logits.shape == (16, 10)
    assert out.probabilities.shape == (16, 10)
    assert out.predictions.shape == (16,)
    assert out.confidences.shape == (16,)

    # Probabilities sum to ~1.0
    prob_sums = out.probabilities.sum(dim=-1)
    assert torch.allclose(
        prob_sums, torch.ones_like(prob_sums), atol=1e-5
    ), "Probabilities do not sum to 1.0!"

    # Predicted classes within [0, 9]
    assert (out.predictions >= 0).all() and (out.predictions < 10).all()

    # Confidences between 0 and 1
    assert (out.confidences >= 0.0).all() and (out.confidences <= 1.0).all()


def test_feature_extraction_dimension(dummy_model):
    x = torch.randn(4, 3, 32, 32)
    logits, features = dummy_model.extract_features(x)

    assert logits.shape == (4, 10)
    assert features.shape == (
        4,
        512,
    ), f"Expected 512-dim feature vector, got {features.shape}"


def test_trainer_smoke_run(dummy_model, dummy_dataloader):
    config = {
        "experiment": {"seed": 42},
        "training": {
            "epochs": 1,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "scheduler": "none",
            "device": "cpu",
        },
        "paths": {"checkpoint_dir": "outputs/models"},
    }

    trainer = Trainer(model=dummy_model, config=config, device=torch.device("cpu"))
    train_loss, train_acc = trainer.train_epoch(dummy_dataloader)
    val_loss, val_acc = trainer.evaluate_epoch(dummy_dataloader)

    assert train_loss >= 0.0
    assert 0.0 <= train_acc <= 1.0
    assert val_loss >= 0.0
    assert 0.0 <= val_acc <= 1.0


def test_checkpoint_save_and_load(dummy_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.pth"
        checkpoint_dict = {
            "epoch": 1,
            "model_state_dict": dummy_model.state_dict(),
            "val_acc": 0.85,
            "feature_dim": 512,
        }
        torch.save(checkpoint_dict, ckpt_path)

        assert ckpt_path.exists()

        # Load back
        loaded = torch.load(ckpt_path, map_location="cpu")
        new_model = get_baseline_model(num_classes=10, cifar_stem=True)
        new_model.load_state_dict(loaded["model_state_dict"])

        assert loaded["epoch"] == 1
        assert loaded["val_acc"] == 0.85
        assert loaded["feature_dim"] == 512
