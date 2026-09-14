import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_cifar10_dataloaders
from src.models import get_baseline_model, Trainer
from src.utils.seed import set_seed


def plot_training_curves(history, output_dir: Path):
    """Plot and save training vs validation loss and accuracy curves.

    Args:
        history: Dictionary containing loss and accuracy lists per epoch.
        output_dir: Directory to save plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-s", label="Validation Loss")
    plt.title("Training vs Validation Loss (ResNet-18 Baseline)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        [a * 100 for a in history["train_acc"]],
        "b-o",
        label="Training Accuracy",
    )
    plt.plot(
        epochs,
        [a * 100 for a in history["val_acc"]],
        "r-s",
        label="Validation Accuracy",
    )
    plt.title("Training vs Validation Accuracy (ResNet-18 Baseline)", fontsize=13)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline ResNet-18 vision model on CIFAR-10."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    torch.set_num_threads(8)

    print("=" * 65)
    print("VISIONTRUST AI — BASELINE VISION MODEL TRAINING (PHASE 3)")
    print("=" * 65)

    # Load DataLoaders
    ds_cfg = config.get("dataset", {})
    dataloaders = get_cifar10_dataloaders(
        data_dir=ds_cfg.get("data_dir", "data/raw"),
        batch_size=ds_cfg.get("batch_size", 128),
        num_workers=ds_cfg.get("num_workers", 0),
        val_split=ds_cfg.get("val_split", 0.1),
        seed=seed,
        download=True,
    )

    # Instantiate Model
    model_cfg = config.get("model", {})
    model = get_baseline_model(
        num_classes=model_cfg.get("num_classes", 10),
        cifar_stem=model_cfg.get("cifar_stem", True),
        pretrained=False,
    )

    # Instantiate Trainer & Fit
    trainer = Trainer(model=model, config=config)
    history = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        save_name="baseline_resnet18_best.pth",
    )

    # Best checkpoint was saved during trainer.fit
    best_ckpt_path = Path("outputs/models").resolve() / "baseline_resnet18_best.pth"
    print(f"Verified best checkpoint saved to: {best_ckpt_path}")

    # Save plots & metrics
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "outputs/plots")).resolve()
    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "outputs/metrics")).resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(history, plots_dir)
    with open(metrics_dir / "baseline_train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(f"Training loss/accuracy plots saved to: {plots_dir}")
    print(
        f"Training history saved to: {metrics_dir / 'baseline_train_history.json'}"
    )


if __name__ == "__main__":
    main()
