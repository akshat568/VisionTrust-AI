import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CIFAR10_CLASSES, get_cifar10_dataloaders
from src.models import get_baseline_model, run_inference
from src.utils.seed import set_seed


def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: Path):
    """Plot and save confusion matrix heatmap.

    Args:
        cm: Confusion matrix array [num_classes, num_classes].
        class_names: List of class name strings.
        output_path: Path to save output plot file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title("Baseline ResNet-18 Confusion Matrix (Untouched CIFAR-10 Test Set)", fontsize=13)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_baseline(
    checkpoint_path: str = "outputs/models/baseline_resnet18_best.pth",
    config_path: str = "configs/baseline.yaml",
):
    """Execute evaluation of the trained baseline model on the untouched test set."""
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at {ckpt_p}. Please train the model first!"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    print("=" * 65)
    print("VISIONTRUST AI — BASELINE MODEL TEST EVALUATION (PHASE 3)")
    print("=" * 65)

    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_p, map_location="cpu")
    print(f"\nLoaded checkpoint from: {ckpt_p}")
    print(f"Saved Epoch: {checkpoint.get('epoch')} | Validation Accuracy: {checkpoint.get('val_acc')*100:.2f}%")

    # 2. Instantiate & Load Model
    model_cfg = config.get("model", {})
    model = get_baseline_model(
        num_classes=model_cfg.get("num_classes", 10),
        cifar_stem=model_cfg.get("cifar_stem", True),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # 3. Load Test DataLoader
    ds_cfg = config.get("dataset", {})
    dataloaders = get_cifar10_dataloaders(
        data_dir=ds_cfg.get("data_dir", "data/raw"),
        batch_size=ds_cfg.get("batch_size", 128),
        num_workers=ds_cfg.get("num_workers", 0),
        seed=seed,
        download=True,
    )
    test_loader = dataloaders["test"]

    # 4. Device Selection & Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device} over {len(test_loader.dataset):,} test samples...")

    out = run_inference(model, test_loader, device)

    # 5. Compute Metrics
    y_true = out.targets.numpy()
    y_pred = out.predictions.numpy()
    y_conf = out.confidences.numpy()
    y_correct = out.correctness.numpy()

    test_acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    # Compute test loss
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = criterion(out.logits, out.targets).item()

    cm = confusion_matrix(y_true, y_pred)
    cls_report = classification_report(
        y_true, y_pred, target_names=CIFAR10_CLASSES, output_dict=True
    )

    print("\n" + "-" * 65)
    print(f"TEST RESULTS (Untouched 10,000 CIFAR-10 Test Images):")
    print(f"  Test Loss:        {test_loss:.4f}")
    print(f"  Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"  Macro Precision:  {macro_p:.4f}")
    print(f"  Macro Recall:     {macro_r:.4f}")
    print(f"  Macro F1 Score:   {macro_f1:.4f}")
    print(f"  Feature Dim:      {out.features.shape[1]}")
    print("-" * 65)

    # 6. Save Predictions & Feature Vectors
    pred_dir = Path(config.get("paths", {}).get("predictions_dir", "outputs/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)

    df_preds = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_true)),
            "true_label": y_true,
            "true_class": [CIFAR10_CLASSES[i] for i in y_true],
            "predicted_label": y_pred,
            "predicted_class": [CIFAR10_CLASSES[i] for i in y_pred],
            "confidence": y_conf,
            "correctness": y_correct,
        }
    )
    df_preds.to_csv(pred_dir / "baseline_test_predictions.csv", index=False)
    np.save(pred_dir / "baseline_test_features.npy", out.features.numpy())
    np.save(pred_dir / "baseline_test_probabilities.npy", out.probabilities.numpy())

    # 7. Save Metrics JSON
    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "outputs/metrics"))
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "model_name": "ResNet-18 Baseline",
        "num_test_samples": int(len(y_true)),
        "feature_dim": int(out.features.shape[1]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_metrics": cls_report,
        "confusion_matrix": cm.tolist(),
    }
    with open(metrics_dir / "baseline_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    # 8. Save Plot
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "outputs/plots"))
    plot_confusion_matrix(cm, list(CIFAR10_CLASSES), plots_dir / "confusion_matrix.png")

    print(f"Predictions saved to: {pred_dir / 'baseline_test_predictions.csv'}")
    print(f"Features saved to:    {pred_dir / 'baseline_test_features.npy'}")
    print(f"Metrics saved to:     {metrics_dir / 'baseline_test_metrics.json'}")
    print(f"Plot saved to:        {plots_dir / 'confusion_matrix.png'}")
    print("=" * 65)

    return metrics_payload


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline ResNet-18 model on CIFAR-10 test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/baseline_resnet18_best.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    evaluate_baseline(checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()
