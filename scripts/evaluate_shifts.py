import os
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    SEVERITY_PARAMS,
    ShiftedCIFAR10Dataset,
    get_cifar10_dataloaders,
)
from src.models import get_baseline_model, run_inference
from src.utils.seed import set_seed


def plot_shift_degradation_curves(df_metrics: pd.DataFrame, output_dir: Path):
    """Generate and save clean plots of accuracy, confidence, and accuracy drop vs severity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    corruptions = [c for c in df_metrics["shift"].unique() if c != "clean"]

    colors = {
        "blur": "#1f77b4",
        "noise": "#ff7f0e",
        "brightness": "#2ca02c",
        "contrast": "#d62728",
        "rotation": "#9467bd",
    }

    # 1. Accuracy vs Severity
    plt.figure(figsize=(9, 5.5))
    clean_row = df_metrics[df_metrics["shift"] == "clean"].iloc[0]
    clean_acc = clean_row["accuracy"] * 100

    plt.axhline(
        y=clean_acc,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Clean Baseline ({clean_acc:.2f}%)",
    )

    for corruption in corruptions:
        sub = df_metrics[df_metrics["shift"] == corruption].sort_values("severity")
        severities = [0] + sub["severity"].tolist()
        accuracies = [clean_acc] + (sub["accuracy"] * 100).tolist()

        plt.plot(
            severities,
            accuracies,
            marker="o",
            linewidth=2,
            color=colors.get(corruption, "blue"),
            label=corruption.capitalize(),
        )

    plt.title("Model Accuracy Degradation Under Distribution Shift (ResNet-18)", fontsize=13)
    plt.xlabel("Corruption Severity (0 = Clean, 1-5 = Shifted)", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.xticks(range(0, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_vs_severity.png", dpi=300)
    plt.close()

    # 2. Confidence vs Severity
    plt.figure(figsize=(9, 5.5))
    clean_conf = clean_row["average_confidence"]

    plt.axhline(
        y=clean_conf,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Clean Baseline ({clean_conf:.4f})",
    )

    for corruption in corruptions:
        sub = df_metrics[df_metrics["shift"] == corruption].sort_values("severity")
        severities = [0] + sub["severity"].tolist()
        confidences = [clean_conf] + sub["average_confidence"].tolist()

        plt.plot(
            severities,
            confidences,
            marker="s",
            linewidth=2,
            color=colors.get(corruption, "orange"),
            label=corruption.capitalize(),
        )

    plt.title("Average Model Confidence vs Corruption Severity", fontsize=13)
    plt.xlabel("Corruption Severity (0 = Clean, 1-5 = Shifted)", fontsize=11)
    plt.ylabel("Mean Softmax Confidence", fontsize=11)
    plt.xticks(range(0, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_vs_severity.png", dpi=300)
    plt.close()

    # 3. Accuracy Drop vs Severity
    plt.figure(figsize=(9, 5.5))
    for corruption in corruptions:
        sub = df_metrics[df_metrics["shift"] == corruption].sort_values("severity")
        severities = sub["severity"].tolist()
        acc_drops = (sub["accuracy_drop"] * 100).tolist()

        plt.plot(
            severities,
            acc_drops,
            marker="^",
            linewidth=2,
            color=colors.get(corruption, "red"),
            label=corruption.capitalize(),
        )

    plt.title("Accuracy Drop (%) Relative to Clean Baseline", fontsize=13)
    plt.xlabel("Corruption Severity Level (1 to 5)", fontsize=11)
    plt.ylabel("Accuracy Drop (%)", fontsize=11)
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_drop_vs_severity.png", dpi=300)
    plt.close()


def evaluate_all_shift_conditions(
    checkpoint_path: str = "outputs/models/baseline_resnet18_best.pth",
    config_path: str = "configs/baseline.yaml",
):
    """Evaluate frozen ResNet-18 baseline model on clean test set and all 25 shifted test datasets."""
    ckpt_p = Path(checkpoint_path).resolve()
    if not ckpt_p.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found at {ckpt_p}. Please train the baseline model first!"
        )

    config_p = Path(config_path).resolve()
    with open(config_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    torch.set_num_threads(8)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — FROZEN BASELINE SHIFT EVALUATION (PHASE 4)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Frozen Model Checkpoint
    checkpoint = torch.load(ckpt_p, map_location="cpu")
    print(f"\nLoaded frozen checkpoint from: {ckpt_p}")
    val_acc_val = checkpoint.get("val_acc") or 0.0
    print(f"Model Epoch: {checkpoint.get('epoch')} | Clean Val Accuracy: {val_acc_val * 100:.2f}%")

    model_cfg = config.get("model", {})
    model = get_baseline_model(
        num_classes=model_cfg.get("num_classes", 10),
        cifar_stem=model_cfg.get("cifar_stem", True),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Device: {device}\n")

    pred_dir = Path(config.get("paths", {}).get("predictions_dir", "outputs/predictions")).resolve()
    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "outputs/metrics")).resolve()
    plots_dir = Path(config.get("paths", {}).get("plots_dir", "outputs/plots")).resolve()

    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # 2. Evaluate Clean Baseline Test Set
    print("[1/26] Evaluating Clean Test Set...", flush=True)
    raw_clean = CIFAR10(root="data/raw", train=False, download=True, transform=None)
    clean_tensors = torch.from_numpy(raw_clean.data).permute(0, 3, 1, 2).float() / 255.0
    normalize = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    clean_tensors = normalize(clean_tensors)
    clean_labels = torch.tensor(raw_clean.targets, dtype=torch.long)
    clean_ds = TensorDataset(clean_tensors, clean_labels)
    clean_loader = DataLoader(clean_ds, batch_size=1024, shuffle=False)
    clean_out = run_inference(model, clean_loader, device)

    clean_y_true = clean_out.targets.numpy()
    clean_y_pred = clean_out.predictions.numpy()
    clean_y_conf = clean_out.confidences.numpy()

    clean_acc = float(accuracy_score(clean_y_true, clean_y_pred))
    if clean_acc < 0.70:
        raise ValueError(
            f"VALIDATION CHECK FAILED: Clean test accuracy is implausibly low ({clean_acc * 100:.2f}% < 70.0%). "
            "Phase 4 evaluation aborted! The checkpoint 'outputs/models/baseline_resnet18_best.pth' appears to be undertrained or invalid. "
            "Please train the baseline model properly before running Phase 4."
        )

    clean_p, clean_r, clean_f1, _ = precision_recall_fscore_support(
        clean_y_true, clean_y_pred, average="macro", zero_division=0
    )
    clean_avg_conf = float(np.mean(clean_y_conf))
    clean_correct = int(np.sum(clean_y_pred == clean_y_true))
    clean_incorrect = int(len(clean_y_true) - clean_correct)

    clean_record = {
        "shift": "clean",
        "severity": 0,
        "accuracy": clean_acc,
        "macro_precision": float(clean_p),
        "macro_recall": float(clean_r),
        "macro_f1": float(clean_f1),
        "average_confidence": clean_avg_conf,
        "correct_count": clean_correct,
        "incorrect_count": clean_incorrect,
        "accuracy_drop": 0.0,
        "confidence_change": 0.0,
    }
    records.append(clean_record)

    # Save clean predictions
    df_clean_preds = pd.DataFrame(
        {
            "sample_index": np.arange(len(clean_y_true)),
            "true_label": clean_y_true,
            "predicted_label": clean_y_pred,
            "confidence": clean_y_conf,
            "correctness": clean_out.correctness.numpy(),
        }
    )
    df_clean_preds.to_csv(pred_dir / "clean_severity_0_predictions.csv", index=False)

    print(
        f"  [OK] Clean Baseline -> Accuracy: {clean_acc*100:.2f}% | Avg Conf: {clean_avg_conf:.4f} | F1: {clean_f1:.4f}\n",
        flush=True,
    )

    # 3. Evaluate 25 Shifted Test Conditions
    corruptions = list(SEVERITY_PARAMS.keys())
    severities = [1, 2, 3, 4, 5]
    count = 1

    for corruption in corruptions:
        for severity in severities:
            count += 1
            shifted_path = Path("data/shifted").resolve() / corruption / f"severity_{severity}.pt"
            if not shifted_path.exists():
                raise FileNotFoundError(
                    f"Shifted dataset missing: {shifted_path}. Run `python scripts/generate_shifts.py` first!"
                )

            shifted_ds = ShiftedCIFAR10Dataset(shifted_path)
            shifted_loader = DataLoader(shifted_ds, batch_size=1024, shuffle=False)

            out = run_inference(model, shifted_loader, device)

            y_true = out.targets.numpy()
            y_pred = out.predictions.numpy()
            y_conf = out.confidences.numpy()

            acc = float(accuracy_score(y_true, y_pred))
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            avg_conf = float(np.mean(y_conf))
            correct = int(np.sum(y_pred == y_true))
            incorrect = int(len(y_true) - correct)

            acc_drop = float(clean_acc - acc)
            conf_change = float(avg_conf - clean_avg_conf)

            rec = {
                "shift": corruption,
                "severity": severity,
                "accuracy": acc,
                "macro_precision": float(p),
                "macro_recall": float(r),
                "macro_f1": float(f1),
                "average_confidence": avg_conf,
                "correct_count": correct,
                "incorrect_count": incorrect,
                "accuracy_drop": acc_drop,
                "confidence_change": conf_change,
            }
            records.append(rec)

            # Save predictions & features
            file_prefix = f"{corruption}_severity_{severity}"
            df_preds = pd.DataFrame(
                {
                    "sample_index": np.arange(len(y_true)),
                    "true_label": y_true,
                    "predicted_label": y_pred,
                    "confidence": y_conf,
                    "correctness": out.correctness.numpy(),
                }
            )
            df_preds.to_csv(pred_dir / f"{file_prefix}_predictions.csv", index=False)
            np.save(pred_dir / f"{file_prefix}_features.npy", out.features.numpy())
            np.save(pred_dir / f"{file_prefix}_probabilities.npy", out.probabilities.numpy())

            print(
                f"[{count:02d}/26] {corruption:10s} | Sev {severity} -> "
                f"Acc: {acc*100:6.2f}% (Drop: -{acc_drop*100:5.2f}%) | "
                f"Avg Conf: {avg_conf:.4f} | F1: {f1:.4f}",
                flush=True,
            )

    # 4. Save Consolidated Metrics CSV & JSON
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv(metrics_dir / "distribution_shift_metrics.csv", index=False)

    with open(metrics_dir / "distribution_shift_metrics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    # 5. Generate & Save Plots
    plot_shift_degradation_curves(df_metrics, plots_dir)

    # 6. Save Confusion Matrices for selected severe shifts (Blur Sev 5 & Noise Sev 5)
    for shift_name, sev in [("blur", 5), ("noise", 5)]:
        s_path = Path("data/shifted").resolve() / shift_name / f"severity_{sev}.pt"
        s_ds = ShiftedCIFAR10Dataset(s_path)
        s_loader = DataLoader(s_ds, batch_size=1024, shuffle=False)
        s_out = run_inference(model, s_loader, device)

        cm = confusion_matrix(s_out.targets.numpy(), s_out.predictions.numpy())
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Oranges",
            xticklabels=list(CIFAR10_CLASSES),
            yticklabels=list(CIFAR10_CLASSES),
        )
        plt.title(
            f"Confusion Matrix under Severe Distribution Shift ({shift_name.capitalize()} Severity {sev})",
            fontsize=12,
        )
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{shift_name}_s{sev}.png", dpi=300)
        plt.close()

    print("\n" + "=" * 70)
    print("ALL 26 EVALUATIONS COMPLETED SUCCESSFULLY!")
    print(f"Consolidated Metrics saved to: {metrics_dir / 'distribution_shift_metrics.csv'}")
    print(f"Degradation Plots saved to:      {plots_dir}")
    print(f"Predictions saved to:            {pred_dir}")
    print("=" * 70)

    return df_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate frozen baseline ResNet-18 model on controlled distribution shifts."
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

    evaluate_all_shift_conditions(checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()
