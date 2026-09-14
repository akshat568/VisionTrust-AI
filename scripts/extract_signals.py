import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CIFAR10_MEAN, CIFAR10_STD, get_cifar10_dataloaders, ShiftedCIFAR10Dataset
from src.models import get_baseline_model, run_inference
from src.reliability import (
    compute_confidence,
    compute_entropy,
    build_feature_centroids,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency,
    compute_image_quality,
)
from src.utils.seed import set_seed


def build_training_centroids(
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    metrics_dir: Path,
) -> np.ndarray:
    """Extract training set features and construct 512-dim class centroids strictly from train data.

    Args:
        model: Frozen baseline model.
        config: Baseline configuration dict.
        device: PyTorch device.
        metrics_dir: Output metrics directory.

    Returns:
        Array of shape [10, 512] containing class-conditional feature centroids.
    """
    centroid_path = metrics_dir / "train_feature_centroids.npy"
    if centroid_path.exists():
        print(f"Loading existing training centroids from: {centroid_path}")
        return np.load(centroid_path)

    print("Building class-conditional feature centroids from 45,000 training images...", flush=True)
    seed = config.get("experiment", {}).get("seed", 42)
    dataloaders = get_cifar10_dataloaders(
        data_dir=config.get("dataset", {}).get("data_dir", "data/raw"),
        batch_size=1024,
        num_workers=0,
        val_split=config.get("dataset", {}).get("val_split", 0.1),
        seed=seed,
        download=True,
    )
    train_loader = dataloaders["train"]

    train_out = run_inference(model, train_loader, device)
    train_features = train_out.features.numpy()
    train_targets = train_out.targets.numpy()

    assert train_features.shape[1] == 512, f"Expected 512 feature dim, got {train_features.shape[1]}"
    assert not np.isnan(train_features).any(), "Training features contain NaNs!"
    assert not np.isinf(train_features).any(), "Training features contain Infs!"

    centroids = build_feature_centroids(train_features, train_targets, num_classes=10)
    np.save(centroid_path, centroids)
    print(f"Training centroids constructed and saved to: {centroid_path} (Shape: {centroids.shape})\n")
    return centroids


def extract_all_reliability_signals(
    checkpoint_path: str = "outputs/models/baseline_resnet18_best.pth",
    config_path: str = "configs/baseline.yaml",
):
    """Extract and consolidate all 6 reliability signals for clean test set and 25 shift conditions."""
    ckpt_p = Path(checkpoint_path).resolve()
    config_p = Path(config_path).resolve()

    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint file missing at {ckpt_p}")

    with open(config_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    pred_dir = Path(config.get("paths", {}).get("predictions_dir", "outputs/predictions")).resolve()
    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "outputs/metrics")).resolve()
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — RELIABILITY SIGNALS EXTRACTION (PHASE 6)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load Frozen Baseline Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_p, map_location="cpu")
    model = get_baseline_model(num_classes=10, cifar_stem=True, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded frozen checkpoint from: {ckpt_p}", flush=True)
    print(f"Device: {device}\n", flush=True)

    # 2. Build Training Class Centroids (From Train Set Only)
    centroids = build_training_centroids(model, config, device, metrics_dir)

    # 3. Process Clean Test Set (10,000 samples)
    print("[1/26] Extracting Reliability Signals for Clean Test Set...", flush=True)
    clean_csv_p = pred_dir / "clean_severity_0_predictions.csv"
    if not clean_csv_p.exists():
        clean_csv_p = pred_dir / "baseline_test_predictions.csv"

    df_clean_preds = pd.read_csv(clean_csv_p)
    clean_probs = np.load(pred_dir / "baseline_test_probabilities.npy")
    clean_features = np.load(pred_dir / "baseline_test_features.npy")

    clean_y_true = df_clean_preds["true_label"].values
    clean_preds = df_clean_preds["predicted_label"].values
    clean_correctness = df_clean_preds["correctness"].values
    clean_logits = np.log(np.clip(clean_probs, 1e-12, 1.0))

    raw_clean_dataset = CIFAR10(root="data/raw", train=False, download=False, transform=None)
    raw_clean_tensors = torch.from_numpy(raw_clean_dataset.data).permute(0, 3, 1, 2).float() / 255.0
    norm_transform = transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    norm_clean_tensors = norm_transform(raw_clean_tensors)

    clean_conf = compute_confidence(clean_probs)
    clean_ent = compute_entropy(clean_probs)
    clean_dist = compute_feature_distance(clean_features, clean_preds, centroids)
    clean_ood = compute_ood_energy_score(clean_logits)
    clean_aug_cons = compute_augmentation_consistency(model, norm_clean_tensors, clean_preds, device, batch_size=512)
    clean_qual = compute_image_quality(raw_clean_tensors)

    df_clean_signals = pd.DataFrame(
        {
            "sample_index": np.arange(len(clean_y_true)),
            "shift": "clean",
            "severity": 0,
            "true_label": clean_y_true,
            "predicted_label": clean_preds,
            "correctness": clean_correctness,
            "confidence": clean_conf,
            "entropy": clean_ent,
            "feature_distance": clean_dist,
            "ood_score": clean_ood,
            "augmentation_consistency": clean_aug_cons,
            "brightness": clean_qual["brightness"],
            "contrast": clean_qual["contrast"],
            "sharpness": clean_qual["sharpness"],
            "composite_quality": clean_qual["composite_quality"],
        }
    )

    df_clean_signals.to_csv(metrics_dir / "reliability_signals_clean.csv", index=False)
    clean_signals_array = df_clean_signals[
        [
            "confidence",
            "entropy",
            "feature_distance",
            "ood_score",
            "augmentation_consistency",
            "sharpness",
        ]
    ].to_numpy()
    np.save(pred_dir / "reliability_signals_clean.npy", clean_signals_array)

    print(f"  [OK] Clean Test Signals saved to: {metrics_dir / 'reliability_signals_clean.csv'}\n", flush=True)

    # 4. Process 25 Shifted Test Datasets
    corruptions = ["blur", "noise", "brightness", "contrast", "rotation"]
    severities = [1, 2, 3, 4, 5]

    shifted_records = [df_clean_signals]
    count = 1

    mean_t = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    for corruption in corruptions:
        for severity in severities:
            count += 1
            shifted_path = Path("data/shifted").resolve() / corruption / f"severity_{severity}.pt"
            if not shifted_path.exists():
                raise FileNotFoundError(f"Missing shifted dataset: {shifted_path}")

            # Load existing Phase 4 prediction artifacts
            s_pred_csv = pred_dir / f"{corruption}_severity_{severity}_predictions.csv"
            s_probs_npy = pred_dir / f"{corruption}_severity_{severity}_probabilities.npy"
            s_feats_npy = pred_dir / f"{corruption}_severity_{severity}_features.npy"

            df_s_preds = pd.read_csv(s_pred_csv)
            s_probs = np.load(s_probs_npy)
            s_features = np.load(s_feats_npy)

            s_y_true = df_s_preds["true_label"].values
            s_preds = df_s_preds["predicted_label"].values
            s_correctness = df_s_preds["correctness"].values
            s_logits = np.log(np.clip(s_probs, 1e-12, 1.0))

            # Load normalized tensors and denormalize for image quality
            shifted_payload = torch.load(shifted_path, map_location="cpu")
            norm_shifted_tensors = shifted_payload["images"]
            raw_shifted_tensors = torch.clamp(norm_shifted_tensors * std_t + mean_t, 0.0, 1.0)

            s_conf = compute_confidence(s_probs)
            s_ent = compute_entropy(s_probs)
            s_dist = compute_feature_distance(s_features, s_preds, centroids)
            s_ood = compute_ood_energy_score(s_logits)
            s_aug_cons = compute_augmentation_consistency(model, norm_shifted_tensors, s_preds, device, batch_size=512)
            s_qual = compute_image_quality(raw_shifted_tensors)

            df_s = pd.DataFrame(
                {
                    "sample_index": np.arange(len(s_y_true)),
                    "shift": corruption,
                    "severity": severity,
                    "true_label": s_y_true,
                    "predicted_label": s_preds,
                    "correctness": s_correctness,
                    "confidence": s_conf,
                    "entropy": s_ent,
                    "feature_distance": s_dist,
                    "ood_score": s_ood,
                    "augmentation_consistency": s_aug_cons,
                    "brightness": s_qual["brightness"],
                    "contrast": s_qual["contrast"],
                    "sharpness": s_qual["sharpness"],
                    "composite_quality": s_qual["composite_quality"],
                }
            )
            shifted_records.append(df_s)

            print(
                f"[{count:02d}/26] {corruption:10s} | Sev {severity} -> "
                f"Acc: {np.mean(s_correctness)*100:5.2f}% | "
                f"Ent: {np.mean(s_ent):.3f} | Dist: {np.mean(s_dist):.2f} | "
                f"OOD Energy: {np.mean(s_ood):.2f} | AugCons: {np.mean(s_aug_cons):.3f}",
                flush=True,
            )

    # 5. Consolidate and Save Shifted Signals
    df_all_shifted = pd.concat(shifted_records, ignore_index=True)
    df_all_shifted.to_csv(metrics_dir / "reliability_signals_shifted.csv", index=False)

    print("\n" + "=" * 70, flush=True)
    print("ALL 26 RELIABILITY SIGNALS EXTRACTED AND CONSOLIDATED!", flush=True)
    print(f"Clean Signals saved to:   {metrics_dir / 'reliability_signals_clean.csv'}", flush=True)
    print(f"Shifted Signals saved to: {metrics_dir / 'reliability_signals_shifted.csv'}", flush=True)
    print("=" * 70, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and consolidate Phase 6 reliability signals."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/baseline_resnet18_best.pth",
        help="Path to baseline model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    extract_all_reliability_signals(checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()
