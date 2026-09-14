import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import CIFAR10
import yaml

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_cifar10_datasets,
    get_cifar10_dataloaders,
)
from src.models import get_baseline_model, run_inference
from src.reliability import (
    compute_confidence,
    compute_entropy,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency,
    compute_image_quality,
)
from src.utils.seed import set_seed


def extract_validation_signals(
    checkpoint_path: str = "outputs/models/baseline_resnet18_best.pth",
    config_path: str = "configs/baseline.yaml",
):
    """Extract and consolidate Phase 6 reliability signals for the clean 5,000 validation set."""
    ckpt_p = Path(checkpoint_path).resolve()
    config_p = Path(config_path).resolve()

    if not ckpt_p.exists():
        raise FileNotFoundError(f"Checkpoint missing at {ckpt_p}")

    with open(config_p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed = config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    metrics_dir = Path(config.get("paths", {}).get("metrics_dir", "outputs/metrics")).resolve()
    metrics_dir.mkdir(parents=True, exist_ok=True)
    val_csv_p = metrics_dir / "reliability_signals_val.csv"

    if val_csv_p.exists():
        print(f"Validation signals already exist at: {val_csv_p}")
        return

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — EXTRACTING VALIDATION RELIABILITY SIGNALS", flush=True)
    print("=" * 70, flush=True)

    # 1. Device and Model Loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_p, map_location="cpu")
    model = get_baseline_model(num_classes=10, cifar_stem=True, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Load Training Feature Centroids (Built strictly from train data)
    centroids_p = metrics_dir / "train_feature_centroids.npy"
    if not centroids_p.exists():
        raise FileNotFoundError(f"Training centroids missing at {centroids_p}")
    centroids = np.load(centroids_p)

    # 3. Load Validation Data
    dataloaders = get_cifar10_dataloaders(
        data_dir=config.get("dataset", {}).get("data_dir", "data/raw"),
        batch_size=512,
        num_workers=0,
        val_split=config.get("dataset", {}).get("val_split", 0.1),
        seed=seed,
        download=True,
    )
    val_loader = dataloaders["val"]

    print("Running vision model inference on 5,000 clean validation samples...", flush=True)
    val_out = run_inference(model, val_loader, device)

    val_probs = val_out.probabilities.numpy()
    val_features = val_out.features.numpy()
    val_y_true = val_out.targets.numpy()
    val_preds = val_out.predictions.numpy()
    val_correctness = (val_preds == val_y_true).astype(np.int32)
    val_logits = np.log(np.clip(val_probs, 1e-12, 1.0))

    # 4. Extract Raw Tensors for Quality & Augmentation Consistency
    train_ds, val_ds, _, split_indices = get_cifar10_datasets(
        data_dir=config.get("dataset", {}).get("data_dir", "data/raw"),
        val_split=config.get("dataset", {}).get("val_split", 0.1),
        seed=seed,
        download=True,
    )
    val_indices = split_indices["val"]
    raw_train_dataset = CIFAR10(root="data/raw", train=True, download=False, transform=None)
    raw_val_imgs = raw_train_dataset.data[val_indices]  # Shape (5000, 32, 32, 3)

    raw_val_tensors = torch.from_numpy(raw_val_imgs).permute(0, 3, 1, 2).float() / 255.0
    mean_t = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    norm_val_tensors = (raw_val_tensors - mean_t) / std_t

    # 5. Calculate Reliability Signals
    val_conf = compute_confidence(val_probs)
    val_ent = compute_entropy(val_probs)
    val_dist = compute_feature_distance(val_features, val_preds, centroids)
    val_ood = compute_ood_energy_score(val_logits)
    val_aug_cons = compute_augmentation_consistency(model, norm_val_tensors, val_preds, device, batch_size=512)
    val_qual = compute_image_quality(raw_val_tensors)

    df_val = pd.DataFrame(
        {
            "sample_index": np.arange(len(val_y_true)),
            "shift": "clean_val",
            "severity": 0,
            "true_label": val_y_true,
            "predicted_label": val_preds,
            "correctness": val_correctness,
            "confidence": val_conf,
            "entropy": val_ent,
            "feature_distance": val_dist,
            "ood_score": val_ood,
            "augmentation_consistency": val_aug_cons,
            "brightness": val_qual["brightness"],
            "contrast": val_qual["contrast"],
            "sharpness": val_qual["sharpness"],
            "composite_quality": val_qual["composite_quality"],
        }
    )

    df_val.to_csv(val_csv_p, index=False)
    print(f"[OK] Validation Signals extracted and saved to: {val_csv_p} ({len(df_val)} samples)\n")


def main():
    parser = argparse.ArgumentParser(description="Extract validation reliability signals.")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/baseline_resnet18_best.pth")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    args = parser.parse_args()

    extract_validation_signals(checkpoint_path=args.checkpoint, config_path=args.config)


if __name__ == "__main__":
    main()
