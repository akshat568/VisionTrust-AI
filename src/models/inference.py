from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class InferenceOutput:
    """Structured container holding comprehensive model inference & feature outputs.

    Designed to provide all necessary outputs for downstream classification evaluation
    and Phase 6 reliability analysis.
    """

    logits: torch.Tensor  # [N, C]
    probabilities: torch.Tensor  # [N, C]
    predictions: torch.Tensor  # [N]
    confidences: torch.Tensor  # [N]
    features: torch.Tensor  # [N, D] (512-dim)
    targets: Optional[torch.Tensor] = None  # [N]
    correctness: Optional[torch.Tensor] = None  # [N] (bool/int)

    def to_dict(self):
        """Convert output tensors to a dictionary format."""
        d = {
            "logits": self.logits.numpy(),
            "probabilities": self.probabilities.numpy(),
            "predictions": self.predictions.numpy(),
            "confidences": self.confidences.numpy(),
            "features": self.features.numpy(),
        }
        if self.targets is not None:
            d["targets"] = self.targets.numpy()
        if self.correctness is not None:
            d["correctness"] = self.correctness.numpy()
        return d


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> InferenceOutput:
    """Execute evaluation and feature extraction over a DataLoader.

    Args:
        model: PyTorch classification model (must implement `extract_features` or standard forward).
        dataloader: PyTorch DataLoader containing images (and optional targets).
        device: Device to execute inference on (CPU/CUDA).

    Returns:
        InferenceOutput dataclass containing all predictions, confidences, and feature vectors.
    """
    model.eval()
    model.to(device)

    all_logits = []
    all_probs = []
    all_preds = []
    all_confs = []
    all_features = []
    all_targets = []

    has_targets = False

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, targets = batch[0], batch[1]
            has_targets = True
            all_targets.append(targets.cpu())
        else:
            images = batch
            has_targets = False

        images = images.to(device)

        if hasattr(model, "extract_features"):
            logits, features = model.extract_features(images)
        else:
            logits = model(images)
            features = logits  # Fallback if model doesn't support feature extraction

        probs = F.softmax(logits, dim=-1)
        confs, preds = torch.max(probs, dim=-1)

        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_confs.append(confs.cpu())
        all_features.append(features.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    confs_tensor = torch.cat(all_confs, dim=0)
    features_tensor = torch.cat(all_features, dim=0)

    if has_targets:
        targets_tensor = torch.cat(all_targets, dim=0)
        correctness_tensor = (preds_tensor == targets_tensor).int()
    else:
        targets_tensor = None
        correctness_tensor = None

    return InferenceOutput(
        logits=logits_tensor,
        probabilities=probs_tensor,
        predictions=preds_tensor,
        confidences=confs_tensor,
        features=features_tensor,
        targets=targets_tensor,
        correctness=correctness_tensor,
    )
