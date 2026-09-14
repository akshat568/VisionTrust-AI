from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.data.cifar10 import CIFAR10_CLASSES
from src.data.transforms import CIFAR10_MEAN, CIFAR10_STD
from src.models.resnet import ResNet18CIFAR, get_baseline_model
from src.reliability.signals import (
    compute_confidence,
    compute_entropy,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency,
    compute_image_quality,
)
from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS


@dataclass
class InferenceResult:
    """Encapsulates prediction output, reliability signals, and trust scores for an image."""

    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    predicted_logits: List[float]
    probabilities: List[float]
    reliability_signals: Dict[str, float]
    trust_score: float
    reliability_status: str
    failure_probability: float
    decision_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert InferenceResult to a plain Python dictionary suitable for JSON serialization."""
        data = asdict(self)
        # Ensure all floating point values are plain python float
        data["confidence"] = float(data["confidence"])
        data["trust_score"] = float(data["trust_score"])
        data["failure_probability"] = float(data["failure_probability"])
        data["decision_threshold"] = float(data["decision_threshold"])
        data["predicted_logits"] = [float(x) for x in data["predicted_logits"]]
        data["probabilities"] = [float(x) for x in data["probabilities"]]
        data["reliability_signals"] = {
            k: float(v) for k, v in data["reliability_signals"].items()
        }
        return data


class VisionTrustPipeline:
    """Unified inference pipeline combining baseline ResNet-18 vision classifier and Phase 7 Trust Model."""

    def __init__(
        self,
        model_path: str = "outputs/models/baseline_resnet18_best.pth",
        centroids_path: str = "outputs/metrics/train_feature_centroids.npy",
        trust_model_path: str = "outputs/models/trust_model.pkl",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.50,
    ):
        """Initialize VisionTrust unified inference pipeline.

        Args:
            model_path: Path to trained ResNet-18 baseline weights.
            centroids_path: Path to pre-computed training feature centroids.
            trust_model_path: Path to trained Trust Model pickle artifact.
            device: PyTorch device ('cuda', 'cpu', or None for auto-detection).
            threshold: Trust score threshold for classification as 'TRUSTED' vs 'UNTRUSTED'.
        """
        self.model_path = Path(model_path).resolve()
        self.centroids_path = Path(centroids_path).resolve()
        self.trust_model_path = Path(trust_model_path).resolve()
        self.threshold = float(threshold)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Validate existence of artifact files
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Baseline ResNet-18 checkpoint not found at '{self.model_path}'. "
                f"Please ensure model file exists."
            )
        if not self.centroids_path.exists():
            raise FileNotFoundError(
                f"Training feature centroids not found at '{self.centroids_path}'. "
                f"Please run feature centroid extraction script first."
            )
        if not self.trust_model_path.exists():
            raise FileNotFoundError(
                f"Trust Model artifact not found at '{self.trust_model_path}'. "
                f"Please run trust model training script first."
            )

        # 1. Load Baseline ResNet-18 Vision Model
        self.model: ResNet18CIFAR = get_baseline_model(num_classes=10, cifar_stem=True)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # 2. Load Training Feature Centroids
        self.centroids: np.ndarray = np.load(self.centroids_path)

        # 3. Load Phase 7 Trust Model Pipeline
        self.trust_model: TrustModelPipeline = TrustModelPipeline.load(
            str(self.trust_model_path)
        )

        # Define image transformation pipelines
        self.raw_transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ]
        )
        self.norm_transform = transforms.Normalize(
            mean=CIFAR10_MEAN, std=CIFAR10_STD
        )

    def _preprocess_input(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert any valid image input into (raw_tensor, normalized_tensor).

        Args:
            image_input: Image path, PIL Image, or NumPy array (HWC or CHW).

        Returns:
            Tuple of (raw_tensor [1, 3, 32, 32] in [0, 1], norm_tensor [1, 3, 32, 32]).
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image file not found: '{img_path}'")
            pil_img = Image.open(img_path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255.0).astype(np.uint8)
                else:
                    image_input = image_input.astype(np.uint8)
            if image_input.ndim == 2:
                pil_img = Image.fromarray(image_input).convert("RGB")
            elif image_input.ndim == 3:
                if image_input.shape[0] in (1, 3):
                    image_input = np.transpose(image_input, (1, 2, 0))
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                raise ValueError(f"Invalid numpy image shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        raw_tensor = self.raw_transform(pil_img).unsqueeze(0)  # [1, 3, 32, 32]
        norm_tensor = self.norm_transform(raw_tensor.clone())  # [1, 3, 32, 32]

        return raw_tensor, norm_tensor

    @torch.no_grad()
    def predict_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> InferenceResult:
        """Run single image through unified inference pipeline.

        Args:
            image_input: Image file path, PIL Image, or NumPy array.

        Returns:
            InferenceResult containing vision prediction, 6 signal categories, and Trust score.
        """
        raw_tensor, norm_tensor = self.to_device(
            self._preprocess_input(image_input)
        )

        norm_tensor_dev = norm_tensor.to(self.device)
        logits_tensor, features_tensor = self.model.extract_features(norm_tensor_dev)

        logits = logits_tensor.cpu().numpy()[0]
        features = features_tensor.cpu().numpy()[0]
        probs = F.softmax(logits_tensor, dim=-1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        pred_name = CIFAR10_CLASSES[pred_id]
        conf = float(compute_confidence(probs[None, :])[0])
        ent = float(compute_entropy(probs[None, :])[0])
        dist = float(
            compute_feature_distance(
                features[None, :], np.array([pred_id]), self.centroids
            )[0]
        )
        ood = float(compute_ood_energy_score(logits[None, :])[0])

        aug_cons = float(
            compute_augmentation_consistency(
                self.model, norm_tensor_dev, np.array([pred_id]), self.device
            )[0]
        )

        qual_dict = compute_image_quality(raw_tensor.cpu())
        sharpness = float(qual_dict["sharpness"][0])
        brightness = float(qual_dict["brightness"][0])
        contrast = float(qual_dict["contrast"][0])
        composite_quality = float(qual_dict["composite_quality"][0])

        signals_dict = {
            "confidence": conf,
            "entropy": ent,
            "feature_distance": dist,
            "ood_score": ood,
            "augmentation_consistency": aug_cons,
            "sharpness": sharpness,
            "brightness": brightness,
            "contrast": contrast,
            "composite_quality": composite_quality,
        }

        df_signals = pd.DataFrame([signals_dict])[FEATURE_COLUMNS]
        trust_score = float(
            self.trust_model.predict_proba(df_signals, model_type="rf")[0]
        )

        status = "TRUSTED" if trust_score >= self.threshold else "UNTRUSTED"
        failure_prob = float(1.0 - trust_score)

        return InferenceResult(
            predicted_class_id=pred_id,
            predicted_class_name=pred_name,
            confidence=conf,
            predicted_logits=[float(x) for x in logits],
            probabilities=[float(x) for x in probs],
            reliability_signals=signals_dict,
            trust_score=trust_score,
            reliability_status=status,
            failure_probability=failure_prob,
            decision_threshold=self.threshold,
        )

    def to_device(
        self, tensors: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensors[0], tensors[1]

    @torch.no_grad()
    def predict_batch(
        self, image_inputs: List[Union[str, Path, Image.Image, np.ndarray]]
    ) -> List[InferenceResult]:
        """Run batch of images through unified inference pipeline.

        Args:
            image_inputs: List of image file paths, PIL Images, or NumPy arrays.

        Returns:
            List of InferenceResult objects.
        """
        results = [self.predict_image(img_in) for img_in in image_inputs]
        return results
