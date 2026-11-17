"""Production ONNX Runtime Inference Pipeline for VisionTrust AI.

Provides high-performance, low-memory CPU inference targeting <= 512 MB RAM environments:
- ResNet-18 FP32 dual-output ONNX execution (logits and 512-D penultimate features)
- Pure PIL + NumPy deterministic image preprocessing (zero torchvision dependency)
- Pure NumPy 2D Laplacian image quality evaluation (zero PyTorch conv2d dependency)
- Vectorized ONNX K=3 augmentation consistency evaluation
- Direct 2D NumPy feature array feeding into Scikit-learn Random Forest (zero pandas dependency)
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.inference.constants import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from src.inference.schema import InferenceResult
from src.reliability.signals import (
    compute_confidence,
    compute_entropy,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency_onnx,
    compute_image_quality_numpy,
)
from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS


class ONNXVisionTrustPipeline:
    """Production inference pipeline using ONNX Runtime CPU and pure NumPy signals."""

    def __init__(
        self,
        model_path: str = "outputs/models/baseline_resnet18.onnx",
        centroids_path: str = "outputs/metrics/train_feature_centroids.npy",
        trust_model_path: str = "outputs/models/trust_model.pkl",
        threshold: float = 0.50,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
    ):
        """Initialize ONNXVisionTrustPipeline.

        Args:
            model_path: Path to exported dual-output ONNX model.
            centroids_path: Path to pre-computed training feature centroids.
            trust_model_path: Path to trained Scikit-learn Trust Model pickle.
            threshold: Trust score threshold for classification as 'TRUSTED' vs 'UNTRUSTED'.
            intra_op_num_threads: Threads for parallel computation within an operator (default 1).
            inter_op_num_threads: Threads for parallel execution of independent nodes (default 1).
        """
        if ort is None:
            raise ImportError(
                "onnxruntime is not installed. Please install 'onnxruntime' to run ONNXVisionTrustPipeline."
            )

        self.model_path = Path(model_path).resolve()
        self.centroids_path = Path(centroids_path).resolve()
        self.trust_model_path = Path(trust_model_path).resolve()
        self.threshold = float(threshold)

        # Validate existence of artifact files
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model artifact not found at '{self.model_path}'. "
                f"Please run 'python scripts/export_onnx.py' first."
            )
        if not self.centroids_path.exists():
            raise FileNotFoundError(
                f"Training feature centroids not found at '{self.centroids_path}'. "
                f"Please ensure feature centroids artifact exists."
            )
        if not self.trust_model_path.exists():
            raise FileNotFoundError(
                f"Trust Model artifact not found at '{self.trust_model_path}'. "
                f"Please ensure trust model pickle exists."
            )

        # 1. Initialize ONNX Runtime Session (optimized for constrained CPU environments)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads
        opts.inter_op_num_threads = inter_op_num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # Verify model input and output names
        self.input_name = self.session.get_inputs()[0].name
        output_names = [out.name for out in self.session.get_outputs()]
        if "logits" not in output_names or "features" not in output_names:
            raise ValueError(
                f"ONNX model outputs must include 'logits' and 'features'. Got: {output_names}"
            )

        # 2. Load Training Feature Centroids [10, 512]
        self.centroids: np.ndarray = np.load(self.centroids_path)

        # 3. Load Scikit-learn Trust Model Pipeline
        self.trust_model: TrustModelPipeline = TrustModelPipeline.load(
            str(self.trust_model_path)
        )

        # Precompute CIFAR-10 normalization constants as float32 tensors [1, 3, 1, 1]
        self.cifar_mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        self.cifar_std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def _preprocess_input(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert input image into (raw_array, normalized_array) without torchvision.

        Args:
            image_input: Image path, PIL Image, or NumPy array.

        Returns:
            Tuple of (raw_arr [1, 3, 32, 32] in [0, 1], norm_arr [1, 3, 32, 32]).
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

        # Deterministic Bilinear resize to 32x32 (identical to torchvision.transforms.Resize((32, 32)))
        pil_resized = pil_img.resize((32, 32), Image.Resampling.BILINEAR)

        # Convert to float32 [0, 1] and transpose from HWC [32, 32, 3] to CHW [3, 32, 32]
        arr_hwc = np.asarray(pil_resized, dtype=np.float32) / 255.0
        raw_arr = np.transpose(arr_hwc, (2, 0, 1))[np.newaxis, ...]  # [1, 3, 32, 32]

        # Standard CIFAR-10 Normalization: (raw - mean) / std
        norm_arr = (raw_arr - self.cifar_mean) / self.cifar_std

        return raw_arr.astype(np.float32), norm_arr.astype(np.float32)

    def predict_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> InferenceResult:
        """Run single image through ONNX production inference pipeline.

        Args:
            image_input: Image file path, PIL Image, or NumPy array.

        Returns:
            InferenceResult containing vision prediction, 6 signal categories, and Trust score.
        """
        raw_arr, norm_arr = self._preprocess_input(image_input)

        # 1. Forward pass through dual-output ONNX model
        ort_outputs = self.session.run(
            ["logits", "features"],
            {self.input_name: norm_arr},
        )
        logits = ort_outputs[0][0]  # [10]
        features = ort_outputs[1][0]  # [512]

        # Numerically stable softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # 2. Extract Vision & Reliability Signals
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
            compute_augmentation_consistency_onnx(
                self.session, norm_arr, np.array([pred_id])
            )[0]
        )

        qual_dict = compute_image_quality_numpy(raw_arr)
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

        # 3. Direct NumPy array feed into Scikit-learn Trust Model (zero pandas)
        feature_vector = np.array(
            [[signals_dict[k] for k in FEATURE_COLUMNS]], dtype=np.float32
        )
        trust_score = float(
            self.trust_model.predict_proba(feature_vector, model_type="rf")[0]
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

    def predict_batch(
        self, image_inputs: List[Union[str, Path, Image.Image, np.ndarray]]
    ) -> List[InferenceResult]:
        """Run batch of images through ONNX production inference pipeline.

        Args:
            image_inputs: List of image file paths, PIL Images, or NumPy arrays.

        Returns:
            List of InferenceResult objects.
        """
        return [self.predict_image(img_in) for img_in in image_inputs]
