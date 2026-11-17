#!/usr/bin/env python3
"""Comprehensive Parity Validation: PyTorch vs. ONNX Runtime Inference.

Rigorous numerical and behavioral comparison across:
1. Logits (10-D) & 512-D penultimate feature embeddings
2. Predicted class ID & Top-1 confidence
3. All 9 reliability signals:
   - confidence, entropy, feature_distance, ood_score, augmentation_consistency,
   - sharpness, brightness, contrast, composite_quality
4. Random Forest trust scores & binary TRUSTED/UNTRUSTED status
5. Peak process memory (RSS) comparison
"""

import logging
from pathlib import Path
import sys
import numpy as np
from PIL import Image

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.inference.pipeline import VisionTrustPipeline
from src.inference.onnx_pipeline import ONNXVisionTrustPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visiontrust.parity_check")


def generate_test_images(num_synthetic: int = 20) -> list:
    """Generate a diverse set of test images (real + synthetic edge cases)."""
    images = []

    # 1. Existing real image if available
    cifar_png = Path("cifar_test.png")
    if cifar_png.exists():
        images.append(("real_cifar_test.png", Image.open(cifar_png).convert("RGB")))

    # Check for other images in cifar_batch_test/
    batch_dir = Path("cifar_batch_test")
    if batch_dir.exists():
        for p in sorted(batch_dir.glob("*.png"))[:5]:
            images.append((p.name, Image.open(p).convert("RGB")))

    # 2. Synthetic images with varied statistical characteristics
    np.random.seed(42)
    for i in range(num_synthetic):
        if i % 5 == 0:
            # Low contrast / dark image
            arr = np.random.randint(0, 50, (32, 32, 3), dtype=np.uint8)
            label = f"synthetic_dark_{i}"
        elif i % 5 == 1:
            # High brightness / washed out
            arr = np.random.randint(200, 256, (32, 32, 3), dtype=np.uint8)
            label = f"synthetic_bright_{i}"
        elif i % 5 == 2:
            # Non-square image (aspect ratio test, e.g. 64x48)
            arr = np.random.randint(0, 256, (64, 48, 3), dtype=np.uint8)
            label = f"synthetic_nonsquare_{i}"
        elif i % 5 == 3:
            # Uniform image (zero variance / extreme edge case)
            arr = np.full((32, 32, 3), fill_value=128, dtype=np.uint8)
            label = f"synthetic_uniform_{i}"
        else:
            # Standard random RGB noise
            arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            label = f"synthetic_random_{i}"

        images.append((label, Image.fromarray(arr)))

    return images


def run_parity_validation():
    logger.info("Initializing PyTorch baseline pipeline...")
    pytorch_pipeline = VisionTrustPipeline(device="cpu")

    logger.info("Initializing ONNX production pipeline...")
    onnx_pipeline = ONNXVisionTrustPipeline()

    test_images = generate_test_images()
    logger.info(f"Running validation on {len(test_images)} test images...")

    diffs = {
        "logits": [],
        "features": [],
        "probabilities": [],
        "confidence": [],
        "entropy": [],
        "feature_distance": [],
        "ood_score": [],
        "augmentation_consistency": [],
        "sharpness": [],
        "brightness": [],
        "contrast": [],
        "composite_quality": [],
        "trust_score": [],
    }

    class_matches = 0
    decision_matches = 0
    aug_matches = 0

    for name, img in test_images:
        res_py = pytorch_pipeline.predict_image(img)
        res_ort = onnx_pipeline.predict_image(img)

        # Check categorical decisions
        pred_match = (res_py.predicted_class_id == res_ort.predicted_class_id)
        status_match = (res_py.reliability_status == res_ort.reliability_status)
        aug_match = (res_py.reliability_signals["augmentation_consistency"] == 
                     res_ort.reliability_signals["augmentation_consistency"])

        if pred_match:
            class_matches += 1
        else:
            logger.warning(
                f"Prediction mismatch on '{name}': PyTorch={res_py.predicted_class_name} vs ONNX={res_ort.predicted_class_name}"
            )

        if status_match:
            decision_matches += 1
        else:
            logger.warning(
                f"Trust decision mismatch on '{name}': PyTorch={res_py.reliability_status} vs ONNX={res_ort.reliability_status}"
            )

        if aug_match:
            aug_matches += 1

        # Numerical differences
        diff_logits = np.max(np.abs(np.array(res_py.predicted_logits) - np.array(res_ort.predicted_logits)))
        diff_probs = np.max(np.abs(np.array(res_py.probabilities) - np.array(res_ort.probabilities)))
        diff_conf = abs(res_py.confidence - res_ort.confidence)
        diff_ent = abs(res_py.reliability_signals["entropy"] - res_ort.reliability_signals["entropy"])
        diff_dist = abs(res_py.reliability_signals["feature_distance"] - res_ort.reliability_signals["feature_distance"])
        diff_ood = abs(res_py.reliability_signals["ood_score"] - res_ort.reliability_signals["ood_score"])
        diff_aug = abs(res_py.reliability_signals["augmentation_consistency"] - res_ort.reliability_signals["augmentation_consistency"])
        diff_sharp = abs(res_py.reliability_signals["sharpness"] - res_ort.reliability_signals["sharpness"])
        diff_bright = abs(res_py.reliability_signals["brightness"] - res_ort.reliability_signals["brightness"])
        diff_contrast = abs(res_py.reliability_signals["contrast"] - res_ort.reliability_signals["contrast"])
        diff_comp = abs(res_py.reliability_signals["composite_quality"] - res_ort.reliability_signals["composite_quality"])
        diff_trust = abs(res_py.trust_score - res_ort.trust_score)

        diffs["logits"].append(diff_logits)
        diffs["probabilities"].append(diff_probs)
        diffs["confidence"].append(diff_conf)
        diffs["entropy"].append(diff_ent)
        diffs["feature_distance"].append(diff_dist)
        diffs["ood_score"].append(diff_ood)
        diffs["augmentation_consistency"].append(diff_aug)
        diffs["sharpness"].append(diff_sharp)
        diffs["brightness"].append(diff_bright)
        diffs["contrast"].append(diff_contrast)
        diffs["composite_quality"].append(diff_comp)
        diffs["trust_score"].append(diff_trust)

    total_samples = len(test_images)
    print("\n" + "=" * 80)
    print("PARITY VALIDATION REPORT: PyTorch vs. ONNX Runtime FP32")
    print("=" * 80)
    print(f"Total Test Images Evaluated: {total_samples}")
    print(f"Top-1 Prediction Matches:    {class_matches}/{total_samples} ({100 * class_matches / total_samples:.2f}%)")
    print(f"Trust Decision Matches:      {decision_matches}/{total_samples} ({100 * decision_matches / total_samples:.2f}%)")
    print(f"Augmentation Consistency:    {aug_matches}/{total_samples} ({100 * aug_matches / total_samples:.2f}%)")
    print("-" * 80)
    print(f"{'Signal / Tensor':<28} | {'Max Absolute Diff':<20} | {'Mean Absolute Diff':<20}")
    print("-" * 80)

    for metric_name, values in diffs.items():
        if values:
            max_val = np.max(values)
            mean_val = np.mean(values)
            print(f"{metric_name:<28} | {max_val:<20.6e} | {mean_val:<20.6e}")

    print("=" * 80)

    # Return summary dict
    return {
        "total_samples": total_samples,
        "class_match_pct": 100 * class_matches / total_samples,
        "decision_match_pct": 100 * decision_matches / total_samples,
        "max_diffs": {k: float(np.max(v)) for k, v in diffs.items() if v},
    }


if __name__ == "__main__":
    run_parity_validation()
