import argparse
import json
from pathlib import Path
import sys
from typing import List
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import VisionTrustPipeline, InferenceResult

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def print_single_report(image_path: str, result: InferenceResult) -> None:
    """Print formatted human-readable report for a single image prediction."""
    sigs = result.reliability_signals
    print("=" * 60)
    print("VISIONTRUST AI — INFERENCE & RELIABILITY REPORT")
    print("=" * 60)
    print(f"Image Input:                {image_path}")
    print(f"Predicted Class:            {result.predicted_class_name} (ID: {result.predicted_class_id})")
    print(f"Prediction Confidence:      {result.confidence * 100:.2f}%")
    print("-" * 60)
    print("RELIABILITY SIGNALS:")
    print(f"  1. Softmax Confidence:       {sigs['confidence']:.4f}")
    print(f"  2. Prediction Entropy:       {sigs['entropy']:.4f}")
    print(f"  3. Feature Distance:         {sigs['feature_distance']:.4f}")
    print(f"  4. OOD / Energy Score:       {sigs['ood_score']:.4f}")
    print(f"  5. Augmentation Consistency: {sigs['augmentation_consistency']:.4f}")
    print(f"  6. Image Quality Metrics:")
    print(f"     - Sharpness:              {sigs['sharpness']:.4f}")
    print(f"     - Brightness:             {sigs['brightness']:.4f}")
    print(f"     - Contrast:               {sigs['contrast']:.4f}")
    print(f"     - Composite Quality:      {sigs['composite_quality']:.4f}")
    print("-" * 60)
    print("TRUST MODEL EVALUATION:")
    print(f"  Trust Score P(correct):     {result.trust_score * 100:.2f}%")
    print(f"  Failure Probability:        {result.failure_probability * 100:.2f}%")
    print(f"  Decision Threshold:         {result.decision_threshold:.2f}")
    print(f"  Reliability Status:         [{result.reliability_status}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="VisionTrust AI — Unified Inference CLI"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=str, help="Path to single input image file."
    )
    group.add_argument(
        "--directory", type=str, help="Path to directory containing input images for batch prediction."
    )

    parser.add_argument(
        "--json", action="store_true", help="Output result as JSON string (single image mode only)."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.50, help="Decision threshold for trust classification (default: 0.50)."
    )
    parser.add_argument(
        "--model-path", type=str, default="outputs/models/baseline_resnet18_best.pth", help="Path to baseline model checkpoint."
    )
    parser.add_argument(
        "--centroids-path", type=str, default="outputs/metrics/train_feature_centroids.npy", help="Path to centroids .npy file."
    )
    parser.add_argument(
        "--trust-model-path", type=str, default="outputs/models/trust_model.pkl", help="Path to trust model .pkl file."
    )
    parser.add_argument(
        "--output-csv", type=str, default="outputs/predictions.csv", help="Path to save CSV output in batch directory mode."
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')."
    )

    args = parser.parse_args()

    # Initialize Pipeline
    pipeline = VisionTrustPipeline(
        model_path=args.model_path,
        centroids_path=args.centroids_path,
        trust_model_path=args.trust_model_path,
        device=args.device,
        threshold=args.threshold,
    )

    if args.image:
        result = pipeline.predict_image(args.image)
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print_single_report(args.image, result)

    elif args.directory:
        dir_path = Path(args.directory).resolve()
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: '{dir_path}'")

        image_files = sorted(
            [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        )

        if not image_files:
            print(f"No matching images with extensions {IMAGE_EXTENSIONS} found in '{dir_path}'.")
            sys.exit(0)

        print(f"Found {len(image_files)} image files in '{dir_path}'. Running batch inference...")
        results: List[InferenceResult] = pipeline.predict_batch(image_files)

        rows = []
        for img_path, res in zip(image_files, results):
            sigs = res.reliability_signals
            rows.append(
                {
                    "filename": img_path.name,
                    "predicted_class_id": res.predicted_class_id,
                    "predicted_class_name": res.predicted_class_name,
                    "confidence": res.confidence,
                    "entropy": sigs["entropy"],
                    "feature_distance": sigs["feature_distance"],
                    "ood_score": sigs["ood_score"],
                    "augmentation_consistency": sigs["augmentation_consistency"],
                    "sharpness": sigs["sharpness"],
                    "brightness": sigs["brightness"],
                    "contrast": sigs["contrast"],
                    "composite_quality": sigs["composite_quality"],
                    "trust_score": res.trust_score,
                    "failure_probability": res.failure_probability,
                    "reliability_status": res.reliability_status,
                }
            )

        df = pd.DataFrame(rows)
        out_csv = Path(args.output_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

        trusted_cnt = (df["reliability_status"] == "TRUSTED").sum()
        untrusted_cnt = (df["reliability_status"] == "UNTRUSTED").sum()

        print("=" * 60)
        print("BATCH INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Total Images Processed: {len(df)}")
        print(f"Trusted Count:          {trusted_cnt} ({trusted_cnt / len(df) * 100:.1f}%)")
        print(f"Untrusted Count:        {untrusted_cnt} ({untrusted_cnt / len(df) * 100:.1f}%)")
        print(f"Output CSV saved to:    {out_csv}")
        print("=" * 60)


if __name__ == "__main__":
    main()
