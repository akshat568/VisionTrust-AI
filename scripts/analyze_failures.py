import argparse
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CIFAR10_CLASSES
from src.evaluation import (
    analyze_clean_failures,
    analyze_high_confidence_errors,
    analyze_confidence_bins,
    analyze_class_failures,
    analyze_confusion,
    analyze_shift_failures,
    extract_high_confidence_failures,
)


def generate_failure_plots(
    df_clean: pd.DataFrame,
    df_bins: pd.DataFrame,
    cm: np.ndarray,
    df_shift: pd.DataFrame,
    output_dir: Path,
):
    """Generate diagnostic failure analysis plots.

    Args:
        df_clean: Clean test predictions DataFrame.
        df_bins: Confidence bins DataFrame.
        cm: Confusion matrix numpy array.
        df_shift: Shift failure analysis DataFrame.
        output_dir: Output directory path for saving plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    correct_conf = df_clean[df_clean["correctness"] == 1]["confidence"]
    incorrect_conf = df_clean[df_clean["correctness"] == 0]["confidence"]

    # 1. Confidence Distribution (Overlay Density)
    plt.figure(figsize=(9, 5.5))
    sns.kdeplot(
        correct_conf,
        color="green",
        fill=True,
        alpha=0.35,
        linewidth=2,
        label=f"Correct Predictions (n={len(correct_conf):,})",
    )
    sns.kdeplot(
        incorrect_conf,
        color="crimson",
        fill=True,
        alpha=0.35,
        linewidth=2,
        label=f"Incorrect Predictions (n={len(incorrect_conf):,})",
    )
    plt.title("Softmax Confidence Distribution: Correct vs Incorrect Predictions (Clean Test)", fontsize=13)
    plt.xlabel("Softmax Confidence Score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.xlim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_distribution.png", dpi=300)
    plt.close()

    # 2. Confidence Histogram: Correct Predictions
    plt.figure(figsize=(8, 5))
    plt.hist(
        correct_conf,
        bins=20,
        range=(0.0, 1.0),
        color="forestgreen",
        edgecolor="black",
        alpha=0.75,
    )
    plt.title("Confidence Histogram: Correct Predictions (Clean CIFAR-10 Test Set)", fontsize=13)
    plt.xlabel("Softmax Confidence", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_hist_correct.png", dpi=300)
    plt.close()

    # 3. Confidence Histogram: Incorrect Predictions
    plt.figure(figsize=(8, 5))
    plt.hist(
        incorrect_conf,
        bins=20,
        range=(0.0, 1.0),
        color="firebrick",
        edgecolor="black",
        alpha=0.75,
    )
    plt.title("Confidence Histogram: Incorrect Predictions (Clean CIFAR-10 Test Set)", fontsize=13)
    plt.xlabel("Softmax Confidence", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_hist_incorrect.png", dpi=300)
    plt.close()

    # 4. Confidence-Bin Accuracy Curve (Reliability Diagram)
    plt.figure(figsize=(8.5, 6))
    bin_centers = df_bins["average_confidence"]
    bin_accs = df_bins["bin_accuracy"] * 100
    bin_counts = df_bins["num_predictions"]

    plt.plot([0, 100], [0, 100], "k--", linewidth=1.5, label="Perfect Calibration (Ideal)")
    plt.plot(
        bin_centers * 100,
        bin_accs,
        "s-",
        color="darkblue",
        linewidth=2.5,
        markersize=8,
        label="Observed Empirical Accuracy",
    )

    for idx, row in df_bins.iterrows():
        if row["num_predictions"] > 0:
            plt.annotate(
                f"n={row['num_predictions']}",
                (row["average_confidence"] * 100, row["bin_accuracy"] * 100),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="navy",
            )

    plt.title("Empirical Accuracy vs Softmax Confidence Bins (Clean CIFAR-10)", fontsize=13)
    plt.xlabel("Average Bin Softmax Confidence (%)", fontsize=11)
    plt.ylabel("Empirical Accuracy (%)", fontsize=11)
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_bin_accuracy.png", dpi=300)
    plt.close()

    # 5. Failure Confusion Matrix
    plt.figure(figsize=(10, 8.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=list(CIFAR10_CLASSES),
        yticklabels=list(CIFAR10_CLASSES),
        cbar=True,
    )
    plt.title("Baseline ResNet-18 Confusion Matrix (Clean CIFAR-10 Test Set)", fontsize=13)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "failure_confusion_matrix.png", dpi=300)
    plt.close()

    # 6. High-Confidence Errors Under Distribution Shift
    plt.figure(figsize=(9.5, 5.5))
    corruptions = [c for c in df_shift["shift"].unique() if c != "clean"]
    colors = {
        "blur": "#1f77b4",
        "noise": "#ff7f0e",
        "brightness": "#2ca02c",
        "contrast": "#d62728",
        "rotation": "#9467bd",
    }

    for corruption in corruptions:
        sub = df_shift[df_shift["shift"] == corruption].sort_values("severity")
        severities = sub["severity"].tolist()
        frac_wrong_090 = (sub["frac_wrong_ge_090"] * 100).tolist()

        plt.plot(
            severities,
            frac_wrong_090,
            marker="o",
            linewidth=2,
            color=colors.get(corruption, "blue"),
            label=corruption.capitalize(),
        )

    plt.title("Percentage of Errors with Confidence >= 0.90 Across Distribution Shifts", fontsize=13)
    plt.xlabel("Corruption Severity Level (1 to 5)", fontsize=11)
    plt.ylabel("High-Confidence Wrong Fraction (% of Errors)", fontsize=11)
    plt.xticks(range(1, 6))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "high_confidence_errors_vs_shift.png", dpi=300)
    plt.close()


def run_failure_analysis_pipeline(
    prediction_dir: Path = Path("outputs/predictions"),
    metrics_dir: Path = Path("outputs/metrics"),
    plots_dir: Path = Path("outputs/plots"),
):
    """Run full Phase 5 failure analysis pipeline."""
    prediction_dir = prediction_dir.resolve()
    metrics_dir = metrics_dir.resolve()
    plots_dir = plots_dir.resolve()

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("VISIONTRUST AI — FAILURE ANALYSIS OF THE VISION PREDICTOR (PHASE 5)")
    print("=" * 75)

    # 1. Load Clean Test Predictions
    clean_path = prediction_dir / "clean_severity_0_predictions.csv"
    if not clean_path.exists():
        clean_path = prediction_dir / "baseline_test_predictions.csv"

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean prediction file not found at {clean_path}. Please run Phase 3/4 evaluation first!"
        )

    df_clean = pd.read_csv(clean_path)
    print(f"\nLoaded clean predictions from: {clean_path}")
    print(f"Total Clean Samples: {len(df_clean):,}\n")

    # 2. Global Clean Failure Analysis
    clean_summary = analyze_clean_failures(df_clean)
    print("-" * 65)
    print("CLEAN TEST FAILURE SUMMARY:")
    print(f"  Total Predictions:     {clean_summary['total_predictions']:,}")
    print(f"  Correct Predictions:   {clean_summary['correct_predictions']:,} ({clean_summary['accuracy']*100:.2f}%)")
    print(f"  Incorrect Predictions: {clean_summary['incorrect_predictions']:,} ({clean_summary['incorrect']['count']/clean_summary['total_predictions']*100:.2f}%)")
    print(f"  Overall Avg Conf:      {clean_summary['average_confidence']:.4f} (Median: {clean_summary['median_confidence']:.4f})")
    print(f"  Correct Avg Conf:      {clean_summary['correct']['average_confidence']:.4f} (Std: {clean_summary['correct']['std_confidence']:.4f})")
    print(f"  Incorrect Avg Conf:    {clean_summary['incorrect']['average_confidence']:.4f} (Std: {clean_summary['incorrect']['std_confidence']:.4f})")
    print("-" * 65)

    with open(metrics_dir / "clean_failure_summary.json", "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=4)

    # 3. High-Confidence Threshold Analysis (Clean)
    df_high_conf_th = analyze_high_confidence_errors(df_clean, thresholds=[0.80, 0.90, 0.95])
    print("\nHIGH-CONFIDENCE WRONG PREDICTIONS (CLEAN TEST):")
    for _, r in df_high_conf_th.iterrows():
        print(
            f"  Conf >= {r['confidence_threshold']:.2f} -> Total: {int(r['num_predictions']):5d} | "
            f"Correct: {int(r['num_correct']):5d} | Incorrect: {int(r['num_incorrect']):4d} | "
            f"Error Rate: {r['error_rate']*100:5.2f}% | "
            f"% of All Wrong: {r['pct_of_all_wrong']:5.2f}%"
        )

    # 4. Confidence Binning
    df_bins = analyze_confidence_bins(df_clean, num_bins=10)
    df_bins.to_csv(metrics_dir / "confidence_bins.csv", index=False)
    print(f"\nConfidence Bins CSV saved to: {metrics_dir / 'confidence_bins.csv'}")

    # 5. Class-Level Failure Analysis
    df_class = analyze_class_failures(df_clean, CIFAR10_CLASSES)
    df_class.to_csv(metrics_dir / "class_failure_analysis.csv", index=False)
    print(f"Class Failure Analysis CSV saved to: {metrics_dir / 'class_failure_analysis.csv'}")

    # 6. Confusion Matrix & Top Confusion Pairs
    cm, df_top_conf = analyze_confusion(df_clean, CIFAR10_CLASSES)
    df_top_conf.to_csv(metrics_dir / "top_confusions.csv", index=False)
    print(f"Top Confusions CSV saved to: {metrics_dir / 'top_confusions.csv'}")

    print("\nTOP 5 MOST FREQUENT CONFUSION PAIRS (CLEAN TEST):")
    for idx, r in df_top_conf.head(5).iterrows():
        print(
            f"  [{idx+1}] {r['true_class']:10s} -> Predicted as {r['predicted_class']:10s} | "
            f"Count: {int(r['count']):4d} | Avg Conf: {r['average_confidence']:.4f}"
        )

    # 7. Distribution Shift Failure Analysis across all 26 conditions
    df_shift = analyze_shift_failures(prediction_dir)
    df_shift.to_csv(metrics_dir / "shift_failure_analysis.csv", index=False)
    print(f"\nShift Failure Analysis CSV saved to: {metrics_dir / 'shift_failure_analysis.csv'}")

    # 8. High-Confidence Failures Metadata Extraction
    df_high_conf_failures = extract_high_confidence_failures(prediction_dir, threshold=0.90)
    df_high_conf_failures.to_csv(metrics_dir / "high_confidence_failures.csv", index=False)
    print(f"High-Confidence Failures (>=0.90) CSV saved to: {metrics_dir / 'high_confidence_failures.csv'} ({len(df_high_conf_failures):,} entries)")

    # 9. Generate Plots
    generate_failure_plots(df_clean, df_bins, cm, df_shift, plots_dir)
    print(f"\nDiagnostic plots generated under: {plots_dir}")

    print("\n" + "=" * 75)
    print("PHASE 5 FAILURE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 5 failure analysis of the vision predictor."
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="outputs/predictions",
        help="Path to prediction CSV directory.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Path to metrics output directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="Path to plots output directory.",
    )
    args = parser.parse_args()

    run_failure_analysis_pipeline(
        prediction_dir=Path(args.prediction_dir),
        metrics_dir=Path(args.metrics_dir),
        plots_dir=Path(args.plots_dir),
    )


if __name__ == "__main__":
    main()
