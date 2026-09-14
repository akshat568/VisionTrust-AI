import argparse
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

from src.reliability import (
    evaluate_single_signal,
    analyze_high_confidence_failures_signals,
)


def plot_reliability_signal_diagnostics(
    df_perf: pd.DataFrame,
    df_clean: pd.DataFrame,
    df_shifted: pd.DataFrame,
    df_high_conf_signals: pd.DataFrame,
    output_dir: Path,
):
    """Generate Phase 6 diagnostic reliability signal plots.

    Args:
        df_perf: Individual signal performance DataFrame.
        df_clean: Clean reliability signals DataFrame.
        df_shifted: Shifted reliability signals DataFrame.
        df_high_conf_signals: High confidence failure signals DataFrame.
        output_dir: Output plots directory path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Individual Signal AUROC Bar Chart
    plt.figure(figsize=(9, 5.5))
    df_sorted = df_perf.sort_values("auroc", ascending=True)

    bars = plt.barh(
        df_sorted["signal_name"],
        df_sorted["auroc"] * 100,
        color="teal",
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(x=50.0, color="gray", linestyle="--", linewidth=1.5, label="Random Chance (50%)")
    plt.title("Individual Reliability Signals — AUROC on Clean CIFAR-10 Test Set", fontsize=13)
    plt.xlabel("AUROC Score (%)", fontsize=11)
    plt.xlim(45.0, 100.0)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "individual_signal_auroc.png", dpi=300)
    plt.close()

    # 2. Signal Distributions: Correct vs Incorrect (2x3 Grid)
    signals_to_plot = [
        ("confidence", "Softmax Confidence", True),
        ("entropy", "Prediction Entropy", False),
        ("feature_distance", "Feature Centroid Distance", False),
        ("ood_score", "OOD Energy Score", True),
        ("augmentation_consistency", "Augmentation Consistency", True),
        ("sharpness", "Laplacian Sharpness", True),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    axes = axes.flatten()

    correct_df = df_clean[df_clean["correctness"] == 1]
    incorrect_df = df_clean[df_clean["correctness"] == 0]

    for idx, (col, title, higher_is_good) in enumerate(signals_to_plot):
        ax = axes[idx]
        if col in df_clean.columns:
            sns.kdeplot(
                correct_df[col],
                ax=ax,
                color="forestgreen",
                fill=True,
                alpha=0.35,
                linewidth=2,
                label="Correct",
            )
            sns.kdeplot(
                incorrect_df[col],
                ax=ax,
                color="crimson",
                fill=True,
                alpha=0.35,
                linewidth=2,
                label="Incorrect",
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Signal Value", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=9)

    plt.suptitle("Reliability Signal Distributions: Correct vs Incorrect Predictions (Clean Test)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "signal_distributions_correct_vs_incorrect.png", dpi=300)
    plt.close()

    # 3. Signals vs Severity across Corruptions
    plt.figure(figsize=(10, 6))
    df_agg = (
        df_shifted.groupby(["shift", "severity"])[
            ["confidence", "entropy", "feature_distance", "ood_score", "augmentation_consistency"]
        ]
        .mean()
        .reset_index()
    )

    corruptions = [c for c in df_agg["shift"].unique() if c != "clean"]
    colors = {
        "blur": "#1f77b4",
        "noise": "#ff7f0e",
        "brightness": "#2ca02c",
        "contrast": "#d62728",
        "rotation": "#9467bd",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Ax1: Feature Distance vs Severity
    clean_dist = df_clean["feature_distance"].mean()
    ax1.axhline(clean_dist, color="black", linestyle="--", label=f"Clean ({clean_dist:.2f})")
    for c in corruptions:
        sub = df_agg[df_agg["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_dist] + sub["feature_distance"].tolist()
        ax1.plot(sevs, vals, marker="o", linewidth=2, color=colors.get(c, "blue"), label=c.capitalize())

    ax1.set_title("Feature Centroid Distance vs Corruption Severity", fontsize=12)
    ax1.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax1.set_ylabel("Mean Feature Distance", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Ax2: Augmentation Consistency vs Severity
    clean_aug = df_clean["augmentation_consistency"].mean()
    ax2.axhline(clean_aug, color="black", linestyle="--", label=f"Clean ({clean_aug:.3f})")
    for c in corruptions:
        sub = df_agg[df_agg["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_aug] + sub["augmentation_consistency"].tolist()
        ax2.plot(sevs, vals, marker="s", linewidth=2, color=colors.get(c, "orange"), label=c.capitalize())

    ax2.set_title("Augmentation Consistency vs Corruption Severity", fontsize=12)
    ax2.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax2.set_ylabel("Mean Augmentation Consistency", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "signal_vs_severity.png", dpi=300)
    plt.close()


def run_signals_evaluation(
    metrics_dir: Path = Path("outputs/metrics"),
    plots_dir: Path = Path("outputs/plots"),
):
    """Run full Phase 6 signal evaluation pipeline."""
    metrics_dir = metrics_dir.resolve()
    plots_dir = plots_dir.resolve()

    clean_signals_path = metrics_dir / "reliability_signals_clean.csv"
    shifted_signals_path = metrics_dir / "reliability_signals_shifted.csv"

    if not clean_signals_path.exists() or not shifted_signals_path.exists():
        raise FileNotFoundError(
            f"Signal metrics missing! Please run `python scripts/extract_signals.py` first."
        )

    df_clean = pd.read_csv(clean_signals_path)
    df_shifted = pd.read_csv(shifted_signals_path)

    print("=" * 70)
    print("VISIONTRUST AI — INDIVIDUAL RELIABILITY SIGNALS EVALUATION (PHASE 6)")
    print("=" * 70)

    # 1. Evaluate Individual Signals on Clean Test Set
    signals_config = [
        ("Confidence", "confidence", True),
        ("Prediction Entropy", "entropy", False),
        ("Feature-Space Distance", "feature_distance", False),
        ("OOD Energy Score", "ood_score", True),
        ("Augmentation Consistency", "augmentation_consistency", True),
        ("Image Sharpness", "sharpness", True),
        ("Composite Quality", "composite_quality", True),
    ]

    perf_records = []
    correctness = df_clean["correctness"].values

    for display_name, col_name, higher_is_good in signals_config:
        if col_name in df_clean.columns:
            vals = df_clean[col_name].values
            res = evaluate_single_signal(vals, correctness, display_name, higher_is_good)
            perf_records.append(res)

    df_perf = pd.DataFrame(perf_records).sort_values("auroc", ascending=False).reset_index(drop=True)
    df_perf.to_csv(metrics_dir / "individual_signal_performance.csv", index=False)

    print("\nINDIVIDUAL RELIABILITY SIGNAL PERFORMANCE (CLEAN CIFAR-10 TEST SET):")
    print("-" * 75)
    for idx, r in df_perf.iterrows():
        print(
            f"[{idx+1}] {r['signal_name']:25s} | AUROC: {r['auroc']*100:5.2f}% | "
            f"AUPRC: {r['auprc']:.4f} | Correct Mean: {r['correct_mean']:7.4f} | "
            f"Incorrect Mean: {r['incorrect_mean']:7.4f} | Direction: {r['direction']}"
        )
    print("-" * 75)
    print(f"Performance summary saved to: {metrics_dir / 'individual_signal_performance.csv'}\n")

    # 2. High-Confidence Failure Signals Analysis
    high_conf_analysis = analyze_high_confidence_failures_signals(df_clean, confidence_threshold=0.90)

    high_conf_records = []
    for sname, sdata in high_conf_analysis["signals"].items():
        high_conf_records.append(
            {
                "signal_name": sname,
                "high_conf_correct_mean": sdata["high_conf_correct_mean"],
                "high_conf_wrong_mean": sdata["high_conf_wrong_mean"],
                "difference": sdata["difference"],
            }
        )

    df_high_conf_signals = pd.DataFrame(high_conf_records)
    df_high_conf_signals.to_csv(metrics_dir / "high_confidence_failure_signals.csv", index=False)
    print("HIGH-CONFIDENCE FAILURES (>=0.90) SIGNAL COMPARISON:")
    print(
        f"  Total High-Confidence Correct: {high_conf_analysis['num_high_conf_correct']:,} | "
        f"Wrong: {high_conf_analysis['num_high_conf_wrong']:,}"
    )
    for _, r in df_high_conf_signals.iterrows():
        print(
            f"  {r['signal_name']:25s} -> Correct Mean: {r['high_conf_correct_mean']:8.4f} | "
            f"Wrong Mean: {r['high_conf_wrong_mean']:8.4f} | Diff: {r['difference']:+8.4f}"
        )
    print(f"High-confidence failure analysis saved to: {metrics_dir / 'high_confidence_failure_signals.csv'}\n")

    # 3. Generate Diagnostic Plots
    plot_reliability_signal_diagnostics(df_perf, df_clean, df_shifted, df_high_conf_signals, plots_dir)
    print(f"Diagnostic plots saved to: {plots_dir}")

    print("\n" + "=" * 70)
    print("PHASE 6 INDIVIDUAL RELIABILITY SIGNALS EVALUATION COMPLETED!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 6 individual reliability signals."
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="outputs/metrics",
        help="Path to metrics directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="outputs/plots",
        help="Path to plots directory.",
    )
    args = parser.parse_args()

    run_signals_evaluation(
        metrics_dir=Path(args.metrics_dir),
        plots_dir=Path(args.plots_dir),
    )


if __name__ == "__main__":
    main()
