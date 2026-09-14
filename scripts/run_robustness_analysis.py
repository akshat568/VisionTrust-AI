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

from src.reliability import TrustModelPipeline, evaluate_cross_shift_robustness
from src.utils.seed import set_seed


def plot_corruption_family_robustness(
    df_all_conditions: pd.DataFrame,
    output_dir: Path,
):
    """Generate 2x2 grid diagnostic plot of Trust Model robustness across corruption families."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    corruptions = [c for c in df_all_conditions["shift"].unique() if c != "clean"]
    colors = {
        "blur": "#1f77b4",
        "noise": "#ff7f0e",
        "brightness": "#2ca02c",
        "contrast": "#d62728",
        "rotation": "#9467bd",
    }

    clean_row = df_all_conditions[df_all_conditions["shift"] == "clean"].iloc[0]

    # 1. Vision Model Accuracy vs Severity
    ax1 = axes[0]
    ax1.axhline(clean_row["vision_accuracy"] * 100, color="black", linestyle="--", label=f"Clean ({clean_row['vision_accuracy']*100:.2f}%)")
    for c in corruptions:
        sub = df_all_conditions[df_all_conditions["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_row["vision_accuracy"] * 100] + (sub["vision_accuracy"] * 100).tolist()
        ax1.plot(sevs, vals, marker="o", linewidth=2, color=colors.get(c, "blue"), label=c.capitalize())

    ax1.set_title("Vision Predictor Accuracy vs Corruption Severity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax1.set_ylabel("Classification Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # 2. Mean Trust Probability vs Severity
    ax2 = axes[1]
    ax2.axhline(clean_row["mean_trust_prob"], color="black", linestyle="--", label=f"Clean ({clean_row['mean_trust_prob']:.3f})")
    for c in corruptions:
        sub = df_all_conditions[df_all_conditions["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_row["mean_trust_prob"]] + sub["mean_trust_prob"].tolist()
        ax2.plot(sevs, vals, marker="s", linewidth=2, color=colors.get(c, "orange"), label=c.capitalize())

    ax2.set_title("Mean Trust Probability P(correct) vs Corruption Severity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax2.set_ylabel("Mean Trust Probability", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    # 3. Failure Detection AUROC vs Severity
    ax3 = axes[2]
    ax3.axhline(clean_row["failure_auroc"] * 100, color="black", linestyle="--", label=f"Clean ({clean_row['failure_auroc']*100:.2f}%)")
    for c in corruptions:
        sub = df_all_conditions[df_all_conditions["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_row["failure_auroc"] * 100] + (sub["failure_auroc"] * 100).tolist()
        ax3.plot(sevs, vals, marker="^", linewidth=2, color=colors.get(c, "green"), label=c.capitalize())

    ax3.set_title("Failure Detection AUROC vs Corruption Severity", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax3.set_ylabel("Failure AUROC (%)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=9)

    # 4. Retained Accuracy at 80% Coverage vs Severity
    ax4 = axes[3]
    ax4.axhline(clean_row["retained_acc_80_trust"] * 100, color="black", linestyle="--", label=f"Clean ({clean_row['retained_acc_80_trust']*100:.2f}%)")
    for c in corruptions:
        sub = df_all_conditions[df_all_conditions["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_row["retained_acc_80_trust"] * 100] + (sub["retained_acc_80_trust"] * 100).tolist()
        ax4.plot(sevs, vals, marker="d", linewidth=2, color=colors.get(c, "purple"), label=c.capitalize())

    ax4.set_title("Selective Prediction Accuracy (80% Coverage) vs Severity", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax4.set_ylabel("Retained Accuracy @ 80% Coverage (%)", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "trust_model_robustness_by_corruption.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Phase 8 Cross-Shift Robustness Analysis.")
    parser.add_argument("--model", type=str, default="outputs/models/trust_model.pkl")
    parser.add_argument("--clean-signals", type=str, default="outputs/metrics/reliability_signals_clean.csv")
    parser.add_argument("--shifted-signals", type=str, default="outputs/metrics/reliability_signals_shifted.csv")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    model_p = Path(args.model).resolve()
    clean_p = Path(args.clean_signals).resolve()
    shifted_p = Path(args.shifted_signals).resolve()
    m_dir = Path(args.metrics_dir).resolve()
    p_dir = Path(args.plots_dir).resolve()

    if not model_p.exists() or not clean_p.exists() or not shifted_p.exists():
        raise FileNotFoundError("Required files missing!")

    pipeline = TrustModelPipeline.load(model_p)
    df_clean = pd.read_csv(clean_p)
    df_shifted = pd.read_csv(shifted_p)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — CROSS-SHIFT ROBUSTNESS ANALYSIS (PHASE 8)", flush=True)
    print("=" * 70, flush=True)

    df_all_conditions, df_family_summary = evaluate_cross_shift_robustness(pipeline, df_clean, df_shifted)

    df_all_conditions.to_csv(m_dir / "robustness_comparison.csv", index=False)

    print("CORRUPTION FAMILY ROBUSTNESS SUMMARY:")
    print("-" * 90)
    for idx, r in df_family_summary.iterrows():
        print(
            f"[{idx+1}] {r['corruption_family']:12s} | "
            f"Sev1 Acc: {r['accuracy_sev1']*100:5.2f}% -> Sev5 Acc: {r['accuracy_sev5']*100:5.2f}% (Drop: {r['accuracy_degradation']*100:5.2f}%) | "
            f"Trust Drop: {r['trust_degradation']:+.3f} | "
            f"Mean AUROC: {r['mean_failure_auroc']*100:5.2f}% | "
            f"Mean 80% Sel Acc: {r['mean_selective_acc_80']*100:5.2f}%"
        )
    print("-" * 90 + "\n")

    plot_corruption_family_robustness(df_all_conditions, p_dir)
    print(f"[OK] Robustness comparison saved to: {m_dir / 'robustness_comparison.csv'}")
    print(f"[OK] Robustness plot saved to: {p_dir / 'trust_model_robustness_by_corruption.png'}\n")


if __name__ == "__main__":
    main()
