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

from src.reliability.ablation import ABLATION_CONFIGURATIONS, run_ablation_experiment
from src.reliability.bootstrap import compute_bootstrap_auroc_ci, compute_bootstrap_auroc_delta_ci
from src.utils.seed import set_seed


def plot_ablation_results(
    df_ablation: pd.DataFrame,
    output_dir: Path,
):
    """Generate signal ablation bar charts for AUROC and AUPRC."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Failure AUROC Bar Chart
    plt.figure(figsize=(10, 6.5))
    df_sorted_auroc = df_ablation.sort_values("failure_auroc", ascending=True)

    bars_auroc = plt.barh(
        df_sorted_auroc["config_name"],
        df_sorted_auroc["failure_auroc"] * 100,
        color="steelblue",
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(x=85.61, color="forestgreen", linestyle="--", linewidth=1.5, label="Raw Confidence Baseline (85.61%)")
    plt.axvline(x=86.69, color="darkblue", linestyle=":", linewidth=1.5, label="Phase 7 All Signals RF (86.69%)")

    plt.title("Signal Ablation Study — Failure Detection AUROC on Clean Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Failure Detection AUROC (%)", fontsize=10)
    plt.xlim(80.0, 88.0)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars_auroc:
        w = bar.get_width()
        plt.text(
            w + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_ablation_auroc.png", dpi=300)
    plt.close()

    # 2. Failure AUPRC Bar Chart
    plt.figure(figsize=(10, 6.5))
    df_sorted_auprc = df_ablation.sort_values("failure_auprc", ascending=True)

    bars_auprc = plt.barh(
        df_sorted_auprc["config_name"],
        df_sorted_auprc["failure_auprc"],
        color="teal",
        edgecolor="black",
        alpha=0.85,
    )
    plt.axvline(x=0.5351, color="forestgreen", linestyle="--", linewidth=1.5, label="Raw Confidence Baseline (0.5351)")
    plt.axvline(x=0.5708, color="darkblue", linestyle=":", linewidth=1.5, label="Phase 7 All Signals RF (0.5708)")

    plt.title("Signal Ablation Study — Failure Detection AUPRC on Clean Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Failure Detection AUPRC Score", fontsize=10)
    plt.xlim(0.48, 0.59)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")

    for bar in bars_auprc:
        w = bar.get_width()
        plt.text(
            w + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "signal_ablation_auprc.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Phase 8 Signal Ablation Study.")
    parser.add_argument("--val-signals", type=str, default="outputs/metrics/reliability_signals_val.csv")
    parser.add_argument("--clean-signals", type=str, default="outputs/metrics/reliability_signals_clean.csv")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    val_p = Path(args.val_signals).resolve()
    clean_p = Path(args.clean_signals).resolve()
    m_dir = Path(args.metrics_dir).resolve()
    p_dir = Path(args.plots_dir).resolve()

    if not val_p.exists() or not clean_p.exists():
        raise FileNotFoundError(f"Signal files missing at {val_p} or {clean_p}")

    set_seed(args.seed)
    df_val = pd.read_csv(val_p)
    df_clean = pd.read_csv(clean_p)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — SIGNAL ABLATION STUDY (PHASE 8)", flush=True)
    print("=" * 70, flush=True)
    print(f"Validation signals: {len(df_val):,} samples (Training data)", flush=True)
    print(f"Clean test signals: {len(df_clean):,} samples (Untouched test evaluation)", flush=True)
    print(f"Total ablation configurations: {len(ABLATION_CONFIGURATIONS)}\n", flush=True)

    # Run Ablation Study with Random Forest
    df_ablation_rf = run_ablation_experiment(df_val, df_clean, model_type="rf", seed=args.seed)
    df_ablation_rf.to_csv(m_dir / "signal_ablation_results.csv", index=False)

    print("SIGNAL ABLATION BENCHMARK RESULTS (RANDOM FOREST):")
    print("-" * 90)
    for idx, r in df_ablation_rf.iterrows():
        print(
            f"[{idx+1:02d}] {r['config_name']:55s} | AUROC: {r['failure_auroc']*100:5.2f}% | "
            f"AUPRC: {r['failure_auprc']:.4f} | ECE: {r['ece']*100:4.2f}% | Brier: {r['brier_score']:.4f}"
        )
    print("-" * 90 + "\n")

    # Generate Ablation Plots
    plot_ablation_results(df_ablation_rf, p_dir)
    print(f"[OK] Ablation metrics saved to: {m_dir / 'signal_ablation_results.csv'}")
    print(f"[OK] Ablation plots saved to: {p_dir / 'signal_ablation_auroc.png'} and {p_dir / 'signal_ablation_auprc.png'}\n")


if __name__ == "__main__":
    main()
