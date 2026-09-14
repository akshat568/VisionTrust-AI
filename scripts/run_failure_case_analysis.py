import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchvision.datasets import CIFAR10

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cifar10 import CIFAR10_CLASSES
from src.reliability import (
    TrustModelPipeline,
    compute_bootstrap_auroc_ci,
    compute_bootstrap_auroc_delta_ci,
    evaluate_selective_prediction,
)
from src.utils.seed import set_seed


def save_failure_case_images(
    df_clean: pd.DataFrame,
    p_trust: np.ndarray,
    output_dir: Path,
    num_examples: int = 4,
):
    """Extract and save real CIFAR-10 test image visualization panels for representative failure modes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cifar10 = CIFAR10(root="data/raw", train=False, download=False)

    df_plot = df_clean.copy()
    df_plot["trust_prob"] = p_trust

    # Pattern A: High Confidence + Wrong Prediction
    pattern_a = df_plot[(df_plot["confidence"] >= 0.90) & (df_plot["correctness"] == 0)]
    # Pattern B: High Trust Probability + Wrong Prediction
    pattern_b = df_plot[(df_plot["trust_prob"] >= 0.85) & (df_plot["correctness"] == 0)]
    # Pattern C: Low Confidence + Correct Prediction
    pattern_c = df_plot[(df_plot["confidence"] <= 0.50) & (df_plot["correctness"] == 1)]

    patterns = [
        ("High_Confidence_Failures", pattern_a),
        ("High_Trust_Failures", pattern_b),
        ("Low_Confidence_Correct", pattern_c),
    ]

    for name, sub in patterns:
        if len(sub) == 0:
            continue
        n_show = min(num_examples, len(sub))
        sample_rows = sub.iloc[:n_show]

        fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
        if n_show == 1:
            axes = [axes]

        for i, (_, r) in enumerate(sample_rows.iterrows()):
            idx = int(r["sample_index"])
            img, label = raw_cifar10[idx]
            pred_cls = CIFAR10_CLASSES[int(r["predicted_label"])]
            true_cls = CIFAR10_CLASSES[int(r["true_label"])]

            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(
                f"Idx: {idx}\nTrue: {true_cls} | Pred: {pred_cls}\n"
                f"Conf: {r['confidence']:.2f} | Trust: {r['trust_prob']:.2f}",
                fontsize=9,
                fontweight="bold",
                color="darkred" if r["correctness"] == 0 else "darkgreen",
            )

        plt.suptitle(f"Real Test Failure Cases — {name.replace('_', ' ')}", fontsize=12, y=0.98)
        plt.tight_layout()
        plt.savefig(output_dir / f"{name.lower()}.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run Phase 8 Failure Case & Selective Prediction Analysis.")
    parser.add_argument("--model", type=str, default="outputs/models/trust_model.pkl")
    parser.add_argument("--val-signals", type=str, default="outputs/metrics/reliability_signals_val.csv")
    parser.add_argument("--clean-signals", type=str, default="outputs/metrics/reliability_signals_clean.csv")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    m_dir = Path(args.metrics_dir).resolve()
    p_dir = Path(args.plots_dir).resolve()
    fc_dir = p_dir / "failure_cases"

    pipeline = TrustModelPipeline.load(Path(args.model).resolve())
    df_val = pd.read_csv(Path(args.val_signals).resolve())
    df_clean = pd.read_csv(Path(args.clean_signals).resolve())

    set_seed(args.seed)

    print("=" * 70, flush=True)
    print("VISIONTRUST AI — FAILURE CASE & BOOTSTRAP ANALYSIS (PHASE 8)", flush=True)
    print("=" * 70, flush=True)

    p_trust_cal = pipeline.predict_proba(df_clean, model_type="calibrated")
    p_fail_cal = 1.0 - p_trust_cal
    p_fail_conf = 1.0 - df_clean["confidence"].values
    y_true = df_clean["correctness"].values.astype(np.int32)
    y_fail = (y_true == 0).astype(np.int32)

    # 1. Bootstrap 95% Confidence Interval Calculation
    print("[1] Computing Non-Parametric Bootstrap 95% CIs (1,000 resamples)...", flush=True)
    ci_rf = compute_bootstrap_auroc_ci(y_fail, p_fail_cal, n_bootstraps=1000, seed=args.seed)
    ci_conf = compute_bootstrap_auroc_ci(y_fail, p_fail_conf, n_bootstraps=1000, seed=args.seed)
    ci_delta = compute_bootstrap_auroc_delta_ci(y_fail, p_fail_cal, p_fail_conf, n_bootstraps=1000, seed=args.seed)

    print(f"  Trust Model AUROC:      {ci_rf['auroc']*100:5.2f}% (95% CI: [{ci_rf['ci_lower']*100:.2f}%, {ci_rf['ci_upper']*100:.2f}%])")
    print(f"  Raw Confidence AUROC:   {ci_conf['auroc']*100:5.2f}% (95% CI: [{ci_conf['ci_lower']*100:.2f}%, {ci_conf['ci_upper']*100:.2f}%])")
    print(f"  AUROC Delta (Gain):     +{ci_delta['delta_auroc']*100:4.2f}% (95% CI: [{ci_delta['ci_lower']*100:.2f}%, {ci_delta['ci_upper']*100:.2f}%], p-value ~ {ci_delta['p_value']:.4f})\n")

    # 2. Selective Prediction Ablation
    print("[2] Running Selective Prediction Ablation across models...", flush=True)
    # Fit Best Ablation Model (H. Without Image Quality: features 0..4)
    from src.reliability.ablation import ABLATION_CONFIGURATIONS
    h_features = ABLATION_CONFIGURATIONS["H. Without image quality"]
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    scaler_h = StandardScaler()
    X_val_h = scaler_h.fit_transform(df_val[h_features].values)
    X_test_h = scaler_h.transform(df_clean[h_features].values)

    rf_h = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=args.seed)
    rf_h.fit(X_val_h, df_val["correctness"].values)
    p_trust_best = np.clip(rf_h.predict_proba(X_test_h)[:, 1], 0.0, 1.0)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    sel_conf = evaluate_selective_prediction(y_true, df_clean["confidence"].values, coverages)
    sel_conf["evaluator"] = "Raw Confidence"

    sel_p7 = evaluate_selective_prediction(y_true, p_trust_cal, coverages)
    sel_p7["evaluator"] = "Phase 7 Trust Model (All Signals)"

    sel_best = evaluate_selective_prediction(y_true, p_trust_best, coverages)
    sel_best["evaluator"] = "Best Ablation Model (No Quality)"

    df_sel_ablation = pd.concat([sel_p7, sel_best, sel_conf], ignore_index=True)
    df_sel_ablation.to_csv(m_dir / "selective_ablation.csv", index=False)

    # Plot Selective Prediction Risk-Coverage Curve
    plt.figure(figsize=(8.5, 6))
    for ev_name, color, fmt in [
        ("Phase 7 Trust Model (All Signals)", "darkblue", "o-"),
        ("Best Ablation Model (No Quality)", "teal", "^-"),
        ("Raw Confidence", "forestgreen", "s--"),
    ]:
        sub = df_sel_ablation[df_sel_ablation["evaluator"] == ev_name].sort_values("coverage")
        plt.plot(sub["coverage"] * 100, sub["risk"] * 100, fmt, color=color, linewidth=2.2, label=ev_name)

    plt.title("Selective Prediction Risk-Coverage Comparison", fontsize=12, fontweight="bold")
    plt.xlabel("Coverage Level (%)", fontsize=10)
    plt.ylabel("Risk / Error Rate of Retained Predictions (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(p_dir / "selective_ablation_risk_coverage.png", dpi=300)
    plt.close()

    print(f"  [OK] Selective ablation saved to: {m_dir / 'selective_ablation.csv'}")
    print(f"  [OK] Selective plot saved to: {p_dir / 'selective_ablation_risk_coverage.png'}\n")

    # 3. High-Confidence Failure Analysis (Confidence >= 0.90)
    print("[3] High-Confidence Failures (>=0.90) Metric Breakdown:", flush=True)
    df_clean_h = df_clean.copy()
    df_clean_h["trust_prob"] = p_trust_cal

    high_correct = df_clean_h[(df_clean_h["confidence"] >= 0.90) & (df_clean_h["correctness"] == 1)]
    high_wrong = df_clean_h[(df_clean_h["confidence"] >= 0.90) & (df_clean_h["correctness"] == 0)]

    print(f"  Count -> Correct: {len(high_correct):,} | Wrong: {len(high_wrong):,}")
    print(f"  Raw Confidence -> Correct Mean: {high_correct['confidence'].mean():.4f} (Med: {high_correct['confidence'].median():.4f}) | Wrong Mean: {high_wrong['confidence'].mean():.4f} (Med: {high_wrong['confidence'].median():.4f})")
    print(f"  Trust Prob     -> Correct Mean: {high_correct['trust_prob'].mean():.4f} (Med: {high_correct['trust_prob'].median():.4f}) | Wrong Mean: {high_wrong['trust_prob'].mean():.4f} (Med: {high_wrong['trust_prob'].median():.4f})")
    print(f"  Entropy        -> Correct Mean: {high_correct['entropy'].mean():.4f} | Wrong Mean: {high_wrong['entropy'].mean():.4f}")
    print(f"  Aug Consist.   -> Correct Mean: {high_correct['augmentation_consistency'].mean():.4f} | Wrong Mean: {high_wrong['augmentation_consistency'].mean():.4f}\n")

    # 4. Save Real Failure Case Images
    save_failure_case_images(df_clean, p_trust_cal, fc_dir, num_examples=4)
    print(f"  [OK] Real failure case plots saved under: {fc_dir}\n")

    print("=" * 70)
    print("PHASE 8 FAILURE CASE & BOOTSTRAP ANALYSIS COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
