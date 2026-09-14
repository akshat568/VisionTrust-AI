import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliability import (
    FEATURE_COLUMNS,
    TrustModelPipeline,
    compute_brier_score,
    compute_calibration_curve_data,
    compute_ece,
    evaluate_selective_prediction,
    evaluate_single_signal,
)


def generate_trust_model_plots(
    df_clean: pd.DataFrame,
    df_shift_metrics: pd.DataFrame,
    df_selective: pd.DataFrame,
    p_trust_cal: np.ndarray,
    p_trust_uncal: np.ndarray,
    p_trust_rf: np.ndarray,
    output_dir: Path,
):
    """Generate the 7 required Phase 7 diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = df_clean["correctness"].values  # 1 for correct, 0 for wrong
    y_failure = (y_true == 0).astype(int)  # 1 for failure, 0 for correct

    p_fail_cal = 1.0 - p_trust_cal
    p_fail_uncal = 1.0 - p_trust_uncal
    p_fail_rf = 1.0 - p_trust_rf
    p_fail_conf = 1.0 - df_clean["confidence"].values

    # 1. Trust Model ROC Curve vs Individual Baselines
    plt.figure(figsize=(8.5, 6.5))
    fpr_cal, tpr_cal, _ = roc_curve(y_failure, p_fail_cal)
    roc_auc_cal = auc(fpr_cal, tpr_cal)

    fpr_conf, tpr_conf, _ = roc_curve(y_failure, p_fail_conf)
    roc_auc_conf = auc(fpr_conf, tpr_conf)

    fpr_ent, tpr_ent, _ = roc_curve(y_failure, df_clean["entropy"].values)
    roc_auc_ent = auc(fpr_ent, tpr_ent)

    fpr_aug, tpr_aug, _ = roc_curve(y_failure, 1.0 - df_clean["augmentation_consistency"].values)
    roc_auc_aug = auc(fpr_aug, tpr_aug)

    fpr_dist, tpr_dist, _ = roc_curve(y_failure, df_clean["feature_distance"].values)
    roc_auc_dist = auc(fpr_dist, tpr_dist)

    plt.plot(fpr_cal, tpr_cal, color="darkblue", linewidth=2.5, label=f"Trust Model (Calibrated) (AUROC = {roc_auc_cal*100:.2f}%)")
    plt.plot(fpr_conf, tpr_conf, color="forestgreen", linewidth=2, linestyle="--", label=f"Raw Confidence (AUROC = {roc_auc_conf*100:.2f}%)")
    plt.plot(fpr_ent, tpr_ent, color="crimson", linewidth=1.5, linestyle=":", label=f"Prediction Entropy (AUROC = {roc_auc_ent*100:.2f}%)")
    plt.plot(fpr_aug, tpr_aug, color="purple", linewidth=1.5, linestyle="-.", label=f"Augmentation Consistency (AUROC = {roc_auc_aug*100:.2f}%)")
    plt.plot(fpr_dist, tpr_dist, color="orange", linewidth=1.5, linestyle=":", label=f"Feature Distance (AUROC = {roc_auc_dist*100:.2f}%)")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline (50.00%)")
    plt.title("Failure Detection ROC Curve — Trust Model vs Individual Baselines", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (FPR)", fontsize=10)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "trust_model_roc_curve.png", dpi=300)
    plt.close()

    # 2. Trust Model Precision-Recall Curve (Failure Detection)
    plt.figure(figsize=(8.5, 6.5))
    prec_cal, rec_cal, _ = precision_recall_curve(y_failure, p_fail_cal)
    pr_auc_cal = auc(rec_cal, prec_cal)

    prec_conf, rec_conf, _ = precision_recall_curve(y_failure, p_fail_conf)
    pr_auc_conf = auc(rec_conf, prec_conf)

    plt.plot(rec_cal, prec_cal, color="darkblue", linewidth=2.5, label=f"Trust Model (Calibrated) (AUPRC = {pr_auc_cal:.4f})")
    plt.plot(rec_conf, rec_conf, color="forestgreen", linewidth=2, linestyle="--", label=f"Raw Confidence (AUPRC = {pr_auc_conf:.4f})")

    plt.axhline(y=np.mean(y_failure), color="gray", linestyle=":", label=f"Base Failure Rate ({np.mean(y_failure)*100:.2f}%)")
    plt.title("Failure Detection Precision-Recall Curve — Clean Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Recall (Fraction of Failures Caught)", fontsize=10)
    plt.ylabel("Precision (Fraction of Flagged Predictions Wrong)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "trust_model_pr_curve.png", dpi=300)
    plt.close()

    # 3. Reliability Diagram: Raw Confidence vs Trust Model
    plt.figure(figsize=(8, 6))
    raw_cal = compute_calibration_curve_data(y_true, df_clean["confidence"].values, n_bins=10)
    uncal_cal = compute_calibration_curve_data(y_true, p_trust_uncal, n_bins=10)
    cal_cal = compute_calibration_curve_data(y_true, p_trust_cal, n_bins=10)

    ece_raw = compute_ece(y_true, df_clean["confidence"].values)
    ece_uncal = compute_ece(y_true, p_trust_uncal)
    ece_cal = compute_ece(y_true, p_trust_cal)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Ideal Calibration")
    plt.plot(raw_cal["bin_confidences"], raw_cal["bin_accuracies"], "s-", color="forestgreen", label=f"Raw Confidence (ECE = {ece_raw*100:.2f}%)")
    plt.plot(uncal_cal["bin_confidences"], uncal_cal["bin_accuracies"], "o-", color="crimson", label=f"Uncalibrated Trust Model (ECE = {ece_uncal*100:.2f}%)")
    plt.plot(cal_cal["bin_confidences"], cal_cal["bin_accuracies"], "^-", color="darkblue", label=f"Calibrated Trust Model (ECE = {ece_cal*100:.2f}%)")

    plt.title("Reliability Calibration Diagram — Clean CIFAR-10 Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Confidence / Trust Probability", fontsize=10)
    plt.ylabel("Empirical Accuracy", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "reliability_diagram_trust_model.png", dpi=300)
    plt.close()

    # 4. Risk-Coverage Curve: Raw Confidence vs Trust Model
    plt.figure(figsize=(8.5, 6))
    df_sel_trust = df_selective[df_selective["evaluator"] == "Trust Model (Calibrated)"].sort_values("coverage")
    df_sel_conf = df_selective[df_selective["evaluator"] == "Raw Confidence"].sort_values("coverage")

    plt.plot(df_sel_trust["coverage"] * 100, df_sel_trust["risk"] * 100, "o-", color="darkblue", linewidth=2.5, label="Trust Model (Calibrated)")
    plt.plot(df_sel_conf["coverage"] * 100, df_sel_conf["risk"] * 100, "s--", color="forestgreen", linewidth=2, label="Raw Softmax Confidence")

    plt.title("Selective Prediction Risk-Coverage Curve — Clean Test Set", fontsize=12, fontweight="bold")
    plt.xlabel("Coverage Level (%)", fontsize=10)
    plt.ylabel("Risk / Error Rate of Retained Predictions (%)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "risk_coverage_curve.png", dpi=300)
    plt.close()

    # 5. Trust Probability Distribution: Correct vs Incorrect
    plt.figure(figsize=(8.5, 5.5))
    df_clean_plot = df_clean.copy()
    df_clean_plot["trust_probability"] = p_trust_cal

    correct_trust = df_clean_plot[df_clean_plot["correctness"] == 1]["trust_probability"]
    incorrect_trust = df_clean_plot[df_clean_plot["correctness"] == 0]["trust_probability"]

    sns.kdeplot(correct_trust, color="forestgreen", fill=True, alpha=0.35, linewidth=2, label=f"Correct Predictions (Mean: {correct_trust.mean():.4f})")
    sns.kdeplot(incorrect_trust, color="crimson", fill=True, alpha=0.35, linewidth=2, label=f"Incorrect Predictions (Mean: {incorrect_trust.mean():.4f})")

    plt.title("Trust Probability Density Distribution — Correct vs Incorrect Predictions", fontsize=12, fontweight="bold")
    plt.xlabel("Calibrated Trust Probability P(correct)", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plt.savefig(output_dir / "trust_probability_distribution.png", dpi=300)
    plt.close()

    # 6. Trust Model Performance Across Corruption Severity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    corruptions = [c for c in df_shift_metrics["shift"].unique() if c != "clean"]
    colors = {"blur": "#1f77b4", "noise": "#ff7f0e", "brightness": "#2ca02c", "contrast": "#d62728", "rotation": "#9467bd"}

    # Ax1: Failure AUROC vs Severity
    clean_auroc = roc_auc_cal * 100
    ax1.axhline(clean_auroc, color="black", linestyle="--", label=f"Clean ({clean_auroc:.2f}%)")
    for c in corruptions:
        sub = df_shift_metrics[df_shift_metrics["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_auroc] + (sub["failure_auroc"] * 100).tolist()
        ax1.plot(sevs, vals, marker="o", linewidth=2, color=colors.get(c, "blue"), label=c.capitalize())

    ax1.set_title("Failure Detection AUROC vs Corruption Severity", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax1.set_ylabel("Failure AUROC (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    # Ax2: Mean Trust Probability vs Severity
    clean_mean_trust = p_trust_cal.mean()
    ax2.axhline(clean_mean_trust, color="black", linestyle="--", label=f"Clean ({clean_mean_trust:.3f})")
    for c in corruptions:
        sub = df_shift_metrics[df_shift_metrics["shift"] == c].sort_values("severity")
        sevs = [0] + sub["severity"].tolist()
        vals = [clean_mean_trust] + sub["mean_trust_prob"].tolist()
        ax2.plot(sevs, vals, marker="s", linewidth=2, color=colors.get(c, "orange"), label=c.capitalize())

    ax2.set_title("Mean Trust Probability vs Corruption Severity", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Severity Level (0 = Clean, 1-5 = Shifted)", fontsize=10)
    ax2.set_ylabel("Mean Trust Probability P(correct)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "trust_model_vs_severity.png", dpi=300)
    plt.close()

    # 7. High-Confidence Correct vs Incorrect Comparison
    high_conf_mask = df_clean["confidence"] >= 0.90
    df_high_conf = df_clean[high_conf_mask].copy()
    df_high_conf["trust_prob"] = p_trust_cal[high_conf_mask]

    high_correct = df_high_conf[df_high_conf["correctness"] == 1]
    high_wrong = df_high_conf[df_high_conf["correctness"] == 0]

    metrics_comp = [
        ("Raw Confidence", high_correct["confidence"].mean(), high_wrong["confidence"].mean()),
        ("Trust Probability", high_correct["trust_prob"].mean(), high_wrong["trust_prob"].mean()),
        ("Entropy (Scaled)", high_correct["entropy"].mean(), high_wrong["entropy"].mean()),
        ("Aug Consist.", high_correct["augmentation_consistency"].mean(), high_wrong["augmentation_consistency"].mean()),
    ]

    plt.figure(figsize=(9, 5.5))
    x = np.arange(len(metrics_comp))
    width = 0.35

    correct_vals = [m[1] for m in metrics_comp]
    wrong_vals = [m[2] for m in metrics_comp]

    plt.bar(x - width/2, correct_vals, width, label=f"High-Conf Correct (n={len(high_correct):,})", color="forestgreen", alpha=0.85)
    plt.bar(x + width/2, wrong_vals, width, label=f"High-Conf Wrong (n={len(high_wrong):,})", color="crimson", alpha=0.85)

    plt.xticks(x, [m[0] for m in metrics_comp], fontsize=10)
    plt.title("High-Confidence Predictions (>=0.90): Correct vs Wrong Signal Comparison", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Value", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(loc="upper right", fontsize=9.5)

    for i in range(len(metrics_comp)):
        plt.text(i - width/2, correct_vals[i] + 0.02, f"{correct_vals[i]:.3f}", ha="center", fontsize=9, fontweight="bold")
        plt.text(i + width/2, wrong_vals[i] + 0.02, f"{wrong_vals[i]:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.ylim(0.0, 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "high_confidence_failures_trust_comparison.png", dpi=300)
    plt.close()


def run_trust_model_evaluation(
    model_path: str = "outputs/models/trust_model.pkl",
    clean_signals_path: str = "outputs/metrics/reliability_signals_clean.csv",
    shifted_signals_path: str = "outputs/metrics/reliability_signals_shifted.csv",
    metrics_dir: str = "outputs/metrics",
    plots_dir: str = "outputs/plots",
):
    """Execute complete Phase 7 evaluation pipeline."""
    model_p = Path(model_path).resolve()
    clean_p = Path(clean_signals_path).resolve()
    shifted_p = Path(shifted_signals_path).resolve()
    m_dir = Path(metrics_dir).resolve()
    p_dir = Path(plots_dir).resolve()

    if not model_p.exists():
        raise FileNotFoundError(f"Trust Model missing at {model_p}. Run `scripts/train_trust_model.py` first.")
    if not clean_p.exists() or not shifted_p.exists():
        raise FileNotFoundError(f"Signal metrics missing at {clean_p} or {shifted_p}.")

    pipeline = TrustModelPipeline.load(model_p)
    df_clean = pd.read_csv(clean_p)
    df_shifted = pd.read_csv(shifted_p)

    print("=" * 70)
    print("VISIONTRUST AI — PHASE 7 TRUST MODEL EVALUATION")
    print("=" * 70)

    # 1. Compute Trust Probabilities on Clean Test Set
    p_trust_cal = pipeline.predict_proba(df_clean, model_type="calibrated")
    p_trust_uncal = pipeline.predict_proba(df_clean, model_type="logistic")
    p_trust_rf = pipeline.predict_proba(df_clean, model_type="rf")

    p_fail_cal = 1.0 - p_trust_cal
    p_fail_uncal = 1.0 - p_trust_uncal
    p_fail_rf = 1.0 - p_trust_rf
    p_fail_conf = 1.0 - df_clean["confidence"].values

    y_true = df_clean["correctness"].values
    y_failure = (y_true == 0).astype(int)

    # Save Trust Model Predictions CSV
    df_preds = df_clean[["sample_index", "true_label", "predicted_label", "correctness", "confidence"]].copy()
    df_preds["trust_prob_calibrated"] = p_trust_cal
    df_preds["trust_prob_uncalibrated"] = p_trust_uncal
    df_preds["trust_prob_rf"] = p_trust_rf
    df_preds["failure_prob"] = p_fail_cal
    df_preds.to_csv(m_dir / "trust_model_predictions.csv", index=False)
    print(f"[OK] Clean Test predictions saved to: {m_dir / 'trust_model_predictions.csv'}\n")

    # 2. Failure Detection Metrics (Positive = Incorrect Prediction)
    models_to_eval = [
        ("Trust Model (Calibrated)", p_fail_cal),
        ("Trust Model (Uncalibrated)", p_fail_uncal),
        ("Trust Model (Random Forest)", p_fail_rf),
        ("Raw Confidence", p_fail_conf),
        ("Prediction Entropy", df_clean["entropy"].values),
        ("Augmentation Consistency", 1.0 - df_clean["augmentation_consistency"].values),
        ("Feature Distance", df_clean["feature_distance"].values),
        ("OOD Energy Score", -df_clean["ood_score"].values),
    ]

    metrics_records = []
    for m_name, p_fail in models_to_eval:
        fpr, tpr, _ = roc_curve(y_failure, p_fail)
        roc_auc = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(y_failure, p_fail)
        pr_auc = auc(rec, prec)

        # Binary failure prediction at 0.5 threshold
        bin_pred = (p_fail >= 0.5).astype(int)
        p_val = precision_score(y_failure, bin_pred, zero_division=0)
        r_val = recall_score(y_failure, bin_pred, zero_division=0)
        f1_val = f1_score(y_failure, bin_pred, zero_division=0)
        cm = confusion_matrix(y_failure, bin_pred)

        metrics_records.append(
            {
                "evaluator": m_name,
                "failure_auroc": float(roc_auc),
                "failure_auprc": float(pr_auc),
                "precision_at_0.5": float(p_val),
                "recall_at_0.5": float(r_val),
                "f1_at_0.5": float(f1_val),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

    df_metrics = pd.DataFrame(metrics_records).sort_values("failure_auroc", ascending=False).reset_index(drop=True)
    df_metrics.to_csv(m_dir / "trust_model_metrics.csv", index=False)

    print("FAILURE DETECTION METRICS (POSITIVE CLASS = INCORRECT PREDICTION):")
    print("-" * 85)
    for idx, r in df_metrics.iterrows():
        print(
            f"[{idx+1}] {r['evaluator']:30s} | AUROC: {r['failure_auroc']*100:5.2f}% | "
            f"AUPRC: {r['failure_auprc']:.4f} | Prec: {r['precision_at_0.5']:.4f} | "
            f"Rec: {r['recall_at_0.5']:.4f} | F1: {r['f1_at_0.5']:.4f}"
        )
    print("-" * 85 + "\n")

    # 3. Calibration Metrics (ECE & Brier Score)
    cal_records = [
        {"evaluator": "Raw Confidence", "ece": compute_ece(y_true, df_clean["confidence"].values), "brier_score": compute_brier_score(y_true, df_clean["confidence"].values)},
        {"evaluator": "Trust Model (Uncalibrated)", "ece": compute_ece(y_true, p_trust_uncal), "brier_score": compute_brier_score(y_true, p_trust_uncal)},
        {"evaluator": "Trust Model (Calibrated)", "ece": compute_ece(y_true, p_trust_cal), "brier_score": compute_brier_score(y_true, p_trust_cal)},
        {"evaluator": "Trust Model (Random Forest)", "ece": compute_ece(y_true, p_trust_rf), "brier_score": compute_brier_score(y_true, p_trust_rf)},
    ]

    df_cal = pd.DataFrame(cal_records)
    df_cal.to_csv(m_dir / "calibration_metrics.csv", index=False)

    print("PROBABILITY CALIBRATION METRICS (CLEAN TEST SET):")
    print("-" * 65)
    for _, r in df_cal.iterrows():
        print(
            f"  {r['evaluator']:30s} -> ECE: {r['ece']*100:5.2f}% | "
            f"Brier Score: {r['brier_score']:.4f}"
        )
    print("-" * 65 + "\n")

    # 4. Selective Prediction Analysis
    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    sel_trust = evaluate_selective_prediction(y_true, p_trust_cal, coverages)
    sel_trust["evaluator"] = "Trust Model (Calibrated)"

    sel_conf = evaluate_selective_prediction(y_true, df_clean["confidence"].values, coverages)
    sel_conf["evaluator"] = "Raw Confidence"

    df_sel_all = pd.concat([sel_trust, sel_conf], ignore_index=True)
    df_sel_all.to_csv(m_dir / "selective_prediction.csv", index=False)

    print("SELECTIVE PREDICTION ACCURACY & RISK:")
    print("-" * 75)
    print(f"Coverage | Trust Model Acc (Risk) | Raw Conf Acc (Risk) | Rejections")
    print("-" * 75)
    for cov in coverages:
        row_t = sel_trust[sel_trust["target_coverage"] == cov].iloc[0]
        row_c = sel_conf[sel_conf["target_coverage"] == cov].iloc[0]
        print(
            f"  {cov*100:5.0f}%  |  "
            f"{row_t['accuracy']*100:5.2f}% ({row_t['risk']*100:5.2f}%)   |  "
            f"{row_c['accuracy']*100:5.2f}% ({row_c['risk']*100:5.2f}%)  |  "
            f"{row_t['rejected_count']:,} samples"
        )
    print("-" * 75 + "\n")

    # 5. High-Confidence Failure Analysis (Confidence >= 0.90 and Incorrect)
    high_conf_mask = df_clean["confidence"] >= 0.90
    df_high_conf = df_clean[high_conf_mask].copy()
    df_high_conf["trust_prob"] = p_trust_cal[high_conf_mask]

    high_correct = df_high_conf[df_high_conf["correctness"] == 1]
    high_wrong = df_high_conf[df_high_conf["correctness"] == 0]

    print("HIGH-CONFIDENCE FAILURES (>=0.90) TRUST MODEL COMPARISON:")
    print("-" * 75)
    print(f"Total High-Confidence Correct: {len(high_correct):,} | Wrong: {len(high_wrong):,}")
    print(f"  Raw Softmax Confidence -> Correct: {high_correct['confidence'].mean():.4f} | Wrong: {high_wrong['confidence'].mean():.4f}")
    print(f"  Trust Model Probability -> Correct: {high_correct['trust_prob'].mean():.4f} | Wrong: {high_wrong['trust_prob'].mean():.4f} (Diff: {high_wrong['trust_prob'].mean() - high_correct['trust_prob'].mean():+.4f})")
    print("-" * 75 + "\n")

    # 6. Evaluate Trust Model Across 25 Shifted Conditions
    shift_records = []
    # Clean baseline
    fpr_c, tpr_c, _ = roc_curve(y_failure, p_fail_cal)
    pr_c, rec_c, _ = precision_recall_curve(y_failure, p_fail_cal)
    shift_records.append(
        {
            "shift": "clean",
            "severity": 0,
            "vision_accuracy": float(np.mean(y_true)),
            "mean_trust_prob": float(np.mean(p_trust_cal)),
            "failure_auroc": float(auc(fpr_c, tpr_c)),
            "failure_auprc": float(auc(rec_c, pr_c)),
            "ece": float(compute_ece(y_true, p_trust_cal)),
            "brier_score": float(compute_brier_score(y_true, p_trust_cal)),
        }
    )

    grouped_shifts = df_shifted.groupby(["shift", "severity"])
    for (shift_name, sev), group in grouped_shifts:
        if shift_name == "clean":
            continue

        p_s_trust = pipeline.predict_proba(group, model_type="calibrated")
        p_s_fail = 1.0 - p_s_trust

        s_y_true = group["correctness"].values
        s_y_fail = (s_y_true == 0).astype(int)

        fpr_s, tpr_s, _ = roc_curve(s_y_fail, p_s_fail)
        pr_s, rec_s, _ = precision_recall_curve(s_y_fail, p_s_fail)

        shift_records.append(
            {
                "shift": shift_name,
                "severity": int(sev),
                "vision_accuracy": float(np.mean(s_y_true)),
                "mean_trust_prob": float(np.mean(p_s_trust)),
                "failure_auroc": float(auc(fpr_s, tpr_s)),
                "failure_auprc": float(auc(rec_s, pr_s)),
                "ece": float(compute_ece(s_y_true, p_s_trust)),
                "brier_score": float(compute_brier_score(s_y_true, p_s_trust)),
            }
        )

    df_shift_metrics = pd.DataFrame(shift_records)
    df_shift_metrics.to_csv(m_dir / "shift_trust_metrics.csv", index=False)
    print(f"[OK] Shifted conditions Trust Model evaluation saved to: {m_dir / 'shift_trust_metrics.csv'}\n")

    # 7. Generate All 7 Diagnostic Visualizations
    generate_trust_model_plots(
        df_clean=df_clean,
        df_shift_metrics=df_shift_metrics,
        df_selective=df_sel_all,
        p_trust_cal=p_trust_cal,
        p_trust_uncal=p_trust_uncal,
        p_trust_rf=p_trust_rf,
        output_dir=p_dir,
    )
    print(f"[OK] All 7 diagnostic visualizations saved under: {p_dir}\n")

    print("=" * 70)
    print("PHASE 7 TRUST MODEL EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 7 Trust Model.")
    parser.add_argument("--model", type=str, default="outputs/models/trust_model.pkl")
    parser.add_argument("--clean-signals", type=str, default="outputs/metrics/reliability_signals_clean.csv")
    parser.add_argument("--shifted-signals", type=str, default="outputs/metrics/reliability_signals_shifted.csv")
    parser.add_argument("--metrics-dir", type=str, default="outputs/metrics")
    parser.add_argument("--plots-dir", type=str, default="outputs/plots")
    args = parser.parse_args()

    run_trust_model_evaluation(
        model_path=args.model,
        clean_signals_path=args.clean_signals,
        shifted_signals_path=args.shifted_signals,
        metrics_dir=args.metrics_dir,
        plots_dir=args.plots_dir,
    )


if __name__ == "__main__":
    main()
