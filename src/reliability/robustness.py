from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from src.reliability.calibration import compute_brier_score, compute_ece
from src.reliability.selective_prediction import evaluate_selective_prediction
from src.reliability.trust_model import TrustModelPipeline


def evaluate_cross_shift_robustness(
    pipeline: TrustModelPipeline,
    df_clean: pd.DataFrame,
    df_shifted: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Phase 7 Trust Model across clean test data and 25 shifted test conditions.

    Args:
        pipeline: Pre-trained Phase 7 TrustModelPipeline (fitted strictly on validation set).
        df_clean: Clean test signals DataFrame.
        df_shifted: Shifted test signals DataFrame (250,000 samples).

    Returns:
        Tuple of (df_all_conditions, df_family_summary).
    """
    records = []

    # Combine clean and shifted dataframes (excluding clean from df_shifted to prevent duplication)
    df_shifted_only = df_shifted[df_shifted["shift"] != "clean"]
    all_dfs = [df_clean] + [group for _, group in df_shifted_only.groupby(["shift", "severity"])]

    for df_cond in all_dfs:
        shift_name = df_cond["shift"].iloc[0]
        severity = df_cond["severity"].iloc[0]

        p_trust = pipeline.predict_proba(df_cond, model_type="calibrated")
        p_failure = 1.0 - p_trust
        raw_conf = df_cond["confidence"].values

        y_true = df_cond["correctness"].values.astype(np.int32)
        y_failure = (y_true == 0).astype(np.int32)

        # Classification accuracy
        vision_acc = float(np.mean(y_true))

        # Failure detection AUROC and AUPRC
        fpr, tpr, _ = roc_curve(y_failure, p_failure)
        failure_auroc = float(auc(fpr, tpr))

        prec, rec, _ = precision_recall_curve(y_failure, p_failure)
        failure_auprc = float(auc(rec, prec))

        # Calibration metrics
        ece = float(compute_ece(y_true, p_trust))
        brier = float(compute_brier_score(y_true, p_trust))

        # Selective prediction at 80% coverage
        df_sel_trust = evaluate_selective_prediction(y_true, p_trust, coverages=[0.8])
        df_sel_conf = evaluate_selective_prediction(y_true, raw_conf, coverages=[0.8])

        sel_acc_trust = float(df_sel_trust["accuracy"].iloc[0])
        sel_risk_trust = float(df_sel_trust["risk"].iloc[0])
        sel_acc_conf = float(df_sel_conf["accuracy"].iloc[0])
        sel_risk_conf = float(df_sel_conf["risk"].iloc[0])

        records.append(
            {
                "shift": shift_name,
                "severity": int(severity),
                "vision_accuracy": vision_acc,
                "mean_trust_prob": float(np.mean(p_trust)),
                "raw_confidence_mean": float(np.mean(raw_conf)),
                "failure_auroc": failure_auroc,
                "failure_auprc": failure_auprc,
                "ece": ece,
                "brier_score": brier,
                "retained_acc_80_trust": sel_acc_trust,
                "retained_risk_80_trust": sel_risk_trust,
                "retained_acc_80_conf": sel_acc_conf,
                "retained_risk_80_conf": sel_risk_conf,
                "selective_gain_80": sel_acc_trust - sel_acc_conf,
            }
        )

    df_all = pd.DataFrame(records)

    # Aggregate by Corruption Family (severities 1 to 5)
    family_records = []
    corruptions = [c for c in df_all["shift"].unique() if c != "clean"]

    for c in corruptions:
        sub = df_all[df_all["shift"] == c]
        family_records.append(
            {
                "corruption_family": c.capitalize(),
                "accuracy_sev1": float(sub[sub["severity"] == 1]["vision_accuracy"].iloc[0]),
                "accuracy_sev5": float(sub[sub["severity"] == 5]["vision_accuracy"].iloc[0]),
                "accuracy_degradation": float(
                    sub[sub["severity"] == 1]["vision_accuracy"].iloc[0]
                    - sub[sub["severity"] == 5]["vision_accuracy"].iloc[0]
                ),
                "trust_prob_sev1": float(sub[sub["severity"] == 1]["mean_trust_prob"].iloc[0]),
                "trust_prob_sev5": float(sub[sub["severity"] == 5]["mean_trust_prob"].iloc[0]),
                "trust_degradation": float(
                    sub[sub["severity"] == 1]["mean_trust_prob"].iloc[0]
                    - sub[sub["severity"] == 5]["mean_trust_prob"].iloc[0]
                ),
                "mean_failure_auroc": float(sub["failure_auroc"].mean()),
                "mean_failure_auprc": float(sub["failure_auprc"].mean()),
                "mean_selective_acc_80": float(sub["retained_acc_80_trust"].mean()),
                "mean_selective_risk_80": float(sub["retained_risk_80_trust"].mean()),
            }
        )

    df_family = pd.DataFrame(family_records)
    return df_all, df_family
