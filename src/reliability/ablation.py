from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

from src.reliability.calibration import compute_brier_score, compute_ece

# Define the 11 feature subset configurations for ablation
ABLATION_CONFIGURATIONS: Dict[str, List[str]] = {
    "A. All signals": [
        "confidence",
        "entropy",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "B. Confidence only": ["confidence"],
    "C. Without confidence": [
        "entropy",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "D. Without entropy": [
        "confidence",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "E. Without feature distance": [
        "confidence",
        "entropy",
        "ood_score",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "F. Without OOD score": [
        "confidence",
        "entropy",
        "feature_distance",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "G. Without augmentation consistency": [
        "confidence",
        "entropy",
        "feature_distance",
        "ood_score",
        "sharpness",
        "brightness",
        "contrast",
        "composite_quality",
    ],
    "H. Without image quality": [
        "confidence",
        "entropy",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
    ],
    "I. Confidence + entropy only": ["confidence", "entropy"],
    "J. Confidence + entropy + aug consistency": [
        "confidence",
        "entropy",
        "augmentation_consistency",
    ],
    "K. Confidence + entropy + feat distance + aug consistency": [
        "confidence",
        "entropy",
        "feature_distance",
        "augmentation_consistency",
    ],
}


def run_ablation_experiment(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_type: str = "rf",
    seed: int = 42,
) -> pd.DataFrame:
    """Run signal ablation study across all 11 feature configurations.

    Args:
        df_val: Validation signals DataFrame (5,000 samples).
        df_test: Clean test signals DataFrame (10,000 samples).
        model_type: 'rf' for Random Forest or 'logistic' for Logistic Regression.
        seed: Random seed.

    Returns:
        DataFrame summarizing failure detection AUROC, AUPRC, ECE, Brier, Precision, Recall, F1.
    """
    y_val = df_val["correctness"].values.astype(np.int32)
    y_test_correct = df_test["correctness"].values.astype(np.int32)
    y_test_failure = (y_test_correct == 0).astype(np.int32)

    records = []

    for name, features in ABLATION_CONFIGURATIONS.items():
        X_val = df_val[features].values
        X_test = df_test[features].values

        scaler = StandardScaler()
        X_val_scaled = scaler.fit_transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        if model_type == "rf":
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
        elif model_type == "logistic":
            model = LogisticRegression(random_state=seed, max_iter=1000)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        model.fit(X_val_scaled, y_val)
        p_correct = np.clip(model.predict_proba(X_test_scaled)[:, 1], 0.0, 1.0)
        p_failure = 1.0 - p_correct

        fpr, tpr, _ = roc_curve(y_test_failure, p_failure)
        failure_auroc = auc(fpr, tpr)

        prec, rec, _ = precision_recall_curve(y_test_failure, p_failure)
        failure_auprc = auc(rec, prec)

        ece = compute_ece(y_test_correct, p_correct)
        brier = compute_brier_score(y_test_correct, p_correct)

        bin_pred_failure = (p_failure >= 0.5).astype(int)
        precision_50 = precision_score(y_test_failure, bin_pred_failure, zero_division=0)
        recall_50 = recall_score(y_test_failure, bin_pred_failure, zero_division=0)
        f1_50 = f1_score(y_test_failure, bin_pred_failure, zero_division=0)

        records.append(
            {
                "config_name": name,
                "model_type": model_type,
                "num_features": len(features),
                "features_used": ", ".join(features),
                "failure_auroc": float(failure_auroc),
                "failure_auprc": float(failure_auprc),
                "ece": float(ece),
                "brier_score": float(brier),
                "precision_at_0.5": float(precision_50),
                "recall_at_0.5": float(recall_50),
                "f1_at_0.5": float(f1_50),
            }
        )

    df_res = pd.DataFrame(records).sort_values("failure_auroc", ascending=False).reset_index(drop=True)
    return df_res
