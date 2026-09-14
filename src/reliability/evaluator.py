from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_single_signal(
    signal_values: np.ndarray,
    correctness: np.ndarray,
    signal_name: str,
    higher_is_reliable: bool = True,
) -> Dict[str, Any]:
    """Evaluate individual reliability signal's ability to distinguish correct from incorrect predictions.

    Args:
        signal_values: NumPy array of signal values [N].
        correctness: Binary indicator array [N] (1 = correct prediction, 0 = incorrect prediction).
        signal_name: Name identifier string for the signal.
        higher_is_reliable: If True, higher signal value indicates higher reliability (correctness=1).
                            If False, lower signal value indicates higher reliability.

    Returns:
        Dictionary containing signal_name, AUROC, AUPRC, correct_mean, incorrect_mean, direction, interpretation.
    """
    correct_mask = correctness == 1
    incorrect_mask = correctness == 0

    n_correct = int(np.sum(correct_mask))
    n_incorrect = int(np.sum(incorrect_mask))

    correct_mean = float(np.mean(signal_values[correct_mask])) if n_correct > 0 else 0.0
    incorrect_mean = float(np.mean(signal_values[incorrect_mask])) if n_incorrect > 0 else 0.0

    # Align score direction so higher score indicates higher probability of correctness (for AUROC calculation)
    if higher_is_reliable:
        auc_score = signal_values
        direction_str = "Higher = More Reliable"
    else:
        auc_score = -signal_values
        direction_str = "Lower = More Reliable"

    # Compute AUROC and AUPRC
    try:
        auroc = float(roc_auc_score(correctness, auc_score))
    except ValueError:
        auroc = 0.50

    try:
        auprc = float(average_precision_score(correctness, auc_score))
    except ValueError:
        auprc = 0.0

    if auroc >= 0.75:
        interp = "Strong separation capability"
    elif auroc >= 0.65:
        interp = "Moderate separation capability"
    elif auroc >= 0.55:
        interp = "Weak separation capability"
    else:
        interp = "Poor or non-discriminative signal"

    return {
        "signal_name": signal_name,
        "auroc": auroc,
        "auprc": auprc,
        "correct_mean": correct_mean,
        "incorrect_mean": incorrect_mean,
        "direction": direction_str,
        "interpretation": interp,
    }


def analyze_high_confidence_failures_signals(
    df_signals: pd.DataFrame, confidence_threshold: float = 0.90
) -> Dict[str, Any]:
    """Compare reliability signals between high-confidence correct and high-confidence incorrect predictions.

    Args:
        df_signals: DataFrame containing all reliability signals and 'confidence', 'correctness'.
        confidence_threshold: Threshold for high confidence (default 0.90).

    Returns:
        Dictionary comparing signal averages for high-confidence correct vs wrong predictions.
    """
    high_conf_df = df_signals[df_signals["confidence"] >= confidence_threshold]
    high_conf_correct = high_conf_df[high_conf_df["correctness"] == 1]
    high_conf_wrong = high_conf_df[high_conf_df["correctness"] == 0]

    signal_cols = [
        "confidence",
        "entropy",
        "feature_distance",
        "ood_score",
        "augmentation_consistency",
        "sharpness",
        "brightness",
        "contrast",
    ]

    res = {
        "confidence_threshold": confidence_threshold,
        "num_high_conf_correct": len(high_conf_correct),
        "num_high_conf_wrong": len(high_conf_wrong),
        "signals": {},
    }

    for col in signal_cols:
        if col in df_signals.columns:
            mean_correct = (
                float(high_conf_correct[col].mean()) if len(high_conf_correct) > 0 else 0.0
            )
            mean_wrong = (
                float(high_conf_wrong[col].mean()) if len(high_conf_wrong) > 0 else 0.0
            )
            res["signals"][col] = {
                "high_conf_correct_mean": mean_correct,
                "high_conf_wrong_mean": mean_wrong,
                "difference": mean_wrong - mean_correct,
            }

    return res
