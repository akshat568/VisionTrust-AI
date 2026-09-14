from typing import List, Dict
import numpy as np
import pandas as pd


def evaluate_selective_prediction(
    correctness: np.ndarray,
    reliability_scores: np.ndarray,
    coverages: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
) -> pd.DataFrame:
    """Evaluate selective prediction accuracy and risk across target coverage levels.

    Args:
        correctness: Ground truth correctness array (1 for correct vision prediction, 0 for wrong).
        reliability_scores: Estimated reliability scores (higher means more trustworthy).
        coverages: List of coverage fractions to evaluate.

    Returns:
        DataFrame containing columns: [coverage, target_coverage, retained_count, rejected_count, accuracy, risk].
    """
    correctness = np.asarray(correctness, dtype=np.int32)
    reliability_scores = np.asarray(reliability_scores, dtype=np.float32)

    total_samples = len(correctness)
    if total_samples == 0:
        return pd.DataFrame()

    # Sort samples by reliability score in descending order (highest reliability first)
    sort_indices = np.argsort(-reliability_scores)
    sorted_correctness = correctness[sort_indices]

    records = []
    for cov_target in coverages:
        # Determine number of retained samples
        k = max(1, int(np.round(cov_target * total_samples)))
        retained_correctness = sorted_correctness[:k]

        retained_count = k
        rejected_count = total_samples - k
        actual_coverage = retained_count / total_samples
        accuracy = float(np.mean(retained_correctness))
        risk = 1.0 - accuracy

        records.append(
            {
                "target_coverage": float(cov_target),
                "coverage": float(actual_coverage),
                "retained_count": int(retained_count),
                "rejected_count": int(rejected_count),
                "accuracy": float(accuracy),
                "risk": float(risk),
            }
        )

    return pd.DataFrame(records)
