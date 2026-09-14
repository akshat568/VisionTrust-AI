from typing import Tuple, Dict
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def compute_bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute non-parametric bootstrap 95% confidence intervals for AUROC.

    Args:
        y_true: Ground truth binary targets (e.g. 1 for failure, 0 for correct).
        y_score: Predicted failure probabilities or risk scores.
        n_bootstraps: Number of bootstrap resamples (default 1000).
        ci_level: Confidence interval level (default 0.95).
        seed: Random seed for reproducible resampling.

    Returns:
        Dict with keys: ['mean_auroc', 'ci_lower', 'ci_upper', 'std_err'].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    n_samples = len(y_true)

    rng = np.random.RandomState(seed)
    bootstrapped_aurocs = []

    # Calculate point estimate
    fpr, tpr, _ = roc_curve(y_true, y_score)
    point_auroc = float(auc(fpr, tpr))

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, size=n_samples)
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip bootstrap samples with only one class

        fpr_b, tpr_b, _ = roc_curve(y_true[indices], y_score[indices])
        bootstrapped_aurocs.append(auc(fpr_b, tpr_b))

    bootstrapped_aurocs = np.array(bootstrapped_aurocs)
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(bootstrapped_aurocs, alpha * 100))
    ci_upper = float(np.percentile(bootstrapped_aurocs, (1.0 - alpha) * 100))
    std_err = float(np.std(bootstrapped_aurocs))

    return {
        "auroc": point_auroc,
        "mean_bootstrap_auroc": float(np.mean(bootstrapped_aurocs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
    }


def compute_bootstrap_auroc_delta_ci(
    y_true: np.ndarray,
    y_score_model: np.ndarray,
    y_score_baseline: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap 95% confidence interval for AUROC delta (Model - Baseline).

    Args:
        y_true: Binary failure targets.
        y_score_model: Trust model failure predictions.
        y_score_baseline: Baseline failure predictions (e.g. 1 - raw_confidence).
        n_bootstraps: Number of bootstrap iterations.
        ci_level: Confidence interval level.
        seed: Random seed.

    Returns:
        Dict with keys: ['delta_auroc', 'ci_lower', 'ci_upper', 'p_value_approx'].
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score_model = np.asarray(y_score_model, dtype=np.float32)
    y_score_baseline = np.asarray(y_score_baseline, dtype=np.float32)
    n_samples = len(y_true)

    rng = np.random.RandomState(seed)
    deltas = []

    fpr_m, tpr_m, _ = roc_curve(y_true, y_score_model)
    fpr_b, tpr_b, _ = roc_curve(y_true, y_score_baseline)
    point_delta = float(auc(fpr_m, tpr_m) - auc(fpr_b, tpr_b))

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n_samples, size=n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue

        fpr_m_i, tpr_m_i, _ = roc_curve(y_true[idx], y_score_model[idx])
        fpr_b_i, tpr_b_i, _ = roc_curve(y_true[idx], y_score_baseline[idx])
        deltas.append(auc(fpr_m_i, tpr_m_i) - auc(fpr_b_i, tpr_b_i))

    deltas = np.array(deltas)
    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(deltas, alpha * 100))
    ci_upper = float(np.percentile(deltas, (1.0 - alpha) * 100))

    # Empirical two-tailed p-value calculation
    p_val = float(np.mean(deltas <= 0.0)) * 2.0
    p_val = min(1.0, p_val)

    return {
        "delta_auroc": point_delta,
        "mean_delta": float(np.mean(deltas)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
    }
