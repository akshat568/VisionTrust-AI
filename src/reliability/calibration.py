from typing import Tuple, Dict
import numpy as np


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width confidence bins.

    Args:
        y_true: Binary ground truth targets (1 for correct/positive, 0 for incorrect/negative).
        y_prob: Predicted confidence/probability values in range [0, 1].
        n_bins: Number of confidence bins (default 10).

    Returns:
        Scalar ECE value in range [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probability predictions).

    Args:
        y_true: Binary ground truth targets.
        y_prob: Predicted probability values in range [0, 1].

    Returns:
        Scalar Brier score in range [0, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_calibration_curve_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Compute empirical accuracy, mean confidence, and sample counts for reliability diagrams.

    Args:
        y_true: Binary ground truth targets.
        y_prob: Predicted probability values in range [0, 1].
        n_bins: Number of confidence bins (default 10).

    Returns:
        Dict containing bin_accuracies, bin_confidences, and bin_counts arrays.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_accs.append(float(np.mean(y_true[mask])))
            bin_confs.append(float(np.mean(y_prob[mask])))
        else:
            bin_accs.append(np.nan)
            bin_confs.append(float((low + high) / 2.0))

    return {
        "bin_accuracies": np.array(bin_accs, dtype=np.float32),
        "bin_confidences": np.array(bin_confs, dtype=np.float32),
        "bin_counts": np.array(bin_counts, dtype=np.int32),
    }
