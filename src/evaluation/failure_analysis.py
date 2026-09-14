from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data import CIFAR10_CLASSES


def analyze_clean_failures(df_clean: pd.DataFrame) -> Dict[str, Any]:
    """Analyze global confidence and correctness metrics on clean test set predictions.

    Args:
        df_clean: DataFrame containing clean test set predictions. Must include
                  'true_label', 'predicted_label', 'confidence', and 'correctness'.

    Returns:
        Dictionary containing overall, correct, and incorrect failure metrics.
    """
    total = len(df_clean)
    correct_df = df_clean[df_clean["correctness"] == 1]
    incorrect_df = df_clean[df_clean["correctness"] == 0]

    n_correct = len(correct_df)
    n_incorrect = len(incorrect_df)
    acc = n_correct / total if total > 0 else 0.0

    overall_conf = df_clean["confidence"]
    correct_conf = correct_df["confidence"]
    incorrect_conf = incorrect_df["confidence"]

    metrics = {
        "total_predictions": int(total),
        "correct_predictions": int(n_correct),
        "incorrect_predictions": int(n_incorrect),
        "accuracy": float(acc),
        "average_confidence": float(overall_conf.mean()),
        "median_confidence": float(overall_conf.median()),
        "min_confidence": float(overall_conf.min()),
        "max_confidence": float(overall_conf.max()),
        "correct": {
            "count": int(n_correct),
            "average_confidence": float(correct_conf.mean()) if n_correct > 0 else 0.0,
            "median_confidence": float(correct_conf.median()) if n_correct > 0 else 0.0,
            "std_confidence": float(correct_conf.std(ddof=0)) if n_correct > 0 else 0.0,
        },
        "incorrect": {
            "count": int(n_incorrect),
            "average_confidence": float(incorrect_conf.mean()) if n_incorrect > 0 else 0.0,
            "median_confidence": float(incorrect_conf.median()) if n_incorrect > 0 else 0.0,
            "std_confidence": float(incorrect_conf.std(ddof=0)) if n_incorrect > 0 else 0.0,
        },
    }
    return metrics


def analyze_high_confidence_errors(
    df_preds: pd.DataFrame, thresholds: List[float] = None
) -> pd.DataFrame:
    """Analyze predictions at high confidence thresholds.

    Args:
        df_preds: DataFrame containing prediction results.
        thresholds: List of confidence thresholds (default [0.80, 0.90, 0.95]).

    Returns:
        DataFrame summarizing counts, correct/incorrect, error rate, and percentages per threshold.
    """
    if thresholds is None:
        thresholds = [0.80, 0.90, 0.95]

    total_samples = len(df_preds)
    total_wrong = len(df_preds[df_preds["correctness"] == 0])

    records = []
    for th in sorted(thresholds):
        sub = df_preds[df_preds["confidence"] >= th]
        n_sub = len(sub)
        n_correct = int(np.sum(sub["correctness"] == 1))
        n_incorrect = int(np.sum(sub["correctness"] == 0))

        error_rate = n_incorrect / n_sub if n_sub > 0 else 0.0
        pct_of_all_wrong = (n_incorrect / total_wrong * 100.0) if total_wrong > 0 else 0.0
        pct_of_all_samples = (n_sub / total_samples * 100.0) if total_samples > 0 else 0.0

        records.append(
            {
                "confidence_threshold": th,
                "num_predictions": n_sub,
                "num_correct": n_correct,
                "num_incorrect": n_incorrect,
                "error_rate": float(error_rate),
                "pct_of_all_wrong": float(pct_of_all_wrong),
                "pct_of_all_samples": float(pct_of_all_samples),
            }
        )

    return pd.DataFrame(records)


def analyze_confidence_bins(
    df_preds: pd.DataFrame, num_bins: int = 10
) -> pd.DataFrame:
    """Group predictions into 10 confidence bins and compute per-bin metrics.

    Args:
        df_preds: DataFrame containing prediction results.
        num_bins: Number of confidence bins (default 10).

    Returns:
        DataFrame with columns: bin_index, bin_range, num_predictions, num_correct,
        num_incorrect, bin_accuracy, bin_error_rate, average_confidence.
    """
    records = []
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for i in range(num_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        # Upper bound inclusive for final bin [0.9, 1.0]
        if i == num_bins - 1:
            mask = (df_preds["confidence"] >= low) & (df_preds["confidence"] <= high)
            range_str = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (df_preds["confidence"] >= low) & (df_preds["confidence"] < high)
            range_str = f"[{low:.1f}, {high:.1f})"

        sub = df_preds[mask]
        n_pred = len(sub)
        n_correct = int(np.sum(sub["correctness"] == 1))
        n_incorrect = int(np.sum(sub["correctness"] == 0))

        bin_acc = n_correct / n_pred if n_pred > 0 else 0.0
        bin_err = n_incorrect / n_pred if n_pred > 0 else 0.0
        avg_conf = float(sub["confidence"].mean()) if n_pred > 0 else (low + high) / 2.0

        records.append(
            {
                "bin_index": i,
                "bin_range": range_str,
                "num_predictions": n_pred,
                "num_correct": n_correct,
                "num_incorrect": n_incorrect,
                "bin_accuracy": float(bin_acc),
                "bin_error_rate": float(bin_err),
                "average_confidence": float(avg_conf),
            }
        )

    return pd.DataFrame(records)


def analyze_class_failures(
    df_preds: pd.DataFrame, class_names: Tuple[str, ...] = CIFAR10_CLASSES
) -> pd.DataFrame:
    """Compute per-class accuracy, error rates, and confidence breakdowns.

    Args:
        df_preds: DataFrame containing prediction results.
        class_names: Tuple of class name strings.

    Returns:
        DataFrame detailing per-class metrics.
    """
    records = []
    for class_idx, class_name in enumerate(class_names):
        sub = df_preds[df_preds["true_label"] == class_idx]
        n_samples = len(sub)

        correct_sub = sub[sub["correctness"] == 1]
        incorrect_sub = sub[sub["correctness"] == 0]

        n_correct = len(correct_sub)
        n_incorrect = len(incorrect_sub)
        acc = n_correct / n_samples if n_samples > 0 else 0.0
        err_rate = n_incorrect / n_samples if n_samples > 0 else 0.0

        avg_conf_overall = float(sub["confidence"].mean()) if n_samples > 0 else 0.0
        avg_conf_correct = (
            float(correct_sub["confidence"].mean()) if n_correct > 0 else 0.0
        )
        avg_conf_incorrect = (
            float(incorrect_sub["confidence"].mean()) if n_incorrect > 0 else 0.0
        )

        records.append(
            {
                "class_index": class_idx,
                "class_name": class_name,
                "num_samples": n_samples,
                "num_correct": n_correct,
                "num_incorrect": n_incorrect,
                "accuracy": float(acc),
                "error_rate": float(err_rate),
                "average_confidence": avg_conf_overall,
                "avg_confidence_correct": avg_conf_correct,
                "avg_confidence_incorrect": avg_conf_incorrect,
            }
        )

    return pd.DataFrame(records)


def analyze_confusion(
    df_preds: pd.DataFrame, class_names: Tuple[str, ...] = CIFAR10_CLASSES
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute confusion matrix and rank top confusion pairs.

    Args:
        df_preds: DataFrame containing predictions.
        class_names: Class name strings.

    Returns:
        Tuple of (confusion_matrix_array, top_confusions_dataframe).
    """
    y_true = df_preds["true_label"].values
    y_pred = df_preds["predicted_label"].values

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                sub = df_preds[(df_preds["true_label"] == i) & (df_preds["predicted_label"] == j)]
                avg_conf = float(sub["confidence"].mean()) if len(sub) > 0 else 0.0
                confusions.append(
                    {
                        "true_label": i,
                        "true_class": class_names[i],
                        "predicted_label": j,
                        "predicted_class": class_names[j],
                        "count": int(cm[i, j]),
                        "average_confidence": avg_conf,
                    }
                )

    df_top = pd.DataFrame(confusions).sort_values("count", ascending=False).reset_index(drop=True)
    return cm, df_top


def analyze_shift_failures(prediction_dir: Path) -> pd.DataFrame:
    """Analyze prediction failures and high-confidence error rates across all 26 shift conditions.

    Args:
        prediction_dir: Path to directory containing prediction CSV files.

    Returns:
        DataFrame summarizing accuracy, error rate, confidence breakdown, and high-confidence
        wrong counts for clean and all 25 shift conditions.
    """
    conditions = [("clean", 0)]
    corruptions = ["blur", "noise", "brightness", "contrast", "rotation"]
    for c in corruptions:
        for s in range(1, 6):
            conditions.append((c, s))

    records = []
    for shift_name, severity in conditions:
        if shift_name == "clean":
            file_name = "clean_severity_0_predictions.csv"
            alt_name = "baseline_test_predictions.csv"
            fpath = prediction_dir / file_name
            if not fpath.exists():
                fpath = prediction_dir / alt_name
        else:
            fpath = prediction_dir / f"{shift_name}_severity_{severity}_predictions.csv"

        if not fpath.exists():
            raise FileNotFoundError(f"Missing prediction artifact: {fpath}")

        df = pd.read_csv(fpath)
        total = len(df)
        correct_df = df[df["correctness"] == 1]
        wrong_df = df[df["correctness"] == 0]

        n_correct = len(correct_df)
        n_wrong = len(wrong_df)

        acc = n_correct / total if total > 0 else 0.0
        err_rate = n_wrong / total if total > 0 else 0.0

        avg_conf = float(df["confidence"].mean())
        avg_conf_correct = float(correct_df["confidence"].mean()) if n_correct > 0 else 0.0
        avg_conf_wrong = float(wrong_df["confidence"].mean()) if n_wrong > 0 else 0.0

        wrong_ge_080 = int(np.sum(wrong_df["confidence"] >= 0.80))
        wrong_ge_090 = int(np.sum(wrong_df["confidence"] >= 0.90))
        wrong_ge_095 = int(np.sum(wrong_df["confidence"] >= 0.95))

        frac_ge_080 = wrong_ge_080 / n_wrong if n_wrong > 0 else 0.0
        frac_ge_090 = wrong_ge_090 / n_wrong if n_wrong > 0 else 0.0
        frac_ge_095 = wrong_ge_095 / n_wrong if n_wrong > 0 else 0.0

        records.append(
            {
                "shift": shift_name,
                "severity": severity,
                "total_samples": total,
                "accuracy": float(acc),
                "error_rate": float(err_rate),
                "average_confidence": avg_conf,
                "avg_confidence_correct": avg_conf_correct,
                "avg_confidence_incorrect": avg_conf_wrong,
                "num_incorrect": n_wrong,
                "wrong_ge_080": wrong_ge_080,
                "wrong_ge_090": wrong_ge_090,
                "wrong_ge_095": wrong_ge_095,
                "frac_wrong_ge_080": float(frac_ge_080),
                "frac_wrong_ge_090": float(frac_ge_090),
                "frac_wrong_ge_095": float(frac_ge_095),
            }
        )

    return pd.DataFrame(records)


def extract_high_confidence_failures(
    prediction_dir: Path, threshold: float = 0.90
) -> pd.DataFrame:
    """Extract metadata for high-confidence incorrect predictions across all test conditions.

    Args:
        prediction_dir: Path to directory containing prediction CSV files.
        threshold: Minimum confidence threshold for wrong predictions (default 0.90).

    Returns:
        DataFrame detailing high-confidence failure metadata.
    """
    conditions = [("clean", 0)]
    corruptions = ["blur", "noise", "brightness", "contrast", "rotation"]
    for c in corruptions:
        for s in range(1, 6):
            conditions.append((c, s))

    records = []
    for shift_name, severity in conditions:
        if shift_name == "clean":
            fpath = prediction_dir / "clean_severity_0_predictions.csv"
            if not fpath.exists():
                fpath = prediction_dir / "baseline_test_predictions.csv"
        else:
            fpath = prediction_dir / f"{shift_name}_severity_{severity}_predictions.csv"

        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)
        wrong_high_conf = df[(df["correctness"] == 0) & (df["confidence"] >= threshold)]

        for _, row in wrong_high_conf.iterrows():
            true_idx = int(row["true_label"])
            pred_idx = int(row["predicted_label"])
            records.append(
                {
                    "sample_index": int(row["sample_index"]),
                    "true_label": true_idx,
                    "true_class": CIFAR10_CLASSES[true_idx] if true_idx < len(CIFAR10_CLASSES) else str(true_idx),
                    "predicted_label": pred_idx,
                    "predicted_class": CIFAR10_CLASSES[pred_idx] if pred_idx < len(CIFAR10_CLASSES) else str(pred_idx),
                    "confidence": float(row["confidence"]),
                    "shift": shift_name,
                    "severity": severity,
                }
            )

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        df_out = df_out.sort_values(["confidence", "shift", "severity"], ascending=[False, True, True]).reset_index(drop=True)
    return df_out
