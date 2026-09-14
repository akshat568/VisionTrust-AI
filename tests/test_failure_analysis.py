from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.failure_analysis import (
    analyze_clean_failures,
    analyze_high_confidence_errors,
    analyze_confidence_bins,
    analyze_class_failures,
    analyze_shift_failures,
)


@pytest.fixture
def sample_clean_predictions():
    """Fixture to load actual clean test predictions or construct fallback DataFrame."""
    clean_path = Path("outputs/predictions/clean_severity_0_predictions.csv")
    if not clean_path.exists():
        clean_path = Path("outputs/predictions/baseline_test_predictions.csv")

    if clean_path.exists():
        return pd.read_csv(clean_path)
    else:
        # Fallback synthetic predictions for isolated unit testing
        np.random.seed(42)
        n = 10000
        true_labels = np.random.randint(0, 10, n)
        pred_labels = np.random.randint(0, 10, n)
        confidences = np.random.uniform(0.1, 1.0, n)
        correctness = (true_labels == pred_labels).astype(int)
        return pd.DataFrame(
            {
                "sample_index": np.arange(n),
                "true_label": true_labels,
                "predicted_label": pred_labels,
                "confidence": confidences,
                "correctness": correctness,
            }
        )


def test_confidence_values_bounds(sample_clean_predictions):
    """Verify all confidence values strictly remain in range [0, 1]."""
    confs = sample_clean_predictions["confidence"].values
    assert np.all(confs >= 0.0), "Confidence values contain numbers below 0!"
    assert np.all(confs <= 1.0), "Confidence values contain numbers above 1!"


def test_correct_incorrect_counts_sum_to_total(sample_clean_predictions):
    """Verify correct + incorrect counts equal total predictions."""
    summary = analyze_clean_failures(sample_clean_predictions)
    total = summary["total_predictions"]
    n_correct = summary["correct_predictions"]
    n_incorrect = summary["incorrect_predictions"]

    assert n_correct + n_incorrect == total, f"Sum ({n_correct}+{n_incorrect}) != total ({total})"
    assert summary["correct"]["count"] == n_correct
    assert summary["incorrect"]["count"] == n_incorrect


def test_confidence_bins_cover_all_predictions(sample_clean_predictions):
    """Verify confidence bins cover 100% of test predictions without loss."""
    df_bins = analyze_confidence_bins(sample_clean_predictions, num_bins=10)
    assert len(df_bins) == 10, f"Expected 10 bins, got {len(df_bins)}"

    total_binned_preds = df_bins["num_predictions"].sum()
    total_samples = len(sample_clean_predictions)
    assert (
        total_binned_preds == total_samples
    ), f"Binned count {total_binned_preds} != total samples {total_samples}"

    for _, row in df_bins.iterrows():
        assert (
            row["num_correct"] + row["num_incorrect"] == row["num_predictions"]
        ), f"Bin correctness mismatch in {row['bin_range']}"


def test_class_counts_sum_to_total(sample_clean_predictions):
    """Verify per-class sample counts sum to total dataset count (10,000 for clean test set)."""
    df_class = analyze_class_failures(sample_clean_predictions)
    assert len(df_class) == 10, f"Expected 10 class rows, got {len(df_class)}"

    total_class_samples = df_class["num_samples"].sum()
    total_samples = len(sample_clean_predictions)
    assert (
        total_class_samples == total_samples
    ), f"Class sample sum {total_class_samples} != total {total_samples}"


def test_high_confidence_error_counts(sample_clean_predictions):
    """Verify high-confidence error counts are non-negative and bounded."""
    df_th = analyze_high_confidence_errors(sample_clean_predictions, thresholds=[0.80, 0.90, 0.95])
    assert len(df_th) == 3

    for _, row in df_th.iterrows():
        assert row["num_predictions"] >= 0
        assert row["num_correct"] >= 0
        assert row["num_incorrect"] >= 0
        assert row["num_correct"] + row["num_incorrect"] == row["num_predictions"]
        assert 0.0 <= row["error_rate"] <= 1.0


def test_shift_failure_analysis_conditions():
    """Verify shift failure analysis processes all 26 conditions."""
    pred_dir = Path("outputs/predictions")
    if not pred_dir.exists():
        pytest.skip("outputs/predictions directory does not exist")

    df_shift = analyze_shift_failures(pred_dir)
    assert len(df_shift) == 26, f"Expected 26 shift rows (clean + 25 shifts), got {len(df_shift)}"

    clean_rows = df_shift[df_shift["shift"] == "clean"]
    assert len(clean_rows) == 1, "Clean condition missing from shift failure analysis!"
