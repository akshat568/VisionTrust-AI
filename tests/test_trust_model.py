from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS
from src.reliability.calibration import compute_ece, compute_brier_score, compute_calibration_curve_data
from src.reliability.selective_prediction import evaluate_selective_prediction


def test_target_construction():
    """Verify target is 1 if predicted_label == true_label, 0 otherwise."""
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 9, 3, 5])
    target = (y_true == y_pred).astype(int)

    np.testing.assert_array_equal(target, [1, 1, 0, 1, 0])


def test_no_label_leakage_in_features():
    """Verify true_label is excluded from input feature columns."""
    df_dummy = pd.DataFrame({col: np.random.rand(10) for col in FEATURE_COLUMNS})
    df_dummy["true_label"] = np.random.randint(0, 10, size=10)
    df_dummy["correctness"] = np.random.randint(0, 2, size=10)

    pipeline = TrustModelPipeline(seed=42)

    assert "true_label" not in pipeline.feature_names
    assert "correctness" not in pipeline.feature_names

    X_mat = pipeline._extract_features(df_dummy)
    assert X_mat.shape == (10, len(FEATURE_COLUMNS))


def test_trust_model_fit_predict_and_bounds():
    """Verify TrustModelPipeline fits and outputs valid probabilities in [0, 1]."""
    np.random.seed(42)
    N = 100
    X_dummy = np.random.randn(N, len(FEATURE_COLUMNS))
    y_dummy = np.random.randint(0, 2, size=N)

    pipeline = TrustModelPipeline(seed=42)
    pipeline.fit(X_dummy, y_dummy)

    p_correct_cal = pipeline.predict_proba(X_dummy, model_type="calibrated")
    p_correct_log = pipeline.predict_proba(X_dummy, model_type="logistic")
    p_correct_rf = pipeline.predict_proba(X_dummy, model_type="rf")
    p_fail = pipeline.predict_failure_proba(X_dummy, model_type="calibrated")

    assert p_correct_cal.shape == (N,)
    assert np.all(p_correct_cal >= 0.0) and np.all(p_correct_cal <= 1.0)
    assert np.all(p_correct_log >= 0.0) and np.all(p_correct_log <= 1.0)
    assert np.all(p_correct_rf >= 0.0) and np.all(p_correct_rf <= 1.0)
    np.testing.assert_allclose(p_fail, 1.0 - p_correct_cal, atol=1e-6)


def test_calibration_metrics():
    """Verify ECE and Brier score calculations."""
    y_true = np.array([1, 1, 0, 0], dtype=np.float32)
    y_prob_perfect = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    y_prob_imperfect = np.array([0.8, 0.8, 0.2, 0.2], dtype=np.float32)

    ece_perfect = compute_ece(y_true, y_prob_perfect)
    brier_perfect = compute_brier_score(y_true, y_prob_perfect)

    assert np.isclose(ece_perfect, 0.0, atol=1e-5)
    assert np.isclose(brier_perfect, 0.0, atol=1e-5)

    ece_imp = compute_ece(y_true, y_prob_imperfect)
    brier_imp = compute_brier_score(y_true, y_prob_imperfect)

    assert ece_imp >= 0.0 and ece_imp <= 1.0
    assert brier_imp > 0.0 and brier_imp <= 1.0

    cal_data = compute_calibration_curve_data(y_true, y_prob_imperfect, n_bins=5)
    assert "bin_accuracies" in cal_data and "bin_confidences" in cal_data and "bin_counts" in cal_data


def test_selective_prediction_evaluation():
    """Verify selective prediction metrics and monotonic accuracy trend."""
    correctness = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01])

    df_sel = evaluate_selective_prediction(correctness, scores, coverages=[1.0, 0.5, 0.4])

    assert len(df_sel) == 3
    assert set(df_sel.columns) == {"target_coverage", "coverage", "retained_count", "rejected_count", "accuracy", "risk"}

    # Accuracy at 40% coverage (retaining top 4 correct samples) should be 100%
    row_40 = df_sel[df_sel["target_coverage"] == 0.4].iloc[0]
    assert np.isclose(row_40["accuracy"], 1.0)
    assert np.isclose(row_40["risk"], 0.0)


def test_pipeline_save_and_load(tmp_path):
    """Verify TrustModelPipeline serialization and deserialization."""
    X_dummy = np.random.randn(50, len(FEATURE_COLUMNS))
    y_dummy = np.random.randint(0, 2, size=50)

    pipe = TrustModelPipeline(seed=42)
    pipe.fit(X_dummy, y_dummy)

    save_path = tmp_path / "test_trust_model.pkl"
    pipe.save(save_path)

    loaded_pipe = TrustModelPipeline.load(save_path)
    assert loaded_pipe.is_fitted

    p_orig = pipe.predict_proba(X_dummy)
    p_loaded = loaded_pipe.predict_proba(X_dummy)

    np.testing.assert_array_equal(p_orig, p_loaded)
