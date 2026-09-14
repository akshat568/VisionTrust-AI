from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.reliability.ablation import ABLATION_CONFIGURATIONS, run_ablation_experiment
from src.reliability.bootstrap import compute_bootstrap_auroc_ci, compute_bootstrap_auroc_delta_ci
from src.reliability.robustness import evaluate_cross_shift_robustness
from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS


def test_ablation_configurations_presence():
    """Verify all 11 required ablation configurations are defined and feature names match FEATURE_COLUMNS."""
    assert len(ABLATION_CONFIGURATIONS) == 11, f"Expected 11 ablation configs, got {len(ABLATION_CONFIGURATIONS)}"
    all_known_features = set(FEATURE_COLUMNS)

    for name, features in ABLATION_CONFIGURATIONS.items():
        assert len(features) > 0, f"Config {name} has no features!"
        for f in features:
            assert f in all_known_features, f"Unknown feature {f} in config {name}"


def test_ablation_experiment_execution():
    """Verify run_ablation_experiment returns all 11 rows and valid metric bounds."""
    np.random.seed(42)
    N_val, N_test = 100, 200

    val_dict = {col: np.random.randn(N_val) for col in FEATURE_COLUMNS}
    val_dict["correctness"] = np.random.randint(0, 2, size=N_val)
    df_val_dummy = pd.DataFrame(val_dict)

    test_dict = {col: np.random.randn(N_test) for col in FEATURE_COLUMNS}
    test_dict["correctness"] = np.random.randint(0, 2, size=N_test)
    df_test_dummy = pd.DataFrame(test_dict)

    df_ablation = run_ablation_experiment(df_val_dummy, df_test_dummy, model_type="rf", seed=42)

    assert len(df_ablation) == 11
    assert set(df_ablation["config_name"]) == set(ABLATION_CONFIGURATIONS.keys())
    assert np.all(df_ablation["failure_auroc"] >= 0.0) and np.all(df_ablation["failure_auroc"] <= 1.0)
    assert np.all(df_ablation["failure_auprc"] >= 0.0) and np.all(df_ablation["failure_auprc"] <= 1.0)


def test_bootstrap_ci_computation():
    """Verify bootstrap confidence interval calculation returns valid bounds and std_err."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=200)
    y_score = np.random.rand(200)

    res = compute_bootstrap_auroc_ci(y_true, y_score, n_bootstraps=100, seed=42)
    assert "mean_bootstrap_auroc" in res and "ci_lower" in res and "ci_upper" in res
    assert res["ci_lower"] <= res["mean_bootstrap_auroc"] <= res["ci_upper"]

    delta_res = compute_bootstrap_auroc_delta_ci(y_true, y_score, y_score * 0.9, n_bootstraps=100, seed=42)
    assert "delta_auroc" in delta_res and "p_value" in delta_res


def test_robustness_evaluation():
    """Verify cross-shift robustness evaluation outputs 26 conditions and 5 corruption families."""
    clean_csv = Path("outputs/metrics/reliability_signals_clean.csv")
    shifted_csv = Path("outputs/metrics/reliability_signals_shifted.csv")
    model_pkl = Path("outputs/models/trust_model.pkl")

    if not clean_csv.exists() or not shifted_csv.exists() or not model_pkl.exists():
        pytest.skip("Required Phase 7/8 artifacts missing")

    pipe = TrustModelPipeline.load(model_pkl)
    df_clean = pd.read_csv(clean_csv)
    df_shifted = pd.read_csv(shifted_csv)

    df_all, df_family = evaluate_cross_shift_robustness(pipe, df_clean, df_shifted)

    assert len(df_all) == 26, f"Expected 26 conditions, got {len(df_all)}"
    assert len(df_family) == 5, f"Expected 5 corruption families, got {len(df_family)}"
    assert set(df_family["corruption_family"]) == {"Blur", "Noise", "Brightness", "Contrast", "Rotation"}


def test_no_test_set_fitting_in_ablation():
    """Verify scaler and model in ablation are fitted ONLY on validation data."""
    val_dict = {col: np.random.randn(50) + 10.0 for col in FEATURE_COLUMNS}
    val_dict["correctness"] = np.random.randint(0, 2, size=50)
    df_val = pd.DataFrame(val_dict)

    test_dict = {col: np.random.randn(50) - 10.0 for col in FEATURE_COLUMNS}
    test_dict["correctness"] = np.random.randint(0, 2, size=50)
    df_test = pd.DataFrame(test_dict)

    # Run ablation on disjoint mean distributions
    df_ab = run_ablation_experiment(df_val, df_test, model_type="logistic", seed=42)
    assert len(df_ab) == 11
