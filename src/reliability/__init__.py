"""Reliability signals module for VisionTrust AI."""

from src.reliability.signals import (
    compute_confidence,
    compute_entropy,
    build_feature_centroids,
    compute_feature_distance,
    compute_ood_energy_score,
    compute_augmentation_consistency,
    compute_image_quality,
)
from src.reliability.evaluator import (
    evaluate_single_signal,
    analyze_high_confidence_failures_signals,
)
from src.reliability.calibration import (
    compute_ece,
    compute_brier_score,
    compute_calibration_curve_data,
)
from src.reliability.selective_prediction import evaluate_selective_prediction
from src.reliability.trust_model import TrustModelPipeline, FEATURE_COLUMNS
from src.reliability.ablation import ABLATION_CONFIGURATIONS, run_ablation_experiment
from src.reliability.robustness import evaluate_cross_shift_robustness
from src.reliability.bootstrap import compute_bootstrap_auroc_ci, compute_bootstrap_auroc_delta_ci

__all__ = [
    "compute_confidence",
    "compute_entropy",
    "build_feature_centroids",
    "compute_feature_distance",
    "compute_ood_energy_score",
    "compute_augmentation_consistency",
    "compute_image_quality",
    "evaluate_single_signal",
    "analyze_high_confidence_failures_signals",
    "compute_ece",
    "compute_brier_score",
    "compute_calibration_curve_data",
    "evaluate_selective_prediction",
    "TrustModelPipeline",
    "FEATURE_COLUMNS",
    "ABLATION_CONFIGURATIONS",
    "run_ablation_experiment",
    "evaluate_cross_shift_robustness",
    "compute_bootstrap_auroc_ci",
    "compute_bootstrap_auroc_delta_ci",
]


