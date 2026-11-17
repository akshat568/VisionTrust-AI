"""Reliability signals module for VisionTrust AI."""

import importlib
from typing import Any

# Mapping of public symbols to (module_path, attribute_name) for PEP 562 lazy exports
_LAZY_EXPORTS = {
    # signals
    "compute_confidence": ("src.reliability.signals", "compute_confidence"),
    "compute_entropy": ("src.reliability.signals", "compute_entropy"),
    "build_feature_centroids": ("src.reliability.signals", "build_feature_centroids"),
    "compute_feature_distance": ("src.reliability.signals", "compute_feature_distance"),
    "compute_ood_energy_score": ("src.reliability.signals", "compute_ood_energy_score"),
    "compute_augmentation_consistency": ("src.reliability.signals", "compute_augmentation_consistency"),
    "compute_augmentation_consistency_onnx": ("src.reliability.signals", "compute_augmentation_consistency_onnx"),
    "compute_image_quality": ("src.reliability.signals", "compute_image_quality"),
    "compute_image_quality_numpy": ("src.reliability.signals", "compute_image_quality_numpy"),
    # evaluator
    "evaluate_single_signal": ("src.reliability.evaluator", "evaluate_single_signal"),
    "analyze_high_confidence_failures_signals": (
        "src.reliability.evaluator",
        "analyze_high_confidence_failures_signals",
    ),
    # calibration
    "compute_ece": ("src.reliability.calibration", "compute_ece"),
    "compute_brier_score": ("src.reliability.calibration", "compute_brier_score"),
    "compute_calibration_curve_data": (
        "src.reliability.calibration",
        "compute_calibration_curve_data",
    ),
    # selective prediction
    "evaluate_selective_prediction": (
        "src.reliability.selective_prediction",
        "evaluate_selective_prediction",
    ),
    # trust model
    "TrustModelPipeline": ("src.reliability.trust_model", "TrustModelPipeline"),
    "FEATURE_COLUMNS": ("src.reliability.trust_model", "FEATURE_COLUMNS"),
    # ablation
    "ABLATION_CONFIGURATIONS": ("src.reliability.ablation", "ABLATION_CONFIGURATIONS"),
    "run_ablation_experiment": ("src.reliability.ablation", "run_ablation_experiment"),
    # robustness
    "evaluate_cross_shift_robustness": (
        "src.reliability.robustness",
        "evaluate_cross_shift_robustness",
    ),
    # bootstrap
    "compute_bootstrap_auroc_ci": (
        "src.reliability.bootstrap",
        "compute_bootstrap_auroc_ci",
    ),
    "compute_bootstrap_auroc_delta_ci": (
        "src.reliability.bootstrap",
        "compute_bootstrap_auroc_delta_ci",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        mod_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(mod_name)
        attr = getattr(module, attr_name)
        globals()[name] = attr  # Cache attribute for subsequent lookups
        return attr
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


__all__ = list(_LAZY_EXPORTS.keys())
