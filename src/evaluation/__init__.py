"""Evaluation module for VisionTrust AI."""

from src.evaluation.failure_analysis import (
    analyze_clean_failures,
    analyze_high_confidence_errors,
    analyze_confidence_bins,
    analyze_class_failures,
    analyze_confusion,
    analyze_shift_failures,
    extract_high_confidence_failures,
)

__all__ = [
    "analyze_clean_failures",
    "analyze_high_confidence_errors",
    "analyze_confidence_bins",
    "analyze_class_failures",
    "analyze_confusion",
    "analyze_shift_failures",
    "extract_high_confidence_failures",
]
