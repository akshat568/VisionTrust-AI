"""Inference package for VisionTrust AI."""

from src.inference.schema import InferenceResult
from src.inference.onnx_pipeline import ONNXVisionTrustPipeline


def __getattr__(name: str):
    if name == "VisionTrustPipeline":
        from src.inference.pipeline import VisionTrustPipeline
        return VisionTrustPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["VisionTrustPipeline", "ONNXVisionTrustPipeline", "InferenceResult"]
