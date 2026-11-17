import logging
import os
from typing import Optional, Any
from src.inference.onnx_pipeline import ONNXVisionTrustPipeline
from src.api.config import (
    INFERENCE_ENGINE,
    ONNX_MODEL_PATH,
    MODEL_PATH,
    CENTROIDS_PATH,
    TRUST_MODEL_PATH,
    DEFAULT_THRESHOLD,
)

logger = logging.getLogger("visiontrust.api")

_pipeline_instance: Optional[Any] = None


def init_pipeline(
    engine: Optional[str] = None,
    onnx_model_path: str = ONNX_MODEL_PATH,
    model_path: str = MODEL_PATH,
    centroids_path: str = CENTROIDS_PATH,
    trust_model_path: str = TRUST_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    force_reload: bool = False,
) -> Any:
    """Initialize the singleton pipeline instance on server startup.

    Supports:
      - engine='onnx' (production default, <= 512 MB RAM footprint)
      - engine='pytorch' (fallback/development engine)
    """
    global _pipeline_instance
    if _pipeline_instance is not None and not force_reload:
        return _pipeline_instance

    target_engine = (engine or INFERENCE_ENGINE).lower()

    if target_engine == "onnx":
        logger.info(f"Initializing ONNXVisionTrustPipeline from '{onnx_model_path}'...")
        _pipeline_instance = ONNXVisionTrustPipeline(
            model_path=onnx_model_path,
            centroids_path=centroids_path,
            trust_model_path=trust_model_path,
            threshold=threshold,
        )
        logger.info("ONNXVisionTrustPipeline successfully loaded and ready for production inference.")
    elif target_engine == "pytorch":
        logger.info(f"Initializing PyTorch VisionTrustPipeline fallback from '{model_path}'...")
        from src.inference.pipeline import VisionTrustPipeline

        _pipeline_instance = VisionTrustPipeline(
            model_path=model_path,
            centroids_path=centroids_path,
            trust_model_path=trust_model_path,
            threshold=threshold,
        )
        logger.info("PyTorch VisionTrustPipeline successfully loaded.")
    else:
        raise ValueError(f"Unknown INFERENCE_ENGINE '{target_engine}'. Expected 'onnx' or 'pytorch'.")

    return _pipeline_instance


def get_pipeline() -> Any:
    """Dependency getter returning the active pipeline instance.

    Raises:
        RuntimeError: If called before pipeline initialization.
    """
    if _pipeline_instance is None:
        raise RuntimeError("Inference pipeline is not initialized! Application failed to start properly.")
    return _pipeline_instance
