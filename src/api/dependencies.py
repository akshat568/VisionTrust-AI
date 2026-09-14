import logging
from typing import Optional
from src.inference.pipeline import VisionTrustPipeline
from src.api.config import MODEL_PATH, CENTROIDS_PATH, TRUST_MODEL_PATH, DEFAULT_THRESHOLD

logger = logging.getLogger("visiontrust.api")

_pipeline_instance: Optional[VisionTrustPipeline] = None


def init_pipeline(
    model_path: str = MODEL_PATH,
    centroids_path: str = CENTROIDS_PATH,
    trust_model_path: str = TRUST_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> VisionTrustPipeline:
    """Initialize the singleton VisionTrustPipeline instance on server startup.

    Fails immediately with an explicit FileNotFoundError if any model artifact is missing.
    """
    global _pipeline_instance
    logger.info("Initializing VisionTrustPipeline from artifacts...")
    _pipeline_instance = VisionTrustPipeline(
        model_path=model_path,
        centroids_path=centroids_path,
        trust_model_path=trust_model_path,
        threshold=threshold,
    )
    logger.info("VisionTrustPipeline successfully loaded and ready for inference.")
    return _pipeline_instance


def get_pipeline() -> VisionTrustPipeline:
    """Dependency getter returning the active VisionTrustPipeline instance.

    Raises:
        RuntimeError: If called before pipeline initialization.
    """
    if _pipeline_instance is None:
        raise RuntimeError("VisionTrustPipeline is not initialized! Application failed to start properly.")
    return _pipeline_instance
