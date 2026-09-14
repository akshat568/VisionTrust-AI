from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Vision classifier output prediction."""

    class_id: int = Field(..., description="Predicted CIFAR-10 class ID (0 to 9).", json_schema_extra={"example": 6})
    class_name: str = Field(..., description="Predicted CIFAR-10 class label.", json_schema_extra={"example": "frog"})


class VisionResponse(BaseModel):
    """Vision prediction confidence and uncertainty metrics."""

    confidence: float = Field(..., description="Maximum class probability confidence [0, 1].", json_schema_extra={"example": 0.9254})
    entropy: float = Field(..., description="Shannon prediction entropy (non-negative).", json_schema_extra={"example": 0.2415})
    probabilities: List[float] = Field(
        ..., description="Complete 10-class softmax probability distribution.", json_schema_extra={"example": [0.01, 0.02, 0.92, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.01]}
    )


class ReliabilityResponse(BaseModel):
    """Six Phase 6 reliability signal categories extracted from model & image."""

    feature_distance: float = Field(
        ..., description="Euclidean distance to predicted class feature centroid.", json_schema_extra={"example": 9.8512}
    )
    ood_score: float = Field(
        ..., description="Logit free-energy score at T=1.0 (higher = in-distribution).", json_schema_extra={"example": 3.4120}
    )
    augmentation_consistency: float = Field(
        ..., description="Agreement rate under 3 light augmentations [0, 1].", json_schema_extra={"example": 1.0}
    )
    sharpness: float = Field(..., description="Variance of 2D Laplacian operator.", json_schema_extra={"example": 0.0825})
    brightness: float = Field(..., description="Mean pixel intensity across channels.", json_schema_extra={"example": 0.4912})
    contrast: float = Field(..., description="Standard deviation of pixel intensity.", json_schema_extra={"example": 0.2451})
    composite_quality: float = Field(..., description="Composite image quality score.", json_schema_extra={"example": 0.3365})


class TrustResponse(BaseModel):
    """Phase 7 Platt-scaled Trust Model reliability evaluation."""

    trust_probability: float = Field(
        ..., description="Calibrated P(original prediction is correct) in [0, 1].", json_schema_extra={"example": 0.9421}
    )
    failure_probability: float = Field(
        ..., description="Model failure probability 1 - P(trust) in [0, 1].", json_schema_extra={"example": 0.0579}
    )
    status: str = Field(
        ..., description="Reliability classification status: 'TRUSTED' or 'UNTRUSTED'.", json_schema_extra={"example": "TRUSTED"}
    )


class PredictResponse(BaseModel):
    """Unified response object returned by POST /predict."""

    prediction: PredictionResponse
    vision: VisionResponse
    reliability: ReliabilityResponse
    trust: TrustResponse
    latency_ms: Optional[float] = Field(None, description="Server processing latency in milliseconds.")


class BatchPredictResponse(BaseModel):
    """Batch response object returned by POST /predict/batch."""

    total_processed: int = Field(..., description="Total number of images processed in batch.")
    results: List[PredictResponse] = Field(..., description="List of prediction responses.")


class HealthResponse(BaseModel):
    """Response object returned by GET /health."""

    status: str = Field(..., json_schema_extra={"example": "ok"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})


class ErrorResponse(BaseModel):
    """Standardized structured error response."""

    detail: str = Field(..., description="Human-readable error explanation.")
    error_code: str = Field(..., description="Machine-readable error identifier.", json_schema_extra={"example": "INVALID_IMAGE_FILE"})
