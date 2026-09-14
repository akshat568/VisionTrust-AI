from contextlib import asynccontextmanager
import io
import logging
from pathlib import Path
import time
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from src.api.config import (
    ALLOWED_CORS_ORIGINS,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
)
from src.api.dependencies import init_pipeline, get_pipeline
from src.api.schemas import (
    HealthResponse,
    PredictResponse,
    BatchPredictResponse,
    PredictionResponse,
    VisionResponse,
    ReliabilityResponse,
    TrustResponse,
    ErrorResponse,
)
from src.inference.pipeline import VisionTrustPipeline, InferenceResult

# Setup structured logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visiontrust.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI application lifespan context manager for startup & shutdown events actions handlers actions actions actions handler actions."""
    logger.info("Starting VisionTrust AI FastAPI Application...")
    try:
        init_pipeline()
    except Exception as e:
        logger.critical(f"Failed to initialize VisionTrustPipeline on startup: {str(e)}", exc_info=True)
        raise e
    yield
    logger.info("Shutting down VisionTrust AI FastAPI Application.")


app = FastAPI(
    title="VisionTrust AI API",
    description=(
        "Reliability-Aware Computer Vision Backend API providing real-time vision classifier "
        "predictions, 6 Phase 6 reliability signal categories, and Phase 7 Platt-scaled Trust Model evaluations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler returning structured error JSON."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"HTTP_{exc.status_code}"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler preventing stack trace leaks to API consumers."""
    logger.error(f"Unhandled internal server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred.", "error_code": "INTERNAL_SERVER_ERROR"},
    )


def map_inference_result(res: InferenceResult, latency_ms: Optional[float] = None) -> PredictResponse:
    """Map pipeline InferenceResult dataclass to API PredictResponse Pydantic schema."""
    sigs = res.reliability_signals
    return PredictResponse(
        prediction=PredictionResponse(
            class_id=res.predicted_class_id,
            class_name=res.predicted_class_name,
        ),
        vision=VisionResponse(
            confidence=res.confidence,
            entropy=sigs["entropy"],
            probabilities=res.probabilities,
        ),
        reliability=ReliabilityResponse(
            feature_distance=sigs["feature_distance"],
            ood_score=sigs["ood_score"],
            augmentation_consistency=sigs["augmentation_consistency"],
            sharpness=sigs["sharpness"],
            brightness=sigs["brightness"],
            contrast=sigs["contrast"],
            composite_quality=sigs["composite_quality"],
        ),
        trust=TrustResponse(
            trust_probability=res.trust_score,
            failure_probability=res.failure_probability,
            status=res.reliability_status,
        ),
        latency_ms=latency_ms,
    )


def validate_and_read_image(file: UploadFile) -> Image.Image:
    """Validate upload file format, size, and convert to PIL Image."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file has no filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext and ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    # Read bytes with file size validation
    contents = file.file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file uploaded.",
        )

    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded image exceeds maximum allowable limit of {MAX_UPLOAD_SIZE_BYTES / (1024 * 1024):.0f}MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(contents))
        pil_img.verify()  # Verify image integrity
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")  # Re-open for actual processing
        return pil_img
    except (UnidentifiedImageError, OSError, ValueError) as e:
        logger.warning(f"Failed image validation for file '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{file.filename}' is not a valid or readable image.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns server status and verifies that the VisionTrust model pipeline is loaded. Does NOT perform inference.",
)
async def health_check():
    """Health check endpoint returning server status."""
    try:
        pipeline = get_pipeline()
        loaded = pipeline is not None
    except Exception:
        loaded = False
    return HealthResponse(status="ok", model_loaded=loaded)


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image file or unsupported format."},
        413: {"model": ErrorResponse, "description": "File payload exceeds 10MB limit."},
        500: {"model": ErrorResponse, "description": "Internal inference error."},
    },
    summary="Predict Reliability for Single Image",
    description=(
        "Upload an image file (PNG, JPG, WEBP, BMP) to obtain the baseline ResNet-18 vision prediction, "
        "6 Phase 6 reliability signals, and Phase 7 Platt-scaled Trust Model failure probability."
    ),
)
async def predict_single(
    image: UploadFile = File(..., description="Image binary file upload (PNG, JPEG, WEBP, BMP)."),
    pipeline: VisionTrustPipeline = Depends(get_pipeline),
):
    """Run real inference on a single uploaded image."""
    start_time = time.perf_counter()
    logger.info(f"Received prediction request for file: '{image.filename}'")

    pil_img = validate_and_read_image(image)

    try:
        inference_res = pipeline.predict_image(pil_img)
    except Exception as e:
        logger.error(f"Inference error processing '{image.filename}': {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during model inference.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    logger.info(
        f"Prediction complete for '{image.filename}' in {elapsed_ms}ms | "
        f"Class: {inference_res.predicted_class_name} | Trust: {inference_res.trust_score:.4f} ({inference_res.reliability_status})"
    )

    return map_inference_result(inference_res, latency_ms=elapsed_ms)


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Predict Reliability for Batch of Images",
    description="Upload multiple image files to obtain VisionTrust predictions and reliability evaluations for each image.",
)
async def predict_batch(
    images: List[UploadFile] = File(..., description="Multiple image binary file uploads."),
    pipeline: VisionTrustPipeline = Depends(get_pipeline),
):
    """Run real inference on a batch of uploaded images."""
    if not images:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded in batch request.",
        )

    logger.info(f"Received batch prediction request for {len(images)} files.")
    pil_images = [validate_and_read_image(f) for f in images]

    start_time = time.perf_counter()
    try:
        results: List[InferenceResult] = pipeline.predict_batch(pil_images)
    except Exception as e:
        logger.error(f"Batch inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during batch model inference.",
        )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    per_image_latency = round(elapsed_ms / len(images), 2)

    response_list = [map_inference_result(r, latency_ms=per_image_latency) for r in results]
    logger.info(f"Batch prediction complete: {len(images)} images in {elapsed_ms}ms ({per_image_latency}ms/img).")

    return BatchPredictResponse(total_processed=len(images), results=response_list)
