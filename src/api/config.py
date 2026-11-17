import os
from typing import List, Set

# API Server Settings
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", 8000))

# CORS Configuration for local & production deployment
DEFAULT_CORS_ORIGINS: List[str] = [
    "https://vision-trust-ai.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

_cors_env = os.getenv("CORS_ORIGINS")
if _cors_env:
    ALLOWED_CORS_ORIGINS: List[str] = [
        origin.strip() for origin in _cors_env.split(",") if origin.strip()
    ]
else:
    ALLOWED_CORS_ORIGINS: List[str] = DEFAULT_CORS_ORIGINS

# Security & Upload Boundaries
MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_IMAGE_CONTENT_TYPES: Set[str] = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
}
ALLOWED_IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

# Inference Engine Selection: "onnx" (production default, <= 512 MB RAM) or "pytorch" (development fallback)
INFERENCE_ENGINE: str = os.getenv("INFERENCE_ENGINE", "onnx").lower()

# Artifact Paths & Decision Threshold
ONNX_MODEL_PATH: str = os.getenv("ONNX_MODEL_PATH", "outputs/models/baseline_resnet18.onnx")
MODEL_PATH: str = os.getenv("MODEL_PATH", "outputs/models/baseline_resnet18_best.pth")
CENTROIDS_PATH: str = os.getenv("CENTROIDS_PATH", "outputs/metrics/train_feature_centroids.npy")
TRUST_MODEL_PATH: str = os.getenv("TRUST_MODEL_PATH", "outputs/models/trust_model.pkl")
DEFAULT_THRESHOLD: float = float(os.getenv("DEFAULT_THRESHOLD", "0.50"))
