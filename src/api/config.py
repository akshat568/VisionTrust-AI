from typing import List, Set

# API Server Settings
HOST: str = "127.0.0.1"
PORT: int = 8000

# CORS Configuration for local frontend development (Vite/React)
ALLOWED_CORS_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

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

# Artifact Paths & Decision Threshold
MODEL_PATH: str = "outputs/models/baseline_resnet18_best.pth"
CENTROIDS_PATH: str = "outputs/metrics/train_feature_centroids.npy"
TRUST_MODEL_PATH: str = "outputs/models/trust_model.pkl"
DEFAULT_THRESHOLD: float = 0.50
