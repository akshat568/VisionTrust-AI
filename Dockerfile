FROM python:3.11-slim

# Prevent Python from writing bytecode files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INFERENCE_ENGINE=onnx \
    PORT=10000 \
    CORS_ORIGINS="https://vision-trust-ai.vercel.app,http://localhost:5173"

WORKDIR /app

# Install minimal system utilities for healthchecks and TLS certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy production dependency manifest and install lightweight production runtime
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy repository source code and artifacts
COPY . .

# Optional build argument to download ONNX weights if not bundled in build context
ARG VISIONTRUST_ONNX_MODEL_URL
ENV VISIONTRUST_ONNX_MODEL_URL=${VISIONTRUST_ONNX_MODEL_URL}

# Verify ONNX model exists, or download if URL is provided
RUN python scripts/download_model.py --onnx --check-only || \
    ( [ -n "$VISIONTRUST_ONNX_MODEL_URL" ] && python scripts/download_model.py --onnx ) || \
    ( [ -f outputs/models/baseline_resnet18.onnx ] && echo "ONNX model confirmed." )

# Validate that all required production artifacts are present before finalizing image
RUN python -c "\
from pathlib import Path; \
assert Path('outputs/models/baseline_resnet18.onnx').is_file(), 'Missing outputs/models/baseline_resnet18.onnx'; \
assert Path('outputs/models/trust_model.pkl').is_file(), 'Missing outputs/models/trust_model.pkl'; \
assert Path('outputs/metrics/train_feature_centroids.npy').is_file(), 'Missing outputs/metrics/train_feature_centroids.npy'; \
print('All production artifacts successfully verified.')"

# Expose container port (Render default web service port is 10000)
EXPOSE 10000

# Health check to ensure Uvicorn is healthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT:-10000}/health || exit 1

# Launch Uvicorn server bound to 0.0.0.0 and dynamic $PORT injected by Render
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
