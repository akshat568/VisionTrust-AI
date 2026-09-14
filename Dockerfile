FROM python:3.11-slim

# Prevent Python from writing bytecode files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies for PyTorch / OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository source code
COPY . .

# Accept VISIONTRUST_MODEL_URL as a required build argument
ARG VISIONTRUST_MODEL_URL
ENV VISIONTRUST_MODEL_URL=${VISIONTRUST_MODEL_URL}

# Download baseline ResNet-18 model weights during container build step
RUN python scripts/download_model.py

# Expose container port
EXPOSE 8080

# Launch Uvicorn server listening on 0.0.0.0 and dynamic $PORT
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
