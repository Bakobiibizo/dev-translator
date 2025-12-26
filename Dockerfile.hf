# Translator Service - Using HuggingFace SeamlessM4Tv2
# This uses HuggingFace's pure Python implementation which works with modern PyTorch
#
# Build: docker build -f Dockerfile.hf -t inference/translator:local .
# Run:   docker run --gpus all -p 7104:8104 inference/translator:local

# Use same NGC container as other working services
FROM nvcr.io/nvidia/pytorch:25.09-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Build torchaudio from source (NGC PyTorch is ABI-incompatible with PyPI wheels)
RUN pip install --no-cache-dir --no-deps --no-build-isolation "git+https://github.com/pytorch/audio@release/2.9"

# Install HuggingFace transformers and dependencies
RUN pip install --no-cache-dir \
    transformers>=4.36.0 \
    sentencepiece \
    protobuf \
    accelerate

# Install other dependencies
RUN pip install --no-cache-dir \
    librosa \
    scipy \
    soundfile

# Copy backend code
WORKDIR /app
COPY backend/ /app/backend/

# Install backend dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    loguru \
    python-dotenv \
    pydub

# Environment defaults
ENV API_HOST=0.0.0.0
ENV API_PORT=8104
ENV PYTHONPATH=/app/backend
# HuggingFace cache directory
ENV HF_HOME=/models/huggingface

EXPOSE 8104

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8104/health')" || exit 1

# Override NGC entrypoint to bypass GPU compatibility check
ENTRYPOINT []
CMD ["python", "-m", "uvicorn", "seamless_api.main:app", "--host", "0.0.0.0", "--port", "8104"]
