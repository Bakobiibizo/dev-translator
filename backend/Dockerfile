# SeamlessM4T Translation API - CPU Version

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY seamless_api/ seamless_api/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install PyTorch CPU version first
RUN uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install seamless_communication and other dependencies
RUN uv pip install .

# Create model cache directory
RUN mkdir -p /app/models
ENV MODEL_CACHE_DIR=/app/models
ENV HF_HOME=/app/models

# Default environment
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cpu
ENV TORCH_DTYPE=float32
ENV MODEL_NAME=seamlessM4T_v2_large
ENV VOCODER_NAME=vocoder_v2

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run the service
CMD ["python", "-m", "seamless_api.main"]
