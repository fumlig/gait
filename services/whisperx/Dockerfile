FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Python 3.11 via deadsnakes, ffmpeg for audio processing, curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Pin uv to Python 3.11
ENV UV_PYTHON=python3.11

# Runtime configuration (overridable via docker-compose or `docker run -e`)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cuda \
    DEFAULT_MODEL=large-v3 \
    COMPUTE_TYPE=float16 \
    BATCH_SIZE=16 \
    ENABLE_DIARIZATION=false

# Copy project metadata + lock file first for layer caching
COPY services/whisperx/pyproject.toml services/whisperx/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

# Copy application source
COPY services/whisperx/src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn whisperx_service.app:app --host ${HOST} --port ${PORT}"]
