FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Python 3.11 (via deadsnakes) — chatterbox-tts pins numpy<1.26 which needs
# distutils, removed in 3.12.  Also install curl (healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
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
    DEFAULT_MODEL=chatterbox-turbo

# Copy project metadata first for layer caching
COPY services/chatterbox/pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application source
COPY services/chatterbox/src/ src/
RUN uv sync --no-dev

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn chatterbox_service.app:app --host ${HOST} --port ${PORT}"]
