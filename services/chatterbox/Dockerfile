FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Python 3.11 (chatterbox-tts needs numpy<1.26 which requires distutils)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_PYTHON=python3.11

ENV HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cuda \
    DEFAULT_MODEL=chatterbox-turbo

COPY services/chatterbox/pyproject.toml services/chatterbox/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY services/chatterbox/src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn chatterbox_service.app:app --host ${HOST} --port ${PORT}"]
