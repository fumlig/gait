FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# ffmpeg for audio conversion (pydub)
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV HOST=0.0.0.0 \
    PORT=8000 \
    BACKEND_TIMEOUT=300

COPY gateway/pyproject.toml gateway/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY gateway/src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn gateway_service.main:app --host ${HOST} --port ${PORT}"]
