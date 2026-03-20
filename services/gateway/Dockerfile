FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install curl for healthcheck and ffmpeg for audio conversion (pydub)
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Runtime configuration (overridable via docker-compose or `docker run -e`)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    SPEECH_URL=http://chatterbox:8000 \
    TRANSCRIPTION_URL=http://whisperx:8000 \
    CHAT_URL= \
    VOICES_DIR=/app/voices \
    BACKEND_TIMEOUT=300

# Copy project metadata + lock file first for layer caching
COPY services/gateway/pyproject.toml services/gateway/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

# Copy application source
COPY services/gateway/src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --frozen

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn gateway_service.main:app --host ${HOST} --port ${PORT}"]
