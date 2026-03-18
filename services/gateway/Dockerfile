FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Runtime configuration (overridable via docker-compose or `docker run -e`)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    CHATTERBOX_URL=http://chatterbox:8000 \
    WHISPERX_URL=http://whisperx:8000 \
    PROXY_TIMEOUT=300

# Copy project metadata first for layer caching
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application source
COPY src/ src/
RUN uv sync --no-dev

EXPOSE ${PORT}

CMD ["sh", "-c", "uv run uvicorn gateway_service.main:app --host ${HOST} --port ${PORT}"]
