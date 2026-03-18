# AGENTS.md

Context and conventions for AI agents working on this repository.

## Project overview

Trave exposes local ML models via OpenAI-compatible REST APIs. Each model runs as an independent service in its own Docker container with GPU passthrough.

## Repository structure

```
trave/
  docker-compose.yml          # orchestrates all services
  .env                        # (optional) host-level env vars
  AGENTS.md                   # this file
  services/
    <service-name>/
      Dockerfile              # CUDA base image + Python + uv
      pyproject.toml           # deps managed by uv, linted by ruff
      README.md                # service-specific docs
      .dockerignore
      voices/ or assets/       # service-specific data (volume-mounted)
      src/<package_name>/
        main.py                # FastAPI app + lifespan
        config.py              # pydantic-settings (env vars, no prefix)
        models.py              # request/response schemas
        engine.py              # model loading + inference singleton
        routes/
          ...                  # one file per endpoint group
      tests/
        test_api.py            # pytest + httpx, mocked engine
```

## Conventions

### Services
- One folder per service under `services/`.
- Each service is an independent Python project with its own `pyproject.toml`.
- Use `uv` for dependency management, `ruff` for linting, `pytest` for testing.
- Every service gets its own `README.md` with endpoint docs, config table, and build notes.

### API design
- Match the OpenAI API contract for the relevant modality (TTS, STT, chat, embeddings, images, etc.).
- Always implement `GET /v1/models` and `GET /health`.
- Accept and ignore `Authorization` headers (local use, no auth).
- Return proper HTTP error codes: 400 for bad input, 503 for model not loaded, 500 for inference failures.

### Docker
- Base image: `nvidia/cuda:<version>-runtime-ubuntu<version>` for GPU services.
- Use `ARG` for build-time configurability (Python version, CUDA version).
- Use `ENV` with defaults for runtime configurability (port, host, device).
- The `CMD` should reference env vars so they can be overridden at runtime.
- Layer caching: copy `pyproject.toml` and `uv sync` before copying source.
- Include `curl` in the image for healthcheck.

### docker-compose.yml
- Every port, volume path, and env var should be configurable via `${VAR:-default}` syntax.
- Mount `HF_HOME` from host for model weight caching.
- Pass `HF_TOKEN` for gated model access.
- Use `deploy.resources.reservations.devices` for GPU passthrough.
- Set `restart: unless-stopped` and a healthcheck with generous `start_period` (models take time to load).

### Configuration (config.py)
- Use `pydantic-settings` with `BaseSettings`.
- No env prefix -- variables map directly (e.g., `DEVICE=cuda`).
- Expose all tunables: device, host, port, data dirs, limits.

### Engine pattern
- Module-level singleton instance created at import time.
- `load()` called during FastAPI lifespan startup, `unload()` on shutdown.
- Lazy import of the ML library inside `load()` to avoid import-time GPU init.
- The engine owns model loading, inference, and any pre/post-processing.

## Dependency pitfalls discovered

These are real issues encountered during development. Future services may hit similar problems.

### Python version constraints
Many ML libraries pin old numpy versions or depend on `distutils` (removed in Python 3.12). If a model's dependency chain includes `numpy<1.26`, use Python 3.11 via deadsnakes PPA in the Dockerfile:
```dockerfile
ARG PYTHON_VERSION=3.11
RUN add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.${PYTHON_VERSION} python3.${PYTHON_VERSION}-venv python3.${PYTHON_VERSION}-dev
```

### onnx / ml-dtypes conflicts
`onnx>=1.17` requires `ml-dtypes>=0.5` which needs `numpy>=1.26`. If the model pins `numpy<1.26`, constrain onnx:
```
"onnx<1.17"
```

### setuptools / pkg_resources
`pkg_resources` was removed in `setuptools>=81`. Libraries that `from pkg_resources import ...` will fail silently or crash. Pin:
```
"setuptools<81"
```

### HuggingFace gated models
Many model repos require authentication. Always pass `HF_TOKEN` through docker-compose and check that `from_pretrained` uses it. The chatterbox library reads `os.getenv("HF_TOKEN")`.

### Transitive bloat
Some ML packages pull in large unnecessary dependencies (e.g., `chatterbox-tts` installs `gradio` at 57MB). Consider excluding them if image size matters, but be careful -- some packages import them at module level.

## Testing

- Unit tests use `httpx.AsyncClient` with FastAPI's `TestClient`.
- The engine is mocked to avoid GPU/model requirements in CI.
- Run: `uv run pytest` from the service directory.
- Lint: `uv run ruff check src/`

## Adding a new service -- checklist

1. Create `services/<name>/` directory structure (mirror chatterbox).
2. Write `pyproject.toml` with deps, ruff config, pytest config.
3. Implement `config.py`, `engine.py`, `models.py`, `audio.py` (or equivalent), `main.py`.
4. Implement routes: the OpenAI-compatible endpoint + `/v1/models` + `/health`.
5. Write `Dockerfile` with configurable ARGs/ENVs.
6. Add service to `docker-compose.yml` with configurable port, volumes, env vars.
7. Write tests with mocked engine.
8. Write `README.md` with endpoints, config, and build notes.
9. Update root `README.md` models table.
10. Run `ruff check` and `pytest` before committing.
