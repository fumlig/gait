# AGENTS.md

Context and conventions for AI agents working on this repository.

## Project overview

Trave exposes local ML models via OpenAI-compatible REST APIs. A FastAPI gateway handles all OpenAI protocol concerns (request validation, response formatting, model listing). Each ML model runs as a thin backend service in its own Docker container with GPU passthrough, exposing low-level RPC-style endpoints via Starlette. Voice management runs as a separate lightweight service sharing a volume with the TTS backend.

## Architecture

```
                   ┌───────────────────┐
                   │     Gateway       │  FastAPI, OpenAI-compatible API
                   │   (port 8080)     │  Format conversion, model list cache
                   └──┬──────┬──────┬──┘
                      │      │      │
           ┌──────────┘      │      └──────────┐
           ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  Chatterbox  │  │   WhisperX   │  │    Voice     │
   │  Starlette   │  │  Starlette   │  │  Starlette   │
   │  GPU / CUDA  │  │  GPU / CUDA  │  │  No GPU      │
   │  /synthesize │  │  /transcribe │  │  /voices     │
   │  /models     │  │  /translate  │  │  /health     │
   │  /health     │  │  /models     │  │              │
   └──────────────┘  │  /health     │  └──────────────┘
                     └──────────────┘
```

- **Gateway** owns the OpenAI API contract: request/response schemas, format conversion (WAV→MP3, segments→SRT/VTT), model list caching.
- **Backends** (chatterbox, whisperx) are thin Starlette apps that expose raw RPC endpoints. They return raw formats (WAV audio, JSON segments). No FastAPI, no trave-common.
- **Voice service** manages WAV reference clips on a shared volume. No GPU, no ML dependencies.
- **Clients** live in the gateway's `clients/` module. Each client is a typed HTTP wrapper for one backend.

## Repository structure

```
trave/
  docker-compose.yml          # orchestrates all services
  .env                        # (optional) host-level env vars
  AGENTS.md                   # this file
  services/
    gateway/                   # FastAPI OpenAI-compatible gateway
      src/gateway_service/
        main.py                # FastAPI app + lifespan
        config.py              # pydantic-settings
        models.py              # shared response schemas
        formatting.py          # response format conversion
        clients/
          speech.py            # typed HTTP client for chatterbox
          transcription.py     # typed HTTP client for whisperx
          voice.py             # typed HTTP client for voice service
        routes/
          health.py            # GET /health
          models.py            # GET /v1/models
          audio/
            speech.py          # POST /v1/audio/speech
            transcriptions.py  # POST /v1/audio/transcriptions
            translations.py    # POST /v1/audio/translations
            voices.py          # voice management (proxied to voice service)
      tests/
        test_api.py            # pytest + httpx, mocked clients
    chatterbox/                # TTS backend (Starlette)
      src/chatterbox_service/
        app.py                 # Starlette app (RPC endpoints)
        config.py              # pydantic-settings
        engine.py              # model loading + inference singleton
      tests/
        test_api.py            # pytest + httpx, mocked engine
    whisperx/                  # STT backend (Starlette)
      src/whisperx_service/
        app.py                 # Starlette app (RPC endpoints)
        config.py              # pydantic-settings
        engine.py              # model loading + inference singleton
      tests/
        test_api.py            # pytest + httpx, mocked engine
    voice/                     # Voice management (Starlette, no GPU)
      src/voice_service/
        app.py                 # Starlette app (CRUD endpoints)
        config.py              # pydantic-settings
      tests/
        test_api.py            # pytest + httpx
```

## Conventions

### Services
- One folder per service under `services/`.
- Each service is an independent Python project with its own `pyproject.toml`.
- No shared library -- each service defines its own types.
- Use `uv` for dependency management, `ruff` for linting, `pytest` for testing.
- Every service gets its own `README.md` with endpoint docs, config table, and build notes.

### Backend services (chatterbox, whisperx)
- Starlette apps (not FastAPI). Single `app.py` + `engine.py`.
- Expose raw RPC endpoints -- no OpenAI request/response schemas.
- Return raw formats: audio/wav for TTS, JSON segments for STT.
- `GET /models` and `GET /health` are kept as HTTP endpoints.

### Gateway
- FastAPI app. Owns all OpenAI API contract concerns.
- `clients/` module contains typed HTTP wrappers, one per backend.
- Clients are stored on `app.state` (e.g. `request.app.state.speech_client`). No registry, no dynamic discovery.
- Backend URLs are static env vars: `SPEECH_URL`, `TRANSCRIPTION_URL`, `VOICE_URL`.
- Models are fetched from backends once at startup and cached on `app.state.models`.
- Format conversion (WAV→MP3, segments→SRT/VTT) happens in the gateway.

### Voice service
- Starlette app. Manages WAV reference clips on a shared volume.
- No GPU, no ML dependencies. Lightweight Python 3.12 base image.
- Shares `VOICES_DIR` volume with chatterbox for voice resolution.

### API design
- The gateway matches the OpenAI API contract for the relevant modality (TTS, STT).
- Gateway implements `GET /v1/models` (merged from backends) and `GET /health` (aggregated).
- Backends implement `GET /models` and `GET /health` (no `/v1/` prefix).
- Accept and ignore `Authorization` headers (local use, no auth).
- Return proper HTTP error codes: 400 for bad input, 503 for model not loaded, 500 for inference failures.

### Docker
- Base image: `nvidia/cuda:<version>-runtime-ubuntu<version>` for GPU services.
- Base image: `python:3.12-slim` for non-GPU services (voice).
- Use `ARG` for build-time configurability (Python version, CUDA version).
- Use `ENV` with defaults for runtime configurability (port, host, device).
- The `CMD` should reference env vars so they can be overridden at runtime.
- Layer caching: copy `pyproject.toml` and `uv sync` before copying source.
- Include `curl` in the image for healthcheck.

### docker-compose.yml
- Every port, volume path, and env var should be configurable via `${VAR:-default}` syntax.
- Mount `HF_HOME` from host for model weight caching.
- Pass `HF_TOKEN` for gated model access.
- Use `deploy.resources.reservations.devices` for GPU passthrough (GPU services only).
- Set `restart: unless-stopped` and a healthcheck with generous `start_period` (models take time to load).
- Shared `VOICES_DIR` volume between chatterbox and voice service.

### Configuration (config.py)
- Use `pydantic-settings` with `BaseSettings`.
- No env prefix -- variables map directly (e.g., `DEVICE=cuda`).
- Expose all tunables: device, host, port, data dirs, limits.

### Engine pattern (GPU backends only)
- Module-level singleton instance created at import time.
- `load()` called during Starlette lifespan startup, `unload()` on shutdown.
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

- Unit tests use `httpx.AsyncClient` against the Starlette/FastAPI test transport.
- Backends mock the engine singleton to avoid GPU/model requirements in CI.
- Gateway tests mock the client classes (no real HTTP calls to backends).
- Run: `uv run pytest` from the service directory.
- Lint: `uv run ruff check src/`

## Adding a new service -- checklist

1. Create `services/<name>/` directory structure (mirror chatterbox or voice).
2. Write `pyproject.toml` with deps, ruff config, pytest config.
3. For GPU backends: implement `config.py`, `engine.py`, `app.py` (Starlette).
4. For non-GPU services: implement `config.py`, `app.py` (Starlette).
5. Expose RPC endpoints + `/models` + `/health`.
6. Write `Dockerfile` with configurable ARGs/ENVs.
7. Add service to `docker-compose.yml` with configurable port, volumes, env vars.
8. Add a typed client in `services/gateway/src/gateway_service/clients/`.
9. Add gateway route(s) that map OpenAI API to the backend's RPC endpoints.
10. Write tests with mocked engine (backend) or mocked client (gateway).
11. Write `README.md` with endpoints, config, and build notes.
12. Update root `README.md` models table.
13. Run `ruff check` and `pytest` before committing.

## Local development

Each service is an independent `uv` project (no workspace). Services can be developed and tested locally without Docker.

### Python versions
- Each service has a `.python-version` file that pins its required Python version.
- `uv` reads this file automatically and downloads the correct interpreter if needed.
- Current pins: chatterbox=3.11, whisperx=3.11, gateway=3.12, voice=3.12.
- GPU services use Python 3.11 because ML libraries often depend on `distutils` (removed in 3.12) or pin `numpy<1.26`.

### Setup
From any service directory:
```bash
uv sync --all-extras   # creates .venv with the pinned Python, installs all deps + dev tools
uv run ruff check src/ # lint
uv run pytest          # tests (engines are mocked, no GPU needed)
```

### IDE configuration
Each service has its own `.venv/`. Point your IDE's Python interpreter to the service you're working on:
- VS Code: set `python.defaultInterpreterPath` in `.vscode/settings.json`, or use the Python interpreter picker per workspace folder.
- PyCharm: configure each service directory as a separate module with its own interpreter.

### Caveats
- `.venv/` directories are gitignored. Always run `uv sync --all-extras` after cloning.
- Do not use the system Python directly. The system may run a different version (e.g. 3.14) which will cause import or build failures.
- If a `.venv` was created with the wrong Python version, delete it and re-run `uv sync --all-extras`.
