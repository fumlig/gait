# AGENTS.md

Context and conventions for AI agents working on this repository.

## Project overview

Trave exposes local ML models via OpenAI-compatible REST APIs. A FastAPI gateway handles all OpenAI protocol concerns (request validation, response formatting, model listing). Each ML model runs as a thin backend service in its own Docker container with GPU passthrough, exposing low-level RPC-style endpoints via Starlette. Voice management runs as a separate lightweight service sharing a volume with the TTS backend.

## Architecture

```
                   ┌────────────────────┐
                   │      Gateway       │  FastAPI, OpenAI-compatible API
                   │    (port 8080)     │  Format conversion, model list cache
                   │                    │  Voice management (local filesystem)
                   └──┬──────┬──────┬──┘
                      │      │      │
           ┌──────────┘      │      └──────────────┐
           ▼                 ▼                      ▼
   ┌──────────────┐  ┌────────────┐  ┌──────────────┐
   │  llama.cpp   │  │ Chatterbox │  │   WhisperX   │
   │  (upstream)  │  │ Starlette  │  │  Starlette   │
   │  GPU / CUDA  │  │ GPU / CUDA │  │  GPU / CUDA  │
   │  /v1/chat/.. │  │ /synthesize│  │  /transcribe │
   │  /v1/complet.│  │ /models    │  │  /translate  │
   │  /v1/embeddi.│  │ /health    │  │  /models     │
   │  /v1/models  │  │            │  │  /health     │
   │  /health     │  └────────────┘  └──────────────┘
   └──────────────┘
```

- **Gateway** owns the OpenAI API contract: request/response schemas, format conversion (WAV→MP3, segments→SRT/VTT), model list caching. Also handles voice management directly on a local directory (shared volume with chatterbox).
- **llamacpp** runs the upstream llama.cpp server image directly. Already OpenAI-compatible — the gateway proxies requests transparently. No custom application code.
- **Backends** (chatterbox, whisperx) are thin Starlette apps that expose raw RPC endpoints. They return raw formats (WAV audio, JSON segments). No FastAPI, no trave-common.
- **Clients** live in the gateway's `backends/` module. Each backend client extends `BaseBackend` (for HTTP backends) and implements resource protocols from `protocols.py`. The gateway auto-discovers which clients to instantiate based on environment variables (`LLAMACPP_URL`, `CHATTERBOX_URL`, `WHISPERX_URL`, `VOICES_DIR`).

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
        protocols.py           # Resource protocols (ChatCompletions, AudioSpeech, etc.)
        backends/
          base.py              # BaseBackend base class (health, models, create)
          __init__.py          # KNOWN_BACKENDS registry, Registerable protocol
          llamacpp.py          # LlamacppClient — transparent proxy for llama.cpp
          chatterbox.py        # ChatterboxClient — TTS via chatterbox
          whisperx.py          # WhisperxClient — STT via whisperx
          voice.py             # VoiceClient — local filesystem voice management
        routes/
          health.py            # GET /health
          models.py            # GET /v1/models
          completions.py     # POST /v1/completions
          responses.py       # POST /v1/responses
          embeddings.py      # POST /v1/embeddings
          chat/
            completions.py   # POST /v1/chat/completions
          audio/
            speech.py          # POST /v1/audio/speech
            transcriptions.py  # POST /v1/audio/transcriptions
            translations.py    # POST /v1/audio/translations
            voices.py          # voice management (proxied to voice service)
      tests/
        test_api.py            # pytest + httpx, mocked clients
    llamacpp/                  # LLM backend (upstream llama.cpp image)
      Dockerfile               # thin wrapper around ghcr.io/ggml-org/llama.cpp
      README.md                # service docs
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

```

## Conventions

### Services
- One folder per service under `services/`.
- Each service is an independent Python project with its own `pyproject.toml`.
- No shared library -- each service defines its own types.
- Use `uv` for dependency management, `ruff` for linting, `ty` for type checking, `pytest` for testing.
- Every service gets its own `README.md` with endpoint docs, config table, and build notes.

### Backend services (chatterbox, whisperx)
- Starlette apps (not FastAPI). Single `app.py` + `engine.py`.
- Expose raw RPC endpoints -- no OpenAI request/response schemas.
- Return raw formats: audio/wav for TTS, JSON segments for STT.
- `GET /models` and `GET /health` are kept as HTTP endpoints.

### Gateway
- FastAPI app. Owns all OpenAI API contract concerns.
- `backends/` module contains typed client classes, one per backend type. `protocols.py` defines resource protocols (one per OpenAI endpoint group).
- HTTP backends extend `BaseBackend` (shared health/models) and implement resource protocols (e.g. `AudioSpeech`, `ChatCompletions`). The `VoiceClient` implements `AudioVoices` directly.
- Each class declares `env_var` and a `create` classmethod. At startup the gateway iterates `KNOWN_BACKENDS`, checks `os.environ` for each class's env var, and calls `create` to instantiate only those whose variable is set.
- Protocol wiring is automatic: the gateway uses `isinstance` checks against each resource protocol to determine which `app.state` slot a client fills. No explicit capabilities list needed.
- Route modules mirror OpenAI API resource grouping: `chat/`, `completions`, `responses`, `embeddings`, `audio/`.
- Models are fetched from each backend's `fetch_models()` at startup and cached on `app.state.models`.
- Format conversion (WAV→MP3, segments→SRT/VTT) happens in the gateway.

### Voice management
- Voice CRUD is handled inside the gateway via a local filesystem client (`backends/voice.py`).
- The `VOICES_DIR` directory is shared with chatterbox via a Docker volume for voice resolution.
- No separate service or container — the gateway reads/writes WAV files directly.

### API design
- The gateway matches the OpenAI API contract for the relevant modality (TTS, STT).
- Gateway implements `GET /v1/models` (merged from backends) and `GET /health` (aggregated).
- Backends implement `GET /models` and `GET /health` (no `/v1/` prefix).
- Accept and ignore `Authorization` headers (local use, no auth).
- Return proper HTTP error codes: 400 for bad input, 503 for model not loaded, 500 for inference failures.

### Docker
- Base image: `nvidia/cuda:<version>-runtime-ubuntu<version>` for GPU services.
- Base image: `python:3.12-slim` for non-GPU services (gateway).
- Use `ARG` for build-time configurability (Python version, CUDA version).
- Use `ENV` with defaults for runtime configurability (port, host, device).
- The `CMD` should reference env vars so they can be overridden at runtime.
- Layer caching: copy `pyproject.toml` **and `uv.lock`** first, then `uv sync --frozen` before copying source.
- Use `--mount=type=cache,target=/root/.cache/uv` on all `RUN uv sync` steps to persist the download cache across builds.
- Include `curl` in the image for healthcheck.
- The root `.dockerignore` excludes `.venv/`, `.git/`, tests, and other dev artifacts from the build context.

### docker-compose.yml
- Only the gateway exposes a host port (`GATEWAY_PORT`). Backend services communicate over the internal Docker network.
- Every volume path and env var should be configurable via `${VAR:-default}` syntax.
- Mount `HF_HOME` from host for audio model weight caching; `MODELS_DIR` (or `LLAMA_CACHE`) for LLM weights.
- Pass `HF_TOKEN` for gated model access.
- Use `deploy.resources.reservations.devices` for GPU passthrough (GPU services only).
- Set `restart: unless-stopped` and a healthcheck with generous `start_period` (models take time to load).
- Shared `VOICES_DIR` volume between gateway and chatterbox.

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
- Type check: `uvx ty check` from the service directory.

## Adding a new service -- checklist

1. Create `services/<name>/` directory structure (mirror chatterbox or voice).
2. Write `pyproject.toml` with deps, ruff config, pytest config.
3. For GPU backends: implement `config.py`, `engine.py`, `app.py` (Starlette).
4. For non-GPU services: implement `config.py`, `app.py` (Starlette).
5. Expose RPC endpoints + `/models` + `/health`.
6. Write `Dockerfile` with configurable ARGs/ENVs.
7. Add service to `docker-compose.yml` with configurable port, volumes, env vars.
8. Add a typed client in `services/gateway/src/gateway_service/backends/` that extends `BaseBackend` and implements the relevant resource protocols from `protocols.py`. Add a `create` classmethod and register it in `KNOWN_BACKENDS` in `backends/__init__.py`.
9. Add gateway route(s) that map OpenAI API to the backend's RPC endpoints.
10. Write tests with mocked engine (backend) or mocked client (gateway).
11. Write `README.md` with endpoints, config, and build notes.
12. Update root `README.md` models table.
13. Run `ruff check`, `ty check`, and `pytest` before committing.

## Local development

Each service is an independent `uv` project (no workspace). Services can be developed and tested locally without Docker.

### Python versions
- Each service has a `.python-version` file that pins its required Python version.
- `uv` reads this file automatically and downloads the correct interpreter if needed.
- Current pins: chatterbox=3.11, whisperx=3.11, gateway=3.12.
- GPU services use Python 3.11 because ML libraries often depend on `distutils` (removed in 3.12) or pin `numpy<1.26`.

### Setup
From any service directory:
```bash
uv sync --all-extras   # creates .venv with the pinned Python, installs all deps + dev tools
uv run ruff check src/ # lint
uvx ty check           # type check (uses .venv for third-party stubs)
uv run pytest          # tests (engines are mocked, no GPU needed)
```

### IDE configuration
Each service has its own `.venv/`. Point your IDE's Python interpreter to the service you're working on:
- **Zed**: Project-level `.zed/settings.json` configures `ty` (type checking) and `ruff` (linting + formatting) as language servers. Both run on every keystroke / save. Requires `ty` and `ruff` on `$PATH` (`uv tool install ty ruff`). Install no extra extensions — the config uses the binaries directly.
- VS Code: set `python.defaultInterpreterPath` in `.vscode/settings.json`, or use the Python interpreter picker per workspace folder.
- PyCharm: configure each service directory as a separate module with its own interpreter.

### Caveats
- `.venv/` directories are gitignored. Always run `uv sync --all-extras` after cloning.
- Do not use the system Python directly. The system may run a different version (e.g. 3.14) which will cause import or build failures.
- If a `.venv` was created with the wrong Python version, delete it and re-run `uv sync --all-extras`.

## Hosting as a self-hosted service

The stack is designed to run locally as a persistent service that survives reboots.

### Systemd integration
A `trave.service` unit file is provided in the repo root. Install it once:
```bash
sudo cp trave.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable (auto-start on boot):
```bash
sudo systemctl enable trave
```

Disable (stop auto-starting on boot):
```bash
sudo systemctl disable trave
```

Manual start/stop/status:
```bash
sudo systemctl start trave
sudo systemctl stop trave
systemctl status trave
```

The unit uses `docker compose up -d --wait` which blocks until all healthchecks pass, and `docker compose down` for clean shutdown. `TimeoutStartSec=600` gives enough time for first-run model downloads.

### Docker build caching
- A root `.dockerignore` excludes `.venv/` directories (13+ GB), `.git/`, test files, and other artifacts from the build context.
- Dockerfiles use `--mount=type=cache,target=/root/.cache/uv` so uv's download cache persists across builds. Rebuilds after source-only changes skip all dependency downloads.
- `uv.lock` is copied alongside `pyproject.toml` and `--frozen` is passed to `uv sync`, ensuring reproducible, faster resolution without re-solving.
- BuildKit is required (default with Docker Compose v2 / Docker Engine 23+).
