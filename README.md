# trave

Local ML models exposed via OpenAI-compatible REST APIs.

A FastAPI gateway unifies all services behind a single port with OpenAI-compatible request/response handling. Each ML model runs as a thin Starlette backend in its own Docker container with GPU passthrough. Voice management runs as a separate lightweight service. Orchestrated with `docker compose`.

## Models

| Service | Model | Params | Type | Port | Docs |
|---------|-------|--------|------|------|------|
| [chatterbox](services/chatterbox/) | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) | 350M | TTS (English) | 8100 | [README](services/chatterbox/README.md) |
| [chatterbox](services/chatterbox/) | [Chatterbox](https://github.com/resemble-ai/chatterbox) | 500M | TTS (English) | 8100 | [README](services/chatterbox/README.md) |
| [chatterbox](services/chatterbox/) | [Chatterbox-Multilingual](https://github.com/resemble-ai/chatterbox) | 500M | TTS (23 languages) | 8100 | [README](services/chatterbox/README.md) |
| [whisperx](services/whisperx/) | [WhisperX](https://github.com/m-bain/whisperX) (large-v3 default) | 1.5B | STT | 8201 | [README](services/whisperx/README.md) |

| Service | Type | Port | Docs |
|---------|------|------|------|
| [gateway](services/gateway/) | API Gateway | 8080 | [README](services/gateway/README.md) |
| [voice](services/voice/) | Voice Management | 8300 | [README](services/voice/README.md) |

All three Chatterbox models run in a single container. The default model (`chatterbox-turbo`) is preloaded at startup; others load on demand.

## Quick start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA support
- `HF_TOKEN` environment variable set (for gated HuggingFace models)

### Run

```bash
# start all services (gateway + all backends + voice)
docker compose up --build -d

# or individual services
docker compose up --build -d chatterbox
docker compose up --build -d whisperx
docker compose up --build -d voice

# gateway starts after backends and voice are healthy
docker compose up --build -d gateway
```

Model weights are downloaded from HuggingFace on first startup and cached on the host at `~/.cache/huggingface` (configurable via `HF_HOME`).

### Try it

All services are accessible through the gateway on port 8080:

```bash
# health check (gateway aggregates backend health)
curl http://localhost:8080/health

# list all models (merged from all backends)
curl http://localhost:8080/v1/models

# generate speech (proxied to chatterbox)
curl http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello from Trave!","voice":"default"}' \
  --output speech.mp3

# transcribe audio (proxied to whisperx)
curl http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1

# translate audio to English (proxied to whisperx)
curl http://localhost:8080/v1/audio/translations \
  -F file=@foreign_audio.wav \
  -F model=whisper-1

# manage voices (proxied to voice service)
curl http://localhost:8080/v1/audio/voices
curl http://localhost:8080/v1/audio/voices \
  -F name=narrator \
  -F file=@narrator.wav
```

## Configuration

Global settings via environment variables or a `.env` file in the repo root:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | -- | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Host path for model weight cache |
| `GATEWAY_PORT` | `8080` | Host port for the API gateway |
| `CHATTERBOX_PORT` | `8100` | Host port for the chatterbox service |
| `CHATTERBOX_DEFAULT_MODEL` | `chatterbox-turbo` | Model to preload at startup |
| `WHISPERX_PORT` | `8201` | Host port for the whisperx service |
| `VOICE_PORT` | `8300` | Host port for the voice service |
| `VOICES_DIR` | `./voices` | Host path for voice reference clips (shared between chatterbox and voice service) |
| `BACKEND_TIMEOUT` | `300` | Gateway timeout for backend requests in seconds |

Per-service configuration is documented in each service's README.

## Project structure

```
trave/
  docker-compose.yml
  services/
    gateway/           # FastAPI API gateway (OpenAI-compatible API)
      src/gateway_service/
        main.py        # FastAPI app + lifespan
        config.py      # pydantic-settings
        models.py      # response schemas
        formatting.py  # WAV→MP3, segments→SRT/VTT
        clients/       # typed HTTP clients for backends
        routes/        # OpenAI-compatible endpoints
      tests/
    chatterbox/        # Chatterbox TTS backend (Starlette, GPU)
      src/chatterbox_service/
        app.py         # Starlette app (RPC endpoints)
        engine.py      # model loading + inference
        config.py      # pydantic-settings
      tests/
    whisperx/          # WhisperX STT backend (Starlette, GPU)
      src/whisperx_service/
        app.py         # Starlette app (RPC endpoints)
        engine.py      # model loading + inference
        config.py      # pydantic-settings
      tests/
    voice/             # Voice management (Starlette, no GPU)
      src/voice_service/
        app.py         # Starlette app (CRUD endpoints)
        config.py      # pydantic-settings
      tests/
```

## Adding a new service

1. Create `services/<name>/` with its own `Dockerfile`, `pyproject.toml`, and `README.md`
2. For GPU backends: implement `app.py` (Starlette) + `engine.py` + `config.py`
3. For non-GPU services: implement `app.py` (Starlette) + `config.py`
4. Expose RPC endpoints + `/models` + `/health`
5. Add service to `docker-compose.yml`
6. Add a typed client in `services/gateway/src/gateway_service/clients/`
7. Add gateway route(s) that map OpenAI API to the backend's RPC endpoints
8. Update the models table above

See [AGENTS.md](AGENTS.md) for detailed conventions and lessons learned.
