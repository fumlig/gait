# trave

Local ML models exposed via OpenAI-compatible REST APIs.

Each model runs in its own Docker container with GPU passthrough, isolated dependencies, and standard HTTP endpoints. A FastAPI gateway unifies all services behind a single port. Orchestrated with `docker compose`.

## Models

| Service | Model | Params | Type | Port | Docs |
|---------|-------|--------|------|------|------|
| [chatterbox](services/chatterbox/) | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) | 350M | TTS | 8100 | [README](services/chatterbox/README.md) |
| [whisperx](services/whisperx/) | [WhisperX](https://github.com/m-bain/whisperX) (large-v3 default) | 1.5B | STT | 8201 | [README](services/whisperx/README.md) |
| [gateway](services/gateway/) | -- | -- | API Gateway | 8080 | [README](services/gateway/README.md) |

## Quick start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA support
- `HF_TOKEN` environment variable set (for gated HuggingFace models)

### Run

```bash
# start all services (gateway + all backends)
docker compose up --build -d

# or individual services
docker compose up --build -d chatterbox
docker compose up --build -d whisperx

# gateway only starts after backends are healthy
docker compose up --build -d gateway
```

Model weights are downloaded from HuggingFace on first startup and cached on the host at `~/.cache/huggingface` (configurable via `HF_HOME`).

### Try it

All services are accessible directly on their own ports, or through the gateway on port 8080:

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

# direct access to services also works
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Direct access!","voice":"default"}' \
  --output speech.mp3
```

## Configuration

Global settings via environment variables or a `.env` file in the repo root:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | -- | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Host path for model weight cache |
| `GATEWAY_PORT` | `8080` | Host port for the API gateway |
| `CHATTERBOX_PORT` | `8100` | Host port for the chatterbox service |
| `WHISPERX_PORT` | `8201` | Host port for the whisperx service |
| `PROXY_TIMEOUT` | `300` | Gateway proxy timeout in seconds |

Per-service configuration is documented in each service's README.

## Project structure

```
trave/
  docker-compose.yml
  services/
    gateway/           # FastAPI API gateway (reverse proxy)
      Dockerfile
      pyproject.toml
      README.md
      src/
      tests/
    chatterbox/        # Chatterbox-Turbo TTS
      Dockerfile
      pyproject.toml
      README.md
      voices/
      src/
      tests/
    whisperx/          # WhisperX STT (transcription + translation)
      Dockerfile
      pyproject.toml
      README.md
      src/
      tests/
```

## Adding a new service

1. Create `services/<name>/` with its own `Dockerfile`, `pyproject.toml`, and `README.md`
2. Add the service to `docker-compose.yml`
3. Implement the appropriate OpenAI-compatible API endpoints
4. Add proxy route in the gateway service
5. Update the models table above

See [AGENTS.md](AGENTS.md) for detailed conventions and lessons learned.
