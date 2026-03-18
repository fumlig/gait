# trave

Local ML models exposed via OpenAI-compatible REST APIs.

Each model runs in its own Docker container with GPU passthrough, isolated dependencies, and standard HTTP endpoints. Orchestrated with `docker compose`.

## Models

| Service | Model | Params | Type | Port | Docs |
|---------|-------|--------|------|------|------|
| [chatterbox](services/chatterbox/) | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) | 350M | TTS | 8100 | [README](services/chatterbox/README.md) |

## Quick start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA support
- `HF_TOKEN` environment variable set (for gated HuggingFace models)

### Run

```bash
# start all services
docker compose up --build -d

# or a single service
docker compose up --build -d chatterbox
```

Model weights are downloaded from HuggingFace on first startup and cached on the host at `~/.cache/huggingface` (configurable via `HF_HOME`).

### Try it

```bash
# health check
curl http://localhost:8100/health

# generate speech
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello from Trave!","voice":"default"}' \
  --output speech.mp3
```

## Configuration

Global settings via environment variables or a `.env` file in the repo root:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | -- | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Host path for model weight cache |
| `CHATTERBOX_PORT` | `8100` | Host port for the chatterbox service |

Per-service configuration is documented in each service's README.

## Project structure

```
trave/
  docker-compose.yml
  services/
    chatterbox/        # Chatterbox-Turbo TTS
      Dockerfile
      pyproject.toml
      README.md
      voices/
      src/
      tests/
```

## Adding a new service

1. Create `services/<name>/` with its own `Dockerfile`, `pyproject.toml`, and `README.md`
2. Add the service to `docker-compose.yml`
3. Implement the appropriate OpenAI-compatible API endpoints
4. Update the models table above

See [AGENTS.md](AGENTS.md) for detailed conventions and lessons learned.
