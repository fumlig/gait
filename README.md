# gait

Local ML models exposed via OpenAI-compatible REST APIs.

A FastAPI gateway unifies all providers behind a single port. Each ML model runs as a thin service in its own Docker container with GPU passthrough. Orchestrated with `docker compose`.

## Providers

| Provider | Service | Resources | Endpoints | GPU |
|----------|---------|-----------|-----------|-----|
| llamacpp ([services/llamacpp/](services/llamacpp/)) | llamacpp | Chat Completions, Completions, Responses, Embeddings | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings` | ✓ |
| chatterbox ([services/chatterbox/](services/chatterbox/)) | chatterbox | Audio Speech | `/v1/audio/speech` | ✓ |
| whisperx ([services/whisperx/](services/whisperx/)) | whisperx | Audio Transcriptions, Audio Translations | `/v1/audio/transcriptions`, `/v1/audio/translations` | ✓ |
| voices | gateway (local) | Audio Voices | `/v1/audio/voices` | — |

All services also contribute to `GET /v1/models` (merged model list) and `GET /health` (aggregated status).

## Quick start

Requires Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and `HF_TOKEN` set for gated HuggingFace models.

```bash
docker compose up --build -d
```

Model weights are downloaded on first startup. Audio model weights are cached at `~/.cache/huggingface` (configurable via `HF_HOME`). LLM weights are cached at `MODELS_DIR` (default: `./models`).

All services are accessible through the gateway on port 3000 (configurable via `GATEWAY_PORT`):

```bash
curl http://localhost:3000/health
curl http://localhost:3000/v1/models
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello!"}]}'
curl http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello!","voice":"default"}' \
  --output speech.mp3
curl http://localhost:3000/v1/audio/transcriptions \
  -F file=@recording.wav -F model=whisper-1
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_PORT` | `3000` | Host port for the gateway |
| `HF_TOKEN` | -- | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Audio model weight cache |
| `MODELS_DIR` | `./models` | LLM model cache |
| `VOICES_DIR` | `./voices` | Voice reference clips (shared between gateway and chatterbox) |
| `BACKEND_TIMEOUT` | `300` | Gateway timeout for provider requests (seconds) |

Per-service configuration is documented in each service's README: [gateway](gateway/), [llamacpp](services/llamacpp/), [chatterbox](services/chatterbox/), [whisperx](services/whisperx/).

See [AGENTS.md](AGENTS.md) for architecture details and development conventions.
