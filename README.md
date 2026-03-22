# gait

Local ML models exposed via OpenAI-compatible REST APIs.

A FastAPI gateway handles all OpenAI protocol concerns — request validation, response formatting (WAV→MP3, segments→SRT/VTT), model list aggregation. Each ML model runs as a thin service in its own Docker container with GPU passthrough. Orchestrated with `docker compose`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported, audio interleaving) |
| `POST` | `/v1/completions` | Text completions (streaming supported) |
| `POST` | `/v1/responses` | Responses API (streaming supported) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/audio/speech` | Text-to-speech (WAV, MP3) |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text (json, text, srt, vtt, verbose_json) |
| `POST` | `/v1/audio/translations` | Translate audio to English (same formats) |
| `GET`  | `/v1/audio/voices` | List voices |
| `POST` | `/v1/audio/voices` | Create voice (multipart: name + WAV file) |
| `GET`  | `/v1/audio/voices/{name}` | Get voice |
| `DELETE` | `/v1/audio/voices/{name}` | Delete voice |
| `GET` | `/v1/models` | Merged model list from all backends (auto-refreshing) |
| `GET` | `/health` | Gateway + per-backend health status |

LLM endpoints are proxied transparently to llama.cpp. Audio endpoints convert between OpenAI API formats and backend RPC formats.

Voice clips are stored on a local directory shared with chatterbox via Docker volume. A virtual `default` voice is always present (maps to chatterbox's built-in reference clip).

When `/v1/chat/completions` includes `modalities: ["text", "audio"]` with `stream: true`, the gateway interleaves text deltas with PCM16 audio chunks synthesised by the TTS backend.

## Services

| Service | Description | Endpoints | GPU | Docs |
|---------|-------------|-----------|-----|------|
| [llamacpp](services/llamacpp/) | LLM inference (upstream llama.cpp) | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings` | ✓ | [README](services/llamacpp/README.md) |
| [chatterbox](services/chatterbox/) | Text-to-speech (Chatterbox TTS) | `/v1/audio/speech` | ✓ | [README](services/chatterbox/README.md) |
| [whisperx](services/whisperx/) | Speech-to-text (WhisperX) | `/v1/audio/transcriptions`, `/v1/audio/translations` | ✓ | [README](services/whisperx/README.md) |
| gateway (local) | Voice management | `/v1/audio/voices` | — | — |

Each backend is optional — it is enabled by setting its URL in the environment. All services contribute to `GET /v1/models` (merged model list) and `GET /health` (aggregated status).

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

### Docker Compose

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_PORT` | `3000` | Host port for the gateway |
| `HF_TOKEN` | — | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Audio model weight cache |
| `MODELS_DIR` | `./models` | LLM model cache |
| `VOICES_DIR` | `./voices` | Voice reference clips (shared between gateway and chatterbox) |
| `BACKEND_TIMEOUT` | `300` | Gateway timeout for provider requests (seconds) |

Per-service configuration is documented in each service's README: [llamacpp](services/llamacpp/README.md), [chatterbox](services/chatterbox/README.md), [whisperx](services/whisperx/README.md).

### Gateway

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `LLAMACPP_URL` | _(unset)_ | llama.cpp backend URL (unset = disabled) |
| `CHATTERBOX_URL` | _(unset)_ | Chatterbox backend URL (unset = disabled) |
| `WHISPERX_URL` | _(unset)_ | WhisperX backend URL (unset = disabled) |
| `VOICES_DIR` | _(unset)_ | Voice clips directory (unset = voice management disabled) |
| `BACKEND_TIMEOUT` | `300` | Backend request timeout (seconds) |

## Local development

The root project is the gateway. Backend services live under `services/` as independent Python projects.

```bash
# Gateway
uv sync --all-extras        # install deps + dev tools
uv run pytest tests/test_api.py  # unit tests (mocked providers, no GPU)
uv run ruff check gateway/  # lint

# Scripts (e.g. clean-voice)
uv sync --extra scripts     # install script dependencies

# Services (from their directory, e.g. services/chatterbox/)
uv sync --all-extras
uv run pytest
uv run ruff check chatterbox_service/
```

## Provider client system

Each OpenAI resource group has a protocol in `gateway/providers/protocols.py`. Provider clients implement the protocols they support. At startup the gateway iterates `KNOWN_PROVIDERS`, instantiates each client whose env var is set, and uses `isinstance` checks to wire `app.state` slots automatically.

To add a new provider: create a client class in `gateway/providers/`, extend `BaseProvider`, implement the relevant protocols, register it in `KNOWN_PROVIDERS`. See [AGENTS.md](AGENTS.md) for the full checklist and architecture details.
