# gait

A gateway for the local ML models I run at home.

- OpenAI-compatible gateway with extensions for voice and model management.
- Gateway does schema validation and forwards requests to model provider services.
- Providers implement a small set of resource protocols and get routes, schemas, model listing and health checks for free.
- Written for personal use, largely with the help of agents. Use at your own risk.

## Quick start

Requires Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Set `HF_TOKEN` if you want gated HuggingFace models.

```bash
docker compose up --build -d
```

The default compose file starts all four providers. Everything routes through the gateway on port 3000:

```bash
curl http://localhost:3000/health
curl http://localhost:3000/v1/models

curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"unsloth/gemma-4-E4B-it-GGUF:Q8_0","messages":[{"role":"user","content":"Hello"}]}'

curl http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello","voice":"default"}' \
  --output speech.mp3

curl http://localhost:3000/v1/audio/transcriptions \
  -F file=@recording.wav -F model=whisper-1
```

FastAPI serves interactive API docs once the gateway is up:

- Swagger UI: http://localhost:3000/docs
- ReDoc: http://localhost:3000/redoc
- OpenAPI schema: http://localhost:3000/openapi.json

Model weights are downloaded on first use. Audio weights live under `HF_HOME` (default `~/.cache/huggingface`). LLM weights live under `LLAMA_CACHE` (falls back to `HF_HOME`).

## Endpoints

Everything under `/v1/...` follows the OpenAI REST contract unless the notes mark it as an extension. The `Resource` column groups endpoints the same way OpenAI's API reference does; the [Providers](#providers) section shows which backend serves each resource.

| Resource | Method | Path | Notes |
|----------|--------|------|-------|
| Chat completions | `POST` | `/v1/chat/completions` | Streaming supported. Transparent proxy — multimodal content parts (`input_audio`, `image_url`, …) are passed through to the backend unchanged. |
| Completions | `POST` | `/v1/completions` | Streaming supported. |
| Responses | `POST` | `/v1/responses` | Streaming supported. |
| Embeddings | `POST` | `/v1/embeddings` | |
| Audio | `POST` | `/v1/audio/speech` | TTS. Returns `wav` or `mp3`. |
| Audio | `POST` | `/v1/audio/transcriptions` | STT. Supports `json`, `text`, `srt`, `vtt`, `verbose_json`. |
| Audio | `POST` | `/v1/audio/translations` | Translate audio to English. Same formats as transcriptions. |
| Audio | `GET` | `/v1/audio/voices` | Extension. List voices. |
| Audio | `GET` | `/v1/audio/voices/{name}` | Extension. Get a single voice. |
| Audio | `POST` | `/v1/audio/voices` | Extension. Multipart: `name`, WAV `file`. |
| Audio | `DELETE` | `/v1/audio/voices/{name}` | Extension. |
| Models | `GET` | `/v1/models` | Merged list from every enabled backend. Each entry carries an extra `status` field (`unloaded` / `loading` / `loaded` / `sleeping`). |
| Models | `POST` | `/v1/models/load` | Extension. `{"model": "<id>"}`, routed to whichever provider owns the model. |
| Models | `POST` | `/v1/models/unload` | Extension. `{"model": "<id>"}`. |
| Health | `GET` | `/health` | Extension. Gateway status plus per-backend health. |

A virtual `default` voice is always present and maps to chatterbox's built-in reference clip.

## Providers

Each provider is a client class that implements one or more *resource protocols* from [`gateway/providers/protocols.py`](gateway/providers/protocols.py). Every protocol corresponds to one row group in the [Endpoints](#endpoints) table (`ChatCompletions` → Chat completions, `AudioSpeech` → `/v1/audio/speech`, and so on). At startup the gateway instantiates every provider listed in `PROVIDERS`, inspects the protocols it implements, and wires it into the matching routes automatically.

A given endpoint is served by whichever enabled provider implements its protocol.

| Provider | Kind | Backend | Protocols implemented |
|----------|------|---------|-----------------------|
| `llamacpp` | LLM | Upstream [`llama.cpp` server](https://github.com/ggml-org/llama.cpp) in router mode | Chat completions, Completions, Responses, Embeddings, Model management |
| `chatterbox` | TTS | [Chatterbox](https://github.com/resemble-ai/chatterbox) (turbo / HD / multilingual) | Audio speech, Model management |
| `whisperx` | STT | [WhisperX](https://github.com/m-bain/whisperX) | Audio transcriptions, Audio translations, Model management |
| `voices` | Local FS | Directory of `.wav` files, shared with chatterbox | Audio voices |

Enable/disable providers with the `PROVIDERS` env var (comma-separated). A provider that isn't listed is not instantiated.

Service-specific details live in each backend's README: [llamacpp](services/llamacpp/README.md), [chatterbox](services/chatterbox/README.md), [whisperx](services/whisperx/README.md).

### Adding a new provider

1. Create a client class in `gateway/providers/`, extend `BaseProvider`, and implement the resource protocols it supports (see `gateway/providers/protocols.py`).
2. Give it `name`, `url_env`, `default_url`, and a `create()` classmethod.
3. Register it in `KNOWN_PROVIDERS` in `gateway/providers/__init__.py`.
4. Add its name to `PROVIDERS` and its URL to your compose file.

The gateway auto-discovers which protocols the class implements and wires it into the matching routes. See [AGENTS.md](AGENTS.md) for the full architectural notes.

## Configuration

Defaults live in `docker-compose.yml`.

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_PORT` | `3000` | Host port for the gateway. |
| `PROVIDERS` | `llamacpp,chatterbox,whisperx,voices` | Which providers to enable. |
| `LLAMACPP_URL` | `http://llamacpp:8000` | Internal URL for the LLM backend. |
| `CHATTERBOX_URL` | `http://chatterbox:8000` | Internal URL for the TTS backend. |
| `WHISPERX_URL` | `http://whisperx:8000` | Internal URL for the STT backend. |
| `VOICES_DIR` | `./voices` | Shared voice clip directory, mounted into both gateway and chatterbox. |
| `BACKEND_TIMEOUT` | `300` | Per-request timeout to backends, in seconds. |
| `HF_HOME` | `~/.cache/huggingface` | Shared HuggingFace cache for audio models and llama.cpp. |
| `HF_TOKEN` | — | HuggingFace token for gated repos. |
| `LLAMA_*` | — | See [llamacpp](#llamacpp). |
| `CHATTERBOX_*` | — | See [chatterbox](#chatterbox). |
| `WHISPERX_*` | — | See [whisperx](#whisperx). |

### llamacpp

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODELS_MAX` | `1` | Max presets loaded simultaneously. |
| `LLAMA_IDLE_TIMEOUT` | `-1` | Seconds of idleness before the active model is put to sleep. `-1` disables. |
| `LLAMA_CACHE` | `${HF_HOME}` | Override for llama.cpp's native cache directory. |
| `LLAMACPP_WEBUI_PORT` | `8090` | Host port for llama-server's built-in web UI / raw API. |

Model-level tuning lives in [`config/llama-models.ini`](config/llama-models.ini). See the [llamacpp service README](services/llamacpp/README.md) for the preset format.

### chatterbox

| Variable | Default | Description |
|----------|---------|-------------|
| `CHATTERBOX_DEVICE` | `cuda` | Inference device. |
| `CHATTERBOX_DEFAULT_MODEL` | `chatterbox-turbo` | Preloaded at startup. |
| `CHATTERBOX_IDLE_TIMEOUT` | `300` | Unload models after N seconds idle. `0` disables. |

### whisperx

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPERX_DEVICE` | `cuda` | Inference device. |
| `WHISPERX_DEFAULT_MODEL` | `turbo` | Model used for `whisper-1`. |
| `WHISPERX_COMPUTE_TYPE` | `float16` | Model precision. |
| `WHISPERX_BATCH_SIZE` | `16` | Transcription batch size. |
| `WHISPERX_ENABLE_DIARIZATION` | `false` | Enable speaker diarization (needs `HF_TOKEN`). |
| `WHISPERX_IDLE_TIMEOUT` | `0` | Unload model after N seconds idle. `0` disables. |

## Local development

The root project is the gateway. Each backend under `services/` is an independent `uv` project with its own Python version pin.

```bash
# Gateway (Python 3.12)
uv sync --all-extras
uv run pytest
uv run ruff check gateway/
uvx ty check

# Any service (e.g. services/chatterbox)
cd services/chatterbox
uv sync --all-extras
uv run pytest
```

Tests mock providers and engines, so no GPU or model downloads are needed.
