# trave

Local ML models exposed via OpenAI-compatible REST APIs.

A FastAPI gateway unifies all services behind a single port with OpenAI-compatible request/response handling. Each ML model runs as a thin Starlette backend in its own Docker container with GPU passthrough. Voice management is handled directly by the gateway on a shared volume. Orchestrated with `docker compose`.

## Models

| Service | Model | Params | Type | Docs |
|---------|-------|--------|------|------|
| [llamacpp](services/llamacpp/) | [llama.cpp](https://github.com/ggml-org/llama.cpp) (any GGUF model) | varies | LLM | [README](services/llamacpp/README.md) |
| [chatterbox](services/chatterbox/) | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) | 350M | TTS (English) | [README](services/chatterbox/README.md) |
| [chatterbox](services/chatterbox/) | [Chatterbox](https://github.com/resemble-ai/chatterbox) | 500M | TTS (English) | [README](services/chatterbox/README.md) |
| [chatterbox](services/chatterbox/) | [Chatterbox-Multilingual](https://github.com/resemble-ai/chatterbox) | 500M | TTS (23 languages) | [README](services/chatterbox/README.md) |
| [whisperx](services/whisperx/) | [WhisperX](https://github.com/m-bain/whisperX) (large-v3 default) | 1.5B | STT | [README](services/whisperx/README.md) |

| Service | Type | Docs |
|---------|------|------|
| [gateway](services/gateway/) | API Gateway (port 8080) | [README](services/gateway/README.md) |

The llama.cpp service runs any GGUF model. The default is [Qwen 3.5 9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (Q8_0 quantization, ~9.5 GB), which fits on an RTX 3090 alongside the TTS and STT models. Set `LLAMA_HF_REPO` to use a different model, or point `LLAMA_MODEL` at a local GGUF file. All three Chatterbox models run in a single container. The default model (`chatterbox-turbo`) is preloaded at startup; others load on demand.

## Quick start

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU with CUDA support
- `HF_TOKEN` environment variable set (for gated HuggingFace models)

### Run

```bash
# start all services (gateway + all backends)
# downloads Qwen 3.5 9B Q8_0 by default on first startup
docker compose up --build -d

# or individual services
docker compose up --build -d chatterbox
docker compose up --build -d whisperx
docker compose up --build -d llamacpp

# gateway starts after audio backends are healthy
# (llamacpp is not a hard dependency — gateway handles it being absent)
docker compose up --build -d gateway

# use a different LLM model
LLAMA_HF_REPO=Qwen/Qwen3-4B-GGUF:Q4_K_M docker compose up --build -d llamacpp
```

Model weights are downloaded from HuggingFace on first startup. Audio model weights are cached at `~/.cache/huggingface` (configurable via `HF_HOME`). LLM weights are cached at `MODELS_DIR` (default: `./models`; the `LLAMA_CACHE` variable is also accepted).

### Try it

All services are accessible through the gateway on port 8080:

```bash
# health check (gateway aggregates backend health)
curl http://localhost:8080/health

# list all models (merged from all backends)
curl http://localhost:8080/v1/models

# chat completion (proxied to llama.cpp)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello!"}]}'

# chat completion with streaming
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# text completion (proxied to llama.cpp)
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","prompt":"Once upon a time"}'

# responses API (proxied to llama.cpp)
curl http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","input":"Write a haiku about coding"}'

# embeddings (proxied to llama.cpp)
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"my-model","input":"Hello world"}'

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

# manage voices (handled by gateway, stored on shared volume)
curl http://localhost:8080/v1/audio/voices
curl http://localhost:8080/v1/audio/voices \
  -F name=narrator \
  -F file=@narrator.wav
```

## Configuration

Global settings via environment variables or a `.env` file in the repo root:

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_PORT` | `8080` | Host port for the API gateway (only exposed port) |
| `HF_TOKEN` | -- | HuggingFace API token |
| `HF_HOME` | `~/.cache/huggingface` | Host path for audio model weight cache |
| `LLAMA_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q8_0` | HuggingFace repo to download |
| `LLAMA_MODEL` | -- | Path to a local GGUF model file (alternative to `LLAMA_HF_REPO`) |
| `LLAMA_CTX_SIZE` | `4096` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `99` | Number of layers to offload to GPU |
| `MODELS_DIR` | `./models` | Host path for LLM model cache (also accepts `LLAMA_CACHE`) |
| `CHATTERBOX_DEFAULT_MODEL` | `chatterbox-turbo` | Chatterbox model to preload at startup |
| `VOICES_DIR` | `./voices` | Host path for voice reference clips (shared between gateway and chatterbox) |
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
    llamacpp/          # llama.cpp LLM backend (upstream image, GPU)
      Dockerfile       # thin wrapper around ghcr.io/ggml-org/llama.cpp
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
