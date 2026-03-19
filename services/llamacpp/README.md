# llamacpp

Thin wrapper around the official [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) Docker image. Provides OpenAI-compatible LLM inference with GPU acceleration via CUDA.

This service has no custom application code — it runs the upstream `llama-server` binary directly. The gateway proxies all language modelling requests to it transparently.

## Endpoints

All endpoints are provided by llama-server and are already OpenAI-compatible:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | Text completions (streaming supported) |
| `POST` | `/v1/responses` | Responses API (streaming supported) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Server health status |

## Configuration

All configuration is via environment variables using the `LLAMA_ARG_` prefix, which maps to llama-server CLI flags. See the [llama-server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_ARG_MODEL` | -- | Path to a local GGUF model file |
| `LLAMA_ARG_HF_REPO` | -- | HuggingFace repo to download (e.g. `Qwen/Qwen3-4B-GGUF:Q4_K_M`) |
| `LLAMA_ARG_HOST` | `0.0.0.0` | Server bind address |
| `LLAMA_ARG_PORT` | `8000` | Server bind port |
| `LLAMA_ARG_CTX_SIZE` | `4096` | Context window size |
| `LLAMA_ARG_N_GPU_LAYERS` | `99` | Number of layers to offload to GPU (-1 or 99 = all) |
| `LLAMA_ARG_FLASH_ATTN` | -- | Enable flash attention (set to any value) |
| `LLAMA_ARG_CONT_BATCHING` | -- | Enable continuous batching (set to any value) |
| `LLAMA_ARG_THREADS` | -- | Number of threads for generation |
| `LLAMA_ARG_PARALLEL` | -- | Number of parallel sequences to decode |
| `LLAMA_ARG_ALIAS` | -- | Model name alias returned in API responses |

## Model files

Models can be loaded in two ways:

**Option 1: Auto-download from HuggingFace (recommended)**

The default model is [Qwen 3.5 9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (Q8_0, ~9.5 GB). Set `LLAMA_HF_REPO` to use a different model. Models are cached in `MODELS_DIR` (default: `./models` on host, `/root/.cache/llama.cpp` in container). The legacy `LLAMA_CACHE` variable is also accepted.

```bash
# default model (Qwen 3.5 9B Q8_0)
docker compose up llamacpp

# use a different model
LLAMA_HF_REPO=Qwen/Qwen3-4B-GGUF:Q4_K_M docker compose up llamacpp
```

**Option 2: Local GGUF file**

Mount a directory containing GGUF files and set `LLAMA_MODEL` to the container path:

```bash
LLAMA_MODEL=/root/.cache/llama.cpp/model.gguf docker compose up llamacpp
```

## Build

```bash
# Default: CUDA server image
docker compose build llamacpp

# CPU-only (override the build arg)
docker compose build --build-arg LLAMA_SERVER_TAG=server llamacpp
```

## Development

No local development setup is needed — this service runs the upstream binary directly. For testing, use the Docker image:

```bash
docker compose up --build llamacpp
curl http://localhost:8400/health
curl http://localhost:8400/v1/models
```
