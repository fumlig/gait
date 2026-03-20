# llamacpp

Thin wrapper around the official [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) Docker image. No custom application code — runs the upstream `llama-server` binary directly. The gateway proxies all requests transparently.

## Endpoints

All provided by llama-server (already OpenAI-compatible):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | Text completions (streaming supported) |
| `POST` | `/v1/responses` | Responses API (streaming supported) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `GET` | `/v1/models` | List loaded models |
| `GET` | `/health` | Server health |

## Configuration

All via `LLAMA_ARG_` environment variables mapping to llama-server CLI flags. See the [llama-server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for all options.

Key variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_ARG_HF_REPO` | -- | HuggingFace repo to download |
| `LLAMA_ARG_MODEL` | -- | Path to a local GGUF file |
| `LLAMA_ARG_CTX_SIZE` | `4096` | Context window size |
| `LLAMA_ARG_N_GPU_LAYERS` | `99` | GPU layers to offload |

## Model files

**Auto-download** (default): Set `LLAMA_HF_REPO`. Cached in `MODELS_DIR` (default: `./models`).

**Local file**: Set `LLAMA_MODEL` to a container path.
