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
| `LLAMA_HF_REPO` | `unsloth/gemma-4-E4B-it-GGUF:Q8_0` | HuggingFace repo (`<user>/<model>[:quant]`) |
| `LLAMA_MODEL` | -- | Path to a local GGUF file (overrides `LLAMA_HF_REPO`) |
| `LLAMA_MMPROJ` | -- | Local multimodal projector file (overrides auto-detect) |
| `LLAMA_MMPROJ_URL` | -- | URL for the multimodal projector (overrides auto-detect) |
| `LLAMA_CTX_SIZE` | `16384` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `32` | GPU layers to offload |
| `LLAMA_REASONING_FORMAT` | `deepseek` | How to parse thinking tokens (`deepseek`, `none`) |
| `LLAMA_TEMP` / `LLAMA_TOP_P` / `LLAMA_TOP_K` | `1.0` / `0.95` / `64` | Sampler defaults (Gemma 4 recommended) |
| `LLAMACPP_WEBUI_PORT` | `8080` | Host port for the llama-server web UI / raw API |

## Model files

**Auto-download** (default): Set `LLAMA_HF_REPO`. Cached in the HuggingFace hub cache (`HF_HOME`). The llama.cpp native cache (`LLAMA_CACHE`) is also mounted. When `-hf` is used, `llama-server` also auto-downloads any sibling `mmproj-*.gguf` from the same repo (controlled by `--mmproj-auto`, on by default), which is what enables vision/audio input for multimodal models like Gemma 4 E2B/E4B.

**Local file**: Set `LLAMA_MODEL` to a container path. For multimodal models you must also set `LLAMA_MMPROJ` (or `LLAMA_MMPROJ_URL`) explicitly — auto-detect only works with `-hf`.

## Native web UI

`llama-server` ships with a built-in chat web UI at `/`, served on the same port as the OpenAI API. The compose file publishes the container's port 8000 to the host on `${LLAMACPP_WEBUI_PORT:-8080}`, so once the stack is up you can open:

- Web UI: `http://localhost:8080/`
- Raw OpenAI API: `http://localhost:8080/v1/...`

The gait gateway (port 3000) still proxies the same backend over the docker-internal network — exposing the host port is purely for direct access to the upstream UI (e.g. to test image/audio input that the gait gateway doesn't yet route through).

## Gemma 4 notes

- Only the **E2B** and **E4B** variants support audio input (max 30 s clips). The 26B-A4B and 31B variants are vision-only.
- The default config pulls **`unsloth/gemma-4-E4B-it-GGUF:Q8_0`**, which fits in ~8 GB and is the unsloth-recommended quant for the small variants.
- Thinking mode is template-controlled via `--chat-template-kwargs`. To toggle it explicitly, set the env var `LLAMA_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'` on the `llamacpp` service (note: this one does **not** use the `LLAMA_ARG_` prefix).
