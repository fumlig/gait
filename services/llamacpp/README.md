# llamacpp

Thin wrapper around the official [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) Docker image. No custom application code — runs the upstream `llama-server` binary directly, in **router mode**. The gateway proxies all requests transparently.

Router mode lets API callers select a configuration per request by passing a preset name in the `model` field of the OpenAI API payload. llama-server spawns a child process for that preset on demand, keeps it warm, and auto-unloads it when idle. See [Using multiple models](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#using-multiple-models) upstream for the full feature description.

## Endpoints

All provided by llama-server (already OpenAI-compatible):

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming supported) |
| `POST` | `/v1/completions` | Text completions (streaming supported) |
| `POST` | `/v1/responses` | Responses API (streaming supported) |
| `POST` | `/v1/embeddings` | Text embeddings |
| `GET`  | `/v1/models` | Presets + cached models with per-model lifecycle `status` |
| `POST` | `/models/load` | Load a preset (`{"model": "<id>"}`) — *note: no `/v1/` prefix* |
| `POST` | `/models/unload` | Unload a preset (`{"model": "<id>"}`) — *note: no `/v1/` prefix* |
| `GET`  | `/health` | Server health |

The gait gateway wraps the two model-management endpoints and re-exposes them under `/v1/models/load` and `/v1/models/unload` (see the root README).

## Configuration

Since the move to router mode, model-level tuning lives in a **preset file** ([`config/llama-models.ini`](../../config/llama-models.ini) at the repo root) rather than in environment variables. Only infrastructure-level settings are still passed via env vars.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODELS_MAX` | `1` | Maximum models loaded simultaneously. Bump to load more presets at once (costs VRAM). |
| `LLAMA_IDLE_TIMEOUT` | `-1` | Seconds of idleness before llama-server puts the current model to sleep. `-1` disables. Applied to every child instance. |
| `LLAMA_CACHE` | `${HF_HOME}` | llama.cpp's native cache dir. Falls back to `HF_HOME` when unset; llama-server now reads `HF_HOME` directly, so the shared mount is enough for most users. |
| `HF_TOKEN` | -- | HuggingFace access token for gated repos (passed to every child). |
| `LLAMACPP_WEBUI_PORT` | `8090` | Host port for the llama-server router web UI / raw API. |

Any other `LLAMA_ARG_*` variable listed in the [upstream docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) can still be set on the compose service — router-level args apply to every child instance.

### Presets

Edit [`config/llama-models.ini`](../../config/llama-models.ini) to add, remove, or retune models. Each section maps to llama-server CLI arguments (without leading dashes):

```ini
version = 1

[*]
; global defaults applied to every preset
ctx-size = 16384
n-gpu-layers = 99
flash-attn = on
jinja = true
reasoning-format = deepseek

[unsloth/gemma-4-E4B-it-GGUF:Q8_0]
; overrides for this preset
load-on-startup = true
fit = off
temp = 1.0
top-k = 64

[my-local-model]
; custom entry — must provide `model` (local path) or `hf-repo`
model = /root/.cache/huggingface/my-model-Q4_K_M.gguf
ctx-size = 8192
```

Precedence (highest wins):

1. Command-line arguments passed to `llama-server`
2. Model-specific section (e.g. `[unsloth/gemma-4-E4B-it-GGUF:Q8_0]`)
3. Global `[*]` section

Router-controlled args (`host`, `port`, `api-key`, `-hf`, `alias`) are stripped or overwritten when llama-server spawns a child.

Two preset-only keys that are *not* CLI arguments:

- `load-on-startup` (bool) — preload this preset on router start to avoid first-request latency.
- `stop-timeout` (int, seconds) — grace period before a forced SIGKILL on unload (default 10).

### Model sources

Presets resolve GGUF files in one of three ways:

1. **HuggingFace cache** — the preset section name is a HF repo spec (e.g. `[unsloth/gemma-4-E4B-it-GGUF:Q8_0]`). Files are downloaded to `HF_HOME` on first load, and sibling `mmproj-*.gguf` files are auto-downloaded (controlled by `--mmproj-auto`, on by default) which is what enables vision/audio input for multimodal models like Gemma 4 E2B/E4B.
2. **Local directory** — point `LLAMA_ARG_MODELS_DIR` at a directory of GGUF files (instead of the preset file, or in addition).
3. **Custom path in preset** — any preset section that specifies `model = /path/to/file.gguf` loads from disk directly.

## Native web UI

`llama-server` ships with a built-in chat web UI at `/`, served on the same port as the OpenAI API. The compose file publishes the container's port 8000 to the host on `${LLAMACPP_WEBUI_PORT:-8080}`, so once the stack is up you can open:

- Web UI: `http://localhost:8080/`
- Raw OpenAI API: `http://localhost:8080/v1/...`

The gait gateway (port 3000) still proxies the same backend over the docker-internal network — exposing the host port is purely for direct access to the upstream UI (e.g. to test image/audio input that the gait gateway doesn't yet route through).

## Gemma 4 notes

- Only the **E2B** and **E4B** variants support audio input (max 30 s clips). The 26B-A4B and 31B variants are vision-only.
- The default config pulls **`unsloth/gemma-4-E4B-it-GGUF:Q8_0`**, which fits in ~8 GB and is the unsloth-recommended quant for the small variants.
- Thinking mode is template-controlled via `--chat-template-kwargs`. To toggle it explicitly, set the env var `LLAMA_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'` on the `llamacpp` service (note: this one does **not** use the `LLAMA_ARG_` prefix).
