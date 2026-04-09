# llamacpp

Thin wrapper around the upstream [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) Docker image running in [router mode](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#using-multiple-models). No custom application code — `llama-server` handles the OpenAI protocol directly, and the gait gateway proxies requests transparently.

Part of [gait](../../README.md). Refer to the upstream llama.cpp server README for flags, API details and router-mode semantics.

## What this service adds

- A Dockerfile that builds llama.cpp with CUDA and runs it in router mode with a preset file.
- A preset file convention (`config/llama-models.ini`) for declaring models and per-model CLI overrides.
- A `LLAMACPP_WEBUI_PORT` mapping so llama-server's built-in chat UI is reachable from the host.

The gait gateway wraps llama-server's `/models/load` and `/models/unload` endpoints and re-exposes them under `/v1/models/load` and `/v1/models/unload` (see the [root README](../../README.md#endpoints)).

## Configuration

Model-level tuning lives in the preset file, not env vars. Only infrastructure settings are passed through the environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODELS_MAX` | `1` | Max presets loaded simultaneously. |
| `LLAMA_IDLE_TIMEOUT` | `-1` | Seconds of idleness before the active model sleeps. `-1` disables. |
| `LLAMA_CACHE` | `${HF_HOME}` | Override for llama.cpp's native cache dir. Falls back to `HF_HOME`. |
| `HF_TOKEN` | — | HuggingFace token for gated repos (passed to every child). |
| `LLAMACPP_WEBUI_PORT` | `8090` | Host port for llama-server's built-in web UI / raw API. |

Any `LLAMA_ARG_*` variable from the [upstream docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) can be set on the compose service — router-level args apply to every child instance.

### Preset file

Edit [`config/llama-models.ini`](../../config/llama-models.ini) to add, remove or retune models. Each section maps to llama-server CLI arguments without the leading dashes:

```ini
version = 1

[*]
; defaults applied to every preset
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

Precedence (highest wins): CLI args passed to `llama-server` → model-specific section → `[*]` section. Router-controlled args (`host`, `port`, `api-key`, `-hf`, `alias`) are stripped or overwritten when llama-server spawns a child.

Two preset-only keys that are *not* CLI arguments:

- `load-on-startup` (bool) — preload this preset on router start to avoid first-request latency.
- `stop-timeout` (int, seconds) — grace period before forced SIGKILL on unload (default 10).

Presets resolve GGUF files in one of three ways:

1. **HuggingFace cache** — the section name is an HF repo spec (e.g. `[unsloth/gemma-4-E4B-it-GGUF:Q8_0]`). Files are downloaded to `HF_HOME` on first load. Sibling `mmproj-*.gguf` files are auto-downloaded (`--mmproj-auto`, on by default), enabling vision/audio input for multimodal models like Gemma 4 E2B/E4B.
2. **Local directory** — point `LLAMA_ARG_MODELS_DIR` at a directory of GGUF files.
3. **Explicit path** — a `model = /path/to/file.gguf` line in the preset section.

## Native web UI

`llama-server` serves its chat UI at `/` on the same port as the OpenAI API. The compose file publishes it on `${LLAMACPP_WEBUI_PORT:-8090}`:

- Web UI: http://localhost:8090/
- Raw OpenAI API: http://localhost:8090/v1/...

The gateway still proxies the same backend over the Docker network; exposing the host port is only useful for direct access to the upstream UI.

## Gemma 4 notes

- Only the **E2B** and **E4B** variants support audio input (max 30 s clips). The 26B-A4B and 31B variants are vision-only.
- The default preset pulls `unsloth/gemma-4-E4B-it-GGUF:Q8_0`, which fits in ~8 GB.
- Thinking mode is template-controlled via `--chat-template-kwargs`. To toggle it explicitly, set `LLAMA_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'` on the `llamacpp` service (note: no `LLAMA_ARG_` prefix).
