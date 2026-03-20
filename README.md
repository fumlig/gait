# gait

Local ML models exposed via OpenAI-compatible REST APIs.

A FastAPI gateway unifies all services behind a single port. Each ML model runs as a thin Starlette backend in its own Docker container with GPU passthrough. Orchestrated with `docker compose`.

## Models

| Service | Model | Params | Type |
|---------|-------|--------|------|
| [llamacpp](services/llamacpp/) | [llama.cpp](https://github.com/ggml-org/llama.cpp) (any GGUF model) | varies | LLM |
| [chatterbox](services/chatterbox/) | [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) | 350M | TTS (English) |
| [chatterbox](services/chatterbox/) | [Chatterbox](https://github.com/resemble-ai/chatterbox) | 500M | TTS (English) |
| [chatterbox](services/chatterbox/) | [Chatterbox-Multilingual](https://github.com/resemble-ai/chatterbox) | 500M | TTS (23 languages) |
| [whisperx](services/whisperx/) | [WhisperX](https://github.com/m-bain/whisperX) (large-v3 default) | 1.5B | STT |

The default LLM is [Qwen 3.5 9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (Q8_0, ~9.5 GB), which fits on an RTX 3090 alongside the TTS and STT models. Set `LLAMA_HF_REPO` to use a different model. All three Chatterbox models run in a single container; the default (`chatterbox-turbo`) is preloaded at startup, others load on demand.

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
| `LLAMA_HF_REPO` | `unsloth/Qwen3.5-9B-GGUF:Q8_0` | HuggingFace repo for LLM |
| `LLAMA_MODEL` | -- | Path to a local GGUF file (alternative to `LLAMA_HF_REPO`) |
| `LLAMA_CTX_SIZE` | `4096` | Context window size |
| `LLAMA_N_GPU_LAYERS` | `99` | GPU layers to offload |
| `MODELS_DIR` | `./models` | LLM model cache |
| `CHATTERBOX_DEFAULT_MODEL` | `chatterbox-turbo` | Chatterbox model to preload |
| `VOICES_DIR` | `./voices` | Voice reference clips (shared between gateway and chatterbox) |
| `BACKEND_TIMEOUT` | `300` | Gateway timeout for backend requests (seconds) |

Per-service configuration is documented in each service's README.

## Systemd

A `gait.service` unit file is provided. Edit `WorkingDirectory` if your checkout isn't at `/home/oskar/projects/gait`, then:

```bash
sudo cp gait.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gait   # auto-start on boot
sudo systemctl start gait    # start now
```

See [AGENTS.md](AGENTS.md) for architecture details and development conventions.
