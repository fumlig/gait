# whisperx

STT backend service backed by [WhisperX](https://github.com/m-bain/whisperX), which adds word-level alignment (via wav2vec2) and optional speaker diarization (via pyannote) on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper). This is a thin Starlette app that exposes raw RPC endpoints — the gateway handles OpenAI API compatibility and response format conversion (JSON→SRT/VTT/text).

## Endpoints

These are internal RPC endpoints consumed by the gateway. End users interact with the OpenAI-compatible API on the gateway.

### `POST /transcribe`

Transcribe audio to text (preserves source language). Returns raw JSON segments.

**Multipart form data:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | yes | -- | Audio file (wav, mp3, mp4, m4a, webm, ogg, flac; max 25 MB) |
| `model` | string | yes | -- | `whisper-1` (alias for default model) or a size like `large-v3`, `base`, `turbo`, etc. |
| `language` | string | no | auto-detect | ISO-639-1 language code (e.g. `en`, `de`, `fr`) |
| `prompt` | string | no | -- | Optional context to guide the model |
| `temperature` | float | no | `0.0` | Sampling temperature |
| `word_timestamps` | bool | no | `false` | Include word-level timestamps in output |

### `POST /translate`

Translate audio to English text. Same parameters as `/transcribe` except no `language` (always translates to English).

### `GET /models`

List available models. Returns `{ "object": "list", "data": [{ "id", "object", "owned_by" }] }`.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}` when the model is loaded and ready.

## Dynamic model loading

The service loads one Whisper model at a time. When a request specifies a different model than what's currently loaded, the service unloads the current model and loads the requested one. Alignment and diarization models are cached independently.

- `whisper-1` maps to the `DEFAULT_MODEL` env var (default: `large-v3`)
- Direct model names (`large-v3`, `base`, `turbo`, etc.) load that exact model

## Configuration

All settings are configurable via environment variables (no prefix, case-insensitive):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device (`cuda`, `cpu`) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `DEFAULT_MODEL` | `large-v3` | Whisper model loaded on startup and when `whisper-1` is requested |
| `COMPUTE_TYPE` | `float16` | Model precision (`float16`, `int8`, `float32`) |
| `BATCH_SIZE` | `16` | Batch size for transcription |
| `ENABLE_DIARIZATION` | `false` | Enable speaker diarization (requires `HF_TOKEN`) |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace model cache directory |
| `MAX_FILE_SIZE` | `26214400` | Maximum upload file size in bytes (default 25 MB) |
| `HF_TOKEN` | -- | HuggingFace token (required for diarization with pyannote) |

## Development

Requires Python 3.11 (pinned in `.python-version`; `uv` auto-downloads it if missing).

```bash
uv sync --all-extras      # creates .venv with Python 3.11, installs all deps + dev tools
uv run ruff check src/    # lint
uv run pytest             # tests (mocked engine, no GPU needed)
```

For IDE support (e.g. VS Code / PyCharm), point the Python interpreter at `services/whisperx/.venv/bin/python`.

## Build notes

The Dockerfile uses Python 3.11 via deadsnakes PPA and CUDA 12.6.3 runtime. WhisperX requires `ffmpeg` for audio decoding, which is installed in the image.

PyTorch CUDA wheels are sourced from the official PyTorch index via `[tool.uv.sources]` overrides in `pyproject.toml`.

## Architecture

```
src/whisperx_service/
  app.py         # Starlette app (RPC endpoints: /transcribe, /translate, /models, /health)
  config.py      # Pydantic settings from env vars
  engine.py      # Model loading, alignment, diarization, inference
```
