# whisperx

OpenAI-compatible speech-to-text API backed by [WhisperX](https://github.com/m-bain/whisperX), which adds word-level alignment (via wav2vec2) and optional speaker diarization (via pyannote) on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

## Endpoints

### `POST /v1/audio/transcriptions`

Transcribe audio into text. Compatible with the [OpenAI Transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | yes | -- | Audio file (wav, mp3, mp4, m4a, webm, ogg, flac; max 25 MB) |
| `model` | string | yes | -- | `whisper-1` (alias for default model) or a size like `large-v3`, `base`, `turbo`, etc. |
| `language` | string | no | auto-detect | ISO-639-1 language code (e.g. `en`, `de`, `fr`) |
| `prompt` | string | no | -- | Optional context to guide the model |
| `response_format` | string | no | `json` | `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | float | no | `0.0` | Sampling temperature |
| `timestamp_granularities[]` | string[] | no | -- | `word` and/or `segment` (requires `verbose_json`) |

```bash
curl http://localhost:8201/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=whisper-1 \
  -F response_format=verbose_json
```

### `POST /v1/audio/translations`

Translate audio into English text. Same parameters as transcriptions except no `language` (always translates to English).

```bash
curl http://localhost:8201/v1/audio/translations \
  -F file=@german_audio.wav \
  -F model=whisper-1
```

### `GET /v1/models`

List available models. Returns an OpenAI-compatible model list.

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
  main.py          # FastAPI app + lifespan (model load/unload)
  config.py        # Pydantic settings from env vars
  models.py        # Request/response schemas
  engine.py        # Model loading, alignment, diarization, inference
  routes/
    transcriptions.py  # POST /v1/audio/transcriptions
    translations.py    # POST /v1/audio/translations
    models.py          # GET /v1/models
    health.py          # GET /health
```
