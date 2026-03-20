# whisperx

STT backend using [WhisperX](https://github.com/m-bain/whisperX), which adds word-level alignment and optional speaker diarization on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Thin Starlette app exposing RPC endpoints — the gateway handles OpenAI API compatibility and format conversion (JSON→SRT/VTT/text).

## Endpoints

### `POST /transcribe`

Transcribe audio (preserves source language). Returns JSON segments. Multipart form data:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | yes | -- | Audio file (wav, mp3, mp4, m4a, webm, ogg, flac; max 25 MB) |
| `model` | string | yes | -- | `whisper-1` (alias for default) or a size like `large-v3`, `base`, `turbo` |
| `language` | string | no | auto-detect | ISO-639-1 language code |
| `prompt` | string | no | -- | Context to guide the model |
| `temperature` | float | no | `0.0` | Sampling temperature |
| `word_timestamps` | bool | no | `false` | Include word-level timestamps |

### `POST /translate`

Translate audio to English. Same parameters as `/transcribe` except no `language`.

### `GET /models`

List available models with loaded status.

### `GET /health`

Service health with loaded model name.

## Dynamic model loading

Loads one Whisper model at a time. When a request specifies a different model, the current one is unloaded first. `whisper-1` maps to `DEFAULT_MODEL`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `DEFAULT_MODEL` | `large-v3` | Model loaded on startup and for `whisper-1` |
| `COMPUTE_TYPE` | `float16` | Model precision |
| `BATCH_SIZE` | `16` | Transcription batch size |
| `ENABLE_DIARIZATION` | `false` | Enable speaker diarization (requires `HF_TOKEN`) |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace cache |
| `MAX_FILE_SIZE` | `26214400` | Maximum upload size in bytes (25 MB) |
| `MODEL_IDLE_TIMEOUT` | `0` | Unload models after N seconds idle (0 = disabled) |
| `HF_TOKEN` | -- | HuggingFace token (required for diarization) |

## Build notes

Uses Python 3.11 via deadsnakes PPA and CUDA 12.6.3 runtime. Requires `ffmpeg` for audio decoding. PyTorch CUDA wheels are sourced from the official PyTorch index via `[tool.uv.sources]` in `pyproject.toml`.
