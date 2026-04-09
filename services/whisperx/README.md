# whisperx

Starlette wrapper around [WhisperX](https://github.com/m-bain/whisperX), which adds word-level alignment and optional speaker diarization on top of [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Refer to the upstream projects for model details and parameter semantics.

Part of [gait](../../README.md) — the gateway handles OpenAI compatibility (including `json` / `text` / `srt` / `vtt` / `verbose_json` conversion) and talks to this service over raw RPC.

## What this service adds

Loads one Whisper model at a time. When a request specifies a different model, the current one is unloaded first. The `whisper-1` alias maps to `DEFAULT_MODEL`.

## Endpoints

RPC endpoints consumed by the gateway — not part of the OpenAI API. Both transcribe/translate endpoints return JSON segments.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/transcribe` | Multipart. Preserves source language. Fields below. |
| `POST` | `/translate` | Same shape as `/transcribe` minus `language`; translates to English. |
| `GET` | `/models` | Model IDs with loaded status. |
| `GET` | `/health` | Service health and loaded model name. |

### `/transcribe` body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | file | yes | — | Audio file (wav, mp3, mp4, m4a, webm, ogg, flac; max 25 MB). |
| `model` | string | yes | — | `whisper-1` (alias for default) or a size like `large-v3`, `base`, `turbo`. |
| `language` | string | no | auto-detect | ISO-639-1 language code. |
| `prompt` | string | no | — | Context to guide the model. |
| `temperature` | float | no | `0.0` | Sampling temperature. |
| `word_timestamps` | bool | no | `false` | Include word-level timestamps. |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device. |
| `HOST` | `0.0.0.0` | Server bind address. |
| `PORT` | `8000` | Server bind port. |
| `DEFAULT_MODEL` | `large-v3` | Model loaded on startup and for `whisper-1`. |
| `COMPUTE_TYPE` | `float16` | Model precision. |
| `BATCH_SIZE` | `16` | Transcription batch size. |
| `ENABLE_DIARIZATION` | `false` | Enable speaker diarization (requires `HF_TOKEN`). |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace cache. |
| `MAX_FILE_SIZE` | `26214400` | Maximum upload size in bytes (25 MB). |
| `MODEL_IDLE_TIMEOUT` | `0` | Unload models after N seconds idle. `0` disables. |
| `HF_TOKEN` | — | HuggingFace token (required for diarization). |

## Build notes

Python 3.11 via deadsnakes PPA, CUDA 12.6.3 runtime, `ffmpeg` required for audio decoding. PyTorch CUDA wheels are pinned via `[tool.uv.sources]` in `pyproject.toml`.
