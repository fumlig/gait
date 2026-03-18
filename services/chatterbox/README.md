# chatterbox

OpenAI-compatible TTS API backed by [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) (350M params) from Resemble AI.

Chatterbox-Turbo is the lowest-latency variant of the Chatterbox family. It supports paralinguistic tags like `[laugh]`, `[sigh]`, etc. embedded directly in the input text.

## Endpoints

### `POST /v1/audio/speech`

Generate speech from text. Compatible with the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | yes | -- | `chatterbox-turbo` (also accepts `tts-1`, `tts-1-hd` for drop-in compat) |
| `input` | string | yes | -- | Text to synthesize (max 4096 chars) |
| `voice` | string | yes | -- | Name of a registered voice, or any string when no voices are registered |
| `response_format` | string | no | `mp3` | `mp3` or `wav` |
| `speed` | float | no | `1.0` | Playback speed multiplier (`0.25` -- `4.0`) |

```bash
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello world!","voice":"default"}' \
  --output speech.mp3
```

### `GET /v1/models`

List available models. Returns an OpenAI-compatible model list.

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}` when the model is loaded and ready.

## Voices

Place `.wav` reference clips (~10 seconds of clean speech) in the `voices/` directory. The filename stem becomes the voice name:

```
voices/
  narrator.wav   -> voice name "narrator"
  emma.wav       -> voice name "emma"
```

When no voice files are registered, any voice name is accepted and Chatterbox uses its default voice (no cloning). When voices are registered, unknown names return a 400 error.

## Configuration

All settings are configurable via environment variables (no prefix, case-insensitive):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device (`cuda`, `cpu`, `mps`) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `VOICES_DIR` | `/app/voices` | Path to voice reference clips inside the container |
| `MAX_INPUT_LENGTH` | `4096` | Maximum input text length (characters) |
| `STREAM_CHUNK_SIZE` | `4096` | Streaming response chunk size in bytes |
| `HF_TOKEN` | -- | HuggingFace token (required for gated model access) |

## Development

```bash
uv sync --dev
uv run ruff check src/
uv run pytest
```

Tests use a mocked engine and do not require a GPU.

## Build notes

The Dockerfile uses Python 3.11 via deadsnakes PPA because `chatterbox-tts` transitively depends on `numpy<1.26`, which requires `distutils` (removed in Python 3.12).

Two additional dependency pins are required:
- `onnx<1.17` -- newer onnx requires `ml-dtypes>=0.5` which conflicts with the numpy pin.
- `setuptools<81` -- the `perth` watermarking library uses `pkg_resources`, removed in setuptools 81+.

## Architecture

```
src/chatterbox_service/
  main.py          # FastAPI app + lifespan (model load/unload)
  config.py        # Pydantic settings from env vars
  models.py        # Request/response schemas
  engine.py        # Model loading, voice resolution, inference
  audio.py         # WAV/MP3 encoding + chunked streaming
  routes/
    speech.py      # POST /v1/audio/speech
    models.py      # GET /v1/models
    health.py      # GET /health
```
