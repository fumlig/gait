# chatterbox

OpenAI-compatible TTS API backed by [Chatterbox](https://github.com/resemble-ai/chatterbox) models from Resemble AI.

Three model variants run in a single container. The default model is preloaded at startup; others are loaded lazily on first request.

| Model ID | Params | Languages | Notes |
|----------|--------|-----------|-------|
| `chatterbox-turbo` | 350M | English | Lowest latency, alias `tts-1` |
| `chatterbox` | 500M | English | Higher quality, alias `tts-1-hd` |
| `chatterbox-multilingual` | 500M | 23 languages | Requires `language` parameter |

All variants support paralinguistic tags like `[laugh]`, `[sigh]`, etc. embedded in the input text.

## Endpoints

### `POST /v1/audio/speech`

Generate speech from text. Compatible with the [OpenAI TTS API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Standard OpenAI parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | yes | -- | `chatterbox-turbo`, `chatterbox`, `chatterbox-multilingual` (also `tts-1`, `tts-1-hd`) |
| `input` | string | yes | -- | Text to synthesize (max 4096 chars) |
| `voice` | string | yes | -- | Name of a registered voice, or any string when no voices are registered |
| `response_format` | string | no | `mp3` | `mp3` or `wav` |
| `speed` | float | no | `1.0` | Playback speed multiplier (`0.25` -- `4.0`) |

**Extended parameters (Chatterbox-specific):**

| Parameter | Type | Default | Models | Description |
|-----------|------|---------|--------|-------------|
| `language` | string | `null` | multilingual | ISO 639-1 code. Required for `chatterbox-multilingual` |
| `exaggeration` | float | `0.5` | original, multilingual | Emotion exaggeration (`0.0` -- `2.0`) |
| `cfg_weight` | float | `0.5` | original, multilingual | Classifier-free guidance for pace (`0.0` -- `1.0`) |
| `temperature` | float | `0.8` | all | Sampling temperature (`0.01` -- `5.0`) |
| `repetition_penalty` | float | `1.2` | all | Token repetition penalty (`1.0` -- `3.0`) |
| `top_p` | float | `1.0` | all | Nucleus sampling threshold (`0.0` -- `1.0`) |
| `min_p` | float | `0.05` | original, multilingual | Min-p sampling threshold (`0.0` -- `1.0`) |
| `top_k` | int | `1000` | turbo | Top-k sampling (>= 1) |
| `seed` | int | `null` | all | Random seed for reproducibility |

```bash
# Turbo (fast, English)
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-turbo","input":"Hello world!","voice":"default"}' \
  --output speech.mp3

# Original (high quality, English)
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox","input":"Hello world!","voice":"default","exaggeration":0.8}' \
  --output speech_hd.mp3

# Multilingual (French)
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"chatterbox-multilingual","input":"Bonjour le monde!","voice":"default","language":"fr"}' \
  --output speech_fr.mp3

# Using OpenAI alias
curl http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Drop-in compatible!","voice":"default"}' \
  --output speech_alias.mp3
```

### Supported languages

`chatterbox-multilingual` supports: ar (Arabic), da (Danish), de (German), el (Greek), en (English), es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi), it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch), no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian), sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese).

### `GET /v1/models`

List available models. Returns an OpenAI-compatible model list with all three variants.

### `GET /health`

Returns service health with per-model loaded status:

```json
{
  "status": "ok",
  "model_loaded": true,
  "models": {
    "chatterbox": false,
    "chatterbox-multilingual": false,
    "chatterbox-turbo": true
  }
}
```

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
| `DEFAULT_MODEL` | `chatterbox-turbo` | Model to preload at startup (others load on demand) |
| `VOICES_DIR` | `/app/voices` | Path to voice reference clips inside the container |
| `MAX_INPUT_LENGTH` | `4096` | Maximum input text length (characters) |
| `STREAM_CHUNK_SIZE` | `4096` | Streaming response chunk size in bytes |
| `HF_TOKEN` | -- | HuggingFace token (required for gated model access) |

## Development

Requires Python 3.11 (pinned in `.python-version`; `uv` auto-downloads it if missing).

```bash
uv sync --all-extras      # creates .venv with Python 3.11, installs all deps + dev tools
uv run ruff check src/    # lint
uv run pytest             # tests (mocked engine, no GPU needed)
```

For IDE support (e.g. VS Code / PyCharm), point the Python interpreter at `services/chatterbox/.venv/bin/python`.

## Build notes

The Dockerfile uses Python 3.11 via deadsnakes PPA because `chatterbox-tts` transitively depends on `numpy<1.26`, which requires `distutils` (removed in Python 3.12).

Two additional dependency pins are required:
- `onnx<1.17` -- newer onnx requires `ml-dtypes>=0.5` which conflicts with the numpy pin.
- `setuptools<81` -- the `perth` watermarking library uses `pkg_resources`, removed in setuptools 81+.

## Architecture

```
src/chatterbox_service/
  main.py          # FastAPI app + lifespan (preload default model)
  config.py        # Pydantic settings from env vars
  models.py        # Request/response schemas
  engine.py        # Multi-model loading, voice resolution, inference
  audio.py         # WAV/MP3 encoding + chunked streaming
  routes/
    speech.py      # POST /v1/audio/speech
    models.py      # GET /v1/models
    health.py      # GET /health
```
