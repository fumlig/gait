# chatterbox

TTS backend service backed by [Chatterbox](https://github.com/resemble-ai/chatterbox) models from Resemble AI. This is a thin Starlette app that exposes raw RPC endpoints — the gateway handles OpenAI API compatibility.

Three model variants run in a single container. The default model is preloaded at startup; others are loaded lazily on first request.

| Model ID | Params | Languages | Notes |
|----------|--------|-----------|-------|
| `chatterbox-turbo` | 350M | English | Lowest latency, alias `tts-1` |
| `chatterbox` | 500M | English | Higher quality, alias `tts-1-hd` |
| `chatterbox-multilingual` | 500M | 23 languages | Requires `language` parameter |

All variants support paralinguistic tags like `[laugh]`, `[sigh]`, etc. embedded in the input text.

## Endpoints

These are internal RPC endpoints consumed by the gateway. End users interact with the OpenAI-compatible API on the gateway.

### `POST /synthesize`

Generate speech from text. Returns raw `audio/wav` binary.

**JSON body:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | -- | Text to synthesize |
| `voice` | string | yes | -- | Name of a registered voice, or any string when no voices are registered |
| `model` | string | yes | -- | `chatterbox-turbo`, `chatterbox`, `chatterbox-multilingual` (also `tts-1`, `tts-1-hd`) |
| `speed` | float | no | `1.0` | Playback speed multiplier |
| `language` | string | no | `null` | ISO 639-1 code (required for `chatterbox-multilingual`) |
| `exaggeration` | float | no | `0.5` | Emotion exaggeration (`0.0` -- `2.0`) |
| `cfg_weight` | float | no | `0.5` | Classifier-free guidance for pace (`0.0` -- `1.0`) |
| `temperature` | float | no | `0.8` | Sampling temperature (`0.01` -- `5.0`) |
| `repetition_penalty` | float | no | `1.2` | Token repetition penalty (`1.0` -- `3.0`) |
| `top_p` | float | no | `1.0` | Nucleus sampling threshold (`0.0` -- `1.0`) |
| `min_p` | float | no | `0.05` | Min-p sampling threshold (`0.0` -- `1.0`) |
| `top_k` | int | no | `1000` | Top-k sampling (>= 1) |
| `seed` | int | no | `null` | Random seed for reproducibility |

### `GET /models`

List available models. Returns `{ "object": "list", "data": [{ "id", "object", "owned_by" }] }`.

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

### Supported languages

`chatterbox-multilingual` supports: ar (Arabic), da (Danish), de (German), el (Greek), en (English), es (Spanish), fi (Finnish), fr (French), he (Hebrew), hi (Hindi), it (Italian), ja (Japanese), ko (Korean), ms (Malay), nl (Dutch), no (Norwegian), pl (Polish), pt (Portuguese), ru (Russian), sv (Swedish), sw (Swahili), tr (Turkish), zh (Chinese).

## Voices

Voice reference clips (`.wav` files, ~10 seconds of clean speech) are stored in the `VOICES_DIR` directory (shared with the voice service via a Docker volume). The filename stem becomes the voice name:

```
voices/
  narrator.wav   -> voice name "narrator"
  emma.wav       -> voice name "emma"
```

When no voice files are registered, any voice name is accepted and Chatterbox uses its default voice (no cloning). When voices are registered, unknown names return a 400 error.

Voice clips are managed through the voice service (or the gateway's `/v1/audio/voices` endpoint), not through this service directly.

## Configuration

All settings are configurable via environment variables (no prefix, case-insensitive):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device (`cuda`, `cpu`, `mps`) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `DEFAULT_MODEL` | `chatterbox-turbo` | Model to preload at startup (others load on demand) |
| `VOICES_DIR` | `/app/voices` | Path to voice reference clips inside the container |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace model cache directory |
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
  app.py         # Starlette app (RPC endpoints: /synthesize, /models, /health)
  config.py      # Pydantic settings from env vars
  engine.py      # Multi-model loading, voice resolution, inference
```
