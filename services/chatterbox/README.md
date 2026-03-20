# chatterbox

TTS backend using [Chatterbox](https://github.com/resemble-ai/chatterbox) models from Resemble AI. Thin Starlette app exposing RPC endpoints — the gateway handles OpenAI API compatibility.

Three model variants run in a single container. The default model is preloaded at startup; others load on first request.

| Model ID | Params | Languages |
|----------|--------|-----------|
| `chatterbox-turbo` | 350M | English (alias `tts-1`) |
| `chatterbox` | 500M | English (alias `tts-1-hd`) |
| `chatterbox-multilingual` | 500M | 23 languages (requires `language` parameter) |

All variants support paralinguistic tags like `[laugh]`, `[sigh]`, etc.

## Endpoints

### `POST /synthesize`

Returns `audio/wav`. JSON body:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | -- | Text to synthesize |
| `voice` | string | yes | -- | Voice name (`default` uses built-in) |
| `model` | string | yes | -- | Model ID or alias |
| `speed` | float | no | `1.0` | Playback speed multiplier |
| `language` | string | no | `null` | ISO 639-1 code (required for multilingual) |
| `exaggeration` | float | no | `0.5` | Emotion exaggeration (0.0–2.0) |
| `cfg_weight` | float | no | `0.5` | Classifier-free guidance for pace (0.0–1.0) |
| `temperature` | float | no | `0.8` | Sampling temperature (0.01–5.0) |
| `repetition_penalty` | float | no | `1.2` | Token repetition penalty (1.0–3.0) |
| `top_p` | float | no | `1.0` | Nucleus sampling (0.0–1.0) |
| `min_p` | float | no | `0.05` | Min-p sampling (0.0–1.0) |
| `top_k` | int | no | `1000` | Top-k sampling |
| `seed` | int | no | `null` | Random seed |

### `GET /models`

List available models with loaded status.

### `GET /health`

Service health with per-model loaded status.

## Voices

Voice reference clips (`.wav`, ~10 seconds of clean speech) are stored in `VOICES_DIR`. The filename stem becomes the voice name. When no voice files exist, any voice name is accepted and chatterbox uses its built-in default. Voice clips are managed through the gateway's `/v1/audio/voices` endpoint.

## Supported languages (multilingual)

ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server bind port |
| `DEFAULT_MODEL` | `chatterbox-turbo` | Model to preload at startup |
| `VOICES_DIR` | `/app/voices` | Voice reference clips directory |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace cache |
| `MODEL_IDLE_TIMEOUT` | `0` | Unload models after N seconds idle (0 = disabled) |
| `HF_TOKEN` | -- | HuggingFace token for gated models |

## Build notes

Uses Python 3.11 via deadsnakes PPA because `chatterbox-tts` transitively depends on `numpy<1.26` which requires `distutils` (removed in Python 3.12). Pins `onnx<1.17` and `setuptools<81` for compatibility.
