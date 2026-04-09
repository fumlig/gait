# chatterbox

Starlette wrapper around [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) (Resemble AI). Refer to the upstream project for model details, licensing and generation parameter semantics.

Part of [gait](../../README.md) — the gateway handles OpenAI compatibility and talks to this service over raw RPC.

## What this service adds

Three upstream model variants served from a single container with OpenAI-style aliases. `DEFAULT_MODEL` is preloaded at startup; others load on first request.

| Model ID | Params | Languages |
|----------|--------|-----------|
| `chatterbox-turbo` | 350M | English (alias `tts-1`) |
| `chatterbox` | 500M | English (alias `tts-1-hd`) |
| `chatterbox-multilingual` | 500M | 23 languages (needs `language` parameter) |

Multilingual codes: `ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh`.

## Endpoints

RPC endpoints consumed by the gateway — not part of the OpenAI API.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/synthesize` | JSON body, returns `audio/wav`. Fields below. |
| `GET` | `/models` | Model IDs with loaded status. |
| `GET` | `/health` | Service health and per-model loaded status. |

### `/synthesize` body

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `text` | string | yes | — | Text to synthesize. |
| `voice` | string | yes | — | Voice name (`default` uses built-in). |
| `model` | string | yes | — | Model ID or alias. |
| `speed` | float | no | `1.0` | Playback speed multiplier. |
| `language` | string | no | `null` | ISO 639-1 code (required for multilingual). |
| `exaggeration` | float | no | `0.5` | Emotion exaggeration (0.0–2.0). |
| `cfg_weight` | float | no | `0.5` | Classifier-free guidance for pace (0.0–1.0). |
| `temperature` | float | no | `0.8` | Sampling temperature (0.01–5.0). |
| `repetition_penalty` | float | no | `1.2` | Token repetition penalty (1.0–3.0). |
| `top_p` | float | no | `1.0` | Nucleus sampling (0.0–1.0). |
| `min_p` | float | no | `0.05` | Min-p sampling (0.0–1.0). |
| `top_k` | int | no | `1000` | Top-k sampling. |
| `seed` | int | no | `null` | Random seed. |

## Voices

Voice reference clips (`.wav`, ~10 s of clean speech) live in `VOICES_DIR`. The filename stem is the voice name. When no voice files exist, any name is accepted and chatterbox falls back to its built-in default. Voices are managed through the gateway's `/v1/audio/voices` endpoint.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Inference device. |
| `HOST` | `0.0.0.0` | Server bind address. |
| `PORT` | `8000` | Server bind port. |
| `DEFAULT_MODEL` | `chatterbox-turbo` | Model to preload at startup. |
| `VOICES_DIR` | `/app/voices` | Voice reference clips directory. |
| `MODEL_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace cache. |
| `MODEL_IDLE_TIMEOUT` | `0` | Unload models after N seconds idle. `0` disables. |
| `HF_TOKEN` | — | HuggingFace token for gated models. |

## Build notes

Python 3.11 via deadsnakes PPA because `chatterbox-tts` transitively pulls `numpy<1.26`, which needs `distutils` (removed in 3.12). Pins `onnx<1.17` and `setuptools<81` for compatibility.
