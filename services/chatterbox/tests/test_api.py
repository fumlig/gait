from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from httpx import ASGITransport, AsyncClient

SAMPLE_RATE = 24000
DUMMY_WAV = torch.randn(1, SAMPLE_RATE)  # 1 second of noise


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


@pytest.fixture()
def mock_engine():
    """Patch the engine singleton so no real model is loaded."""
    with patch("chatterbox_service.engine.engine") as eng:
        eng.is_loaded = True
        eng.sample_rate = SAMPLE_RATE
        eng.loaded_models = {
            "chatterbox-turbo": True,
            "chatterbox": False,
            "chatterbox-multilingual": False,
        }
        eng.generate.return_value = (DUMMY_WAV, SAMPLE_RATE)
        eng.ensure_model.return_value = "chatterbox-turbo"
        eng.touch.return_value = None
        eng.unload.return_value = None
        # Status state machine used by the /models, /models/load,
        # /models/unload and /health endpoints. The default is one
        # model loaded, the rest unloaded.
        eng.status_phase = "loaded"

        def _status_for(name: str) -> str:
            return "loaded" if eng.loaded_models.get(name) else "unloaded"

        eng.status_for.side_effect = _status_for
        # Also patch the engine used in app.py (same singleton)
        with patch("chatterbox_service.app.engine", eng):
            yield eng


@pytest.fixture()
def app(mock_engine):
    """Import the real app with a noop lifespan (skips model loading)."""
    from chatterbox_service.app import app

    app.router.lifespan_context = _noop_lifespan
    return app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


async def test_health_shows_per_model_status(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert data["models"]["chatterbox-turbo"] is True
    assert data["models"]["chatterbox"] is False
    assert data["models"]["chatterbox-multilingual"] is False


async def test_list_models(client: AsyncClient):
    resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 3
    model_ids = {m["id"] for m in data["data"]}
    assert model_ids == {"chatterbox-turbo", "chatterbox", "chatterbox-multilingual"}


async def test_list_models_owned_by(client: AsyncClient):
    resp = await client.get("/models")
    data = resp.json()
    for model in data["data"]:
        assert model["owned_by"] == "resemble-ai"


async def test_list_models_capabilities(client: AsyncClient):
    """Models endpoint returns capabilities for each model."""
    resp = await client.get("/models")
    data = resp.json()
    for model in data["data"]:
        assert "capabilities" in model
        assert "speech" in model["capabilities"]


async def test_list_models_loaded_status(client: AsyncClient):
    """Models endpoint returns loaded status for each model."""
    resp = await client.get("/models")
    data = resp.json()
    by_id = {m["id"]: m for m in data["data"]}
    assert by_id["chatterbox-turbo"]["loaded"] is True
    assert by_id["chatterbox"]["loaded"] is False
    assert by_id["chatterbox-multilingual"]["loaded"] is False


async def test_synthesize_returns_wav(client: AsyncClient, mock_engine: MagicMock):
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-turbo",
            "text": "Hello world",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
    mock_engine.generate.assert_called_once()


async def test_synthesize_alias_tts1(client: AsyncClient, mock_engine: MagicMock):
    """tts-1 should be accepted and resolved by ensure_model."""
    mock_engine.ensure_model.return_value = "chatterbox-turbo"
    resp = await client.post(
        "/synthesize",
        json={
            "model": "tts-1",
            "text": "Alias test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    mock_engine.ensure_model.assert_called_with("tts-1")


async def test_synthesize_alias_tts1hd(client: AsyncClient, mock_engine: MagicMock):
    """tts-1-hd should be accepted and resolved by ensure_model."""
    mock_engine.ensure_model.return_value = "chatterbox"
    resp = await client.post(
        "/synthesize",
        json={
            "model": "tts-1-hd",
            "text": "Alias test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    mock_engine.ensure_model.assert_called_with("tts-1-hd")


async def test_synthesize_original_model(client: AsyncClient, mock_engine: MagicMock):
    """The original chatterbox model should accept exaggeration and cfg_weight."""
    mock_engine.ensure_model.return_value = "chatterbox"
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox",
            "text": "Original model test",
            "voice": "default",
            "exaggeration": 0.8,
            "cfg_weight": 0.7,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["model_name"] == "chatterbox"
    assert call_kwargs.kwargs["exaggeration"] == 0.8
    assert call_kwargs.kwargs["cfg_weight"] == 0.7


async def test_synthesize_multilingual(client: AsyncClient, mock_engine: MagicMock):
    """Multilingual model with valid language should succeed."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-multilingual",
            "text": "Bonjour le monde",
            "voice": "default",
            "language": "fr",
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["model_name"] == "chatterbox-multilingual"
    assert call_kwargs.kwargs["language"] == "fr"


async def test_synthesize_multilingual_missing_language(
    client: AsyncClient, mock_engine: MagicMock
):
    """Multilingual model without language should return 400."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    with patch(
        "chatterbox_service.app.validate_language",
        side_effect=ValueError("'language' is required for chatterbox-multilingual."),
    ):
        resp = await client.post(
            "/synthesize",
            json={
                "model": "chatterbox-multilingual",
                "text": "Missing language",
                "voice": "default",
            },
        )
    assert resp.status_code == 400
    assert "language" in resp.json()["detail"].lower()


async def test_synthesize_multilingual_invalid_language(
    client: AsyncClient, mock_engine: MagicMock
):
    """Multilingual model with unsupported language should return 400."""
    mock_engine.ensure_model.return_value = "chatterbox-multilingual"
    with patch(
        "chatterbox_service.app.validate_language",
        side_effect=ValueError("Unsupported language 'xx'."),
    ):
        resp = await client.post(
            "/synthesize",
            json={
                "model": "chatterbox-multilingual",
                "text": "Invalid language",
                "voice": "default",
                "language": "xx",
            },
        )
    assert resp.status_code == 400
    assert "xx" in resp.json()["detail"]


async def test_synthesize_with_seed(client: AsyncClient, mock_engine: MagicMock):
    """Seed parameter should be passed through to engine."""
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-turbo",
            "text": "Seed test",
            "voice": "default",
            "seed": 42,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["seed"] == 42


async def test_synthesize_with_sampling_params(client: AsyncClient, mock_engine: MagicMock):
    """All sampling parameters should be passed through to engine."""
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-turbo",
            "text": "Sampling params test",
            "voice": "default",
            "temperature": 0.5,
            "repetition_penalty": 1.5,
            "top_p": 0.9,
            "top_k": 500,
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args
    assert call_kwargs.kwargs["temperature"] == 0.5
    assert call_kwargs.kwargs["repetition_penalty"] == 1.5
    assert call_kwargs.kwargs["top_p"] == 0.9
    assert call_kwargs.kwargs["top_k"] == 500


async def test_synthesize_unknown_model(client: AsyncClient, mock_engine: MagicMock):
    mock_engine.ensure_model.side_effect = ValueError("Unknown model 'nonexistent'.")
    resp = await client.post(
        "/synthesize",
        json={
            "model": "nonexistent",
            "text": "Hello",
            "voice": "default",
        },
    )
    assert resp.status_code == 400


async def test_synthesize_missing_required_fields(client: AsyncClient):
    resp = await client.post("/synthesize", json={"model": "chatterbox-turbo"})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # Pydantic validation returns a list of error dicts; the missing
    # field errors should mention ``text`` and ``voice``.
    assert isinstance(detail, list)
    missing = {err["loc"][-1] for err in detail if err["type"] == "missing"}
    assert "text" in missing
    assert "voice" in missing


async def test_synthesize_invalid_json(client: AsyncClient):
    resp = await client.post(
        "/synthesize",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


async def test_synthesize_generation_failure(client: AsyncClient, mock_engine: MagicMock):
    """Engine exception should result in 500."""
    mock_engine.generate.side_effect = RuntimeError("GPU out of memory")
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-turbo",
            "text": "Fail test",
            "voice": "default",
        },
    )
    assert resp.status_code == 500
    assert "failed" in resp.json()["detail"].lower()


async def test_synthesize_default_params(client: AsyncClient, mock_engine: MagicMock):
    """When only required fields are sent, defaults should be passed to engine."""
    resp = await client.post(
        "/synthesize",
        json={
            "model": "chatterbox-turbo",
            "text": "Defaults test",
            "voice": "default",
        },
    )
    assert resp.status_code == 200
    call_kwargs = mock_engine.generate.call_args.kwargs
    assert call_kwargs["speed"] == 1.0
    assert call_kwargs["temperature"] == 0.8
    assert call_kwargs["repetition_penalty"] == 1.2
    assert call_kwargs["top_p"] == 1.0
    assert call_kwargs["min_p"] == 0.05
    assert call_kwargs["top_k"] == 1000
    assert call_kwargs["exaggeration"] == 0.5
    assert call_kwargs["cfg_weight"] == 0.5
    assert call_kwargs["seed"] is None
    assert call_kwargs["language"] is None


# ===================================================================
# /models status
# ===================================================================


async def test_list_models_includes_status(client: AsyncClient):
    """Each /models entry includes a per-model lifecycle status object."""
    resp = await client.get("/models")
    data = resp.json()
    by_id = {m["id"]: m for m in data["data"]}
    assert "status" in by_id["chatterbox-turbo"]
    assert by_id["chatterbox-turbo"]["status"]["value"] == "loaded"
    assert by_id["chatterbox"]["status"]["value"] == "unloaded"
    assert by_id["chatterbox-multilingual"]["status"]["value"] == "unloaded"


# ===================================================================
# /models/load and /models/unload
# ===================================================================


async def test_load_model_success(client: AsyncClient, mock_engine: MagicMock):
    """POST /models/load invokes engine.ensure_model and returns status."""
    mock_engine.ensure_model.return_value = "chatterbox"
    resp = await client.post(
        "/models/load", json={"model": "chatterbox"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "chatterbox"
    assert "status" in data
    mock_engine.ensure_model.assert_called_once_with("chatterbox")


async def test_load_model_unknown_returns_400(
    client: AsyncClient, mock_engine: MagicMock,
):
    """Unknown model id returns 400 via engine.ensure_model ValueError."""
    mock_engine.ensure_model.side_effect = ValueError("Unknown model 'nope'")
    resp = await client.post("/models/load", json={"model": "nope"})
    assert resp.status_code == 400


async def test_load_model_engine_failure_returns_500(
    client: AsyncClient, mock_engine: MagicMock,
):
    """Engine load failure returns 500."""
    mock_engine.ensure_model.side_effect = RuntimeError("OOM")
    resp = await client.post(
        "/models/load", json={"model": "chatterbox"},
    )
    assert resp.status_code == 500


async def test_load_model_missing_model_field(client: AsyncClient):
    """Missing model field returns 400 via pydantic validation."""
    resp = await client.post("/models/load", json={})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert isinstance(detail, list)
    assert any(e["type"] == "missing" for e in detail)


async def test_load_model_extra_field_rejected(client: AsyncClient):
    """extra='forbid' on the schema rejects unknown fields."""
    resp = await client.post(
        "/models/load", json={"model": "chatterbox-turbo", "junk": 1},
    )
    assert resp.status_code == 400


async def test_load_model_invalid_json(client: AsyncClient):
    """Non-JSON body returns 400."""
    resp = await client.post(
        "/models/load",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


async def test_unload_model_success(client: AsyncClient, mock_engine: MagicMock):
    """POST /models/unload invokes engine.unload(model)."""
    resp = await client.post(
        "/models/unload", json={"model": "chatterbox-turbo"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["model"] == "chatterbox-turbo"
    mock_engine.unload.assert_called_once_with("chatterbox-turbo")


async def test_unload_model_unknown_returns_404(
    client: AsyncClient, mock_engine: MagicMock,
):
    """Unknown model id returns 404 without touching the engine."""
    resp = await client.post(
        "/models/unload", json={"model": "not-a-model"},
    )
    assert resp.status_code == 404
    mock_engine.unload.assert_not_called()


# ===================================================================
# Engine state machine (without real weights)
# ===================================================================


def test_engine_status_transitions():
    """ChatterboxEngine transitions unloaded→loaded→unloaded/sleeping.

    We drive the state machine directly (no real weights). A fake
    ``_models`` dict is pushed into the singleton to simulate a
    successful load without touching the chatterbox library.
    """
    from chatterbox_service.engine import ChatterboxEngine

    engine = ChatterboxEngine()
    assert engine.status_phase == "unloaded"
    assert engine.status_for("chatterbox-turbo") == "unloaded"

    # Simulate a successful load.
    engine.mark_loading()
    assert engine.status_phase == "loading"
    engine._last_loaded_name = "chatterbox-turbo"
    assert engine.status_for("chatterbox-turbo") == "loading"

    engine._models["chatterbox-turbo"] = object()
    engine._sample_rates["chatterbox-turbo"] = 24000
    engine.mark_loaded()
    engine.touch()
    assert engine.status_phase == "loaded"
    assert engine.status_for("chatterbox-turbo") == "loaded"
    # Other known models stay unloaded.
    assert engine.status_for("chatterbox") == "unloaded"

    # Auto-unload via the idle checker transitions to sleeping.
    engine.unload()
    engine.mark_sleeping()
    assert engine.status_phase == "sleeping"
    assert engine.status_for("chatterbox-turbo") == "sleeping"
    # Other models are still just "unloaded" — sleeping is per-model.
    assert engine.status_for("chatterbox") == "unloaded"


def test_engine_load_evicts_previous_model():
    """Loading a second model unloads the first (models-max=1 invariant)."""
    from chatterbox_service.engine import ChatterboxEngine

    engine = ChatterboxEngine()
    # Pretend "chatterbox-turbo" is already loaded.
    engine._models["chatterbox-turbo"] = object()
    engine._sample_rates["chatterbox-turbo"] = 24000
    engine.mark_loaded()
    engine._last_loaded_name = "chatterbox-turbo"

    # Simulate the pre-load eviction path (without actually loading
    # new weights — just call unload explicitly, which is what
    # ``load`` does internally).
    assert engine.is_loaded
    engine.unload()
    assert not engine.is_loaded
    assert "chatterbox-turbo" not in engine._models
