"""Tests for the voice management service."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

# Minimal valid WAV file (44-byte header + 2 bytes of silence)
_MINIMAL_WAV = (
    b"RIFF"
    b"\x2e\x00\x00\x00"  # file size - 8
    b"WAVE"
    b"fmt "
    b"\x10\x00\x00\x00"  # chunk size
    b"\x01\x00"  # PCM
    b"\x01\x00"  # mono
    b"\x80\x3e\x00\x00"  # 16000 Hz
    b"\x00\x7d\x00\x00"  # byte rate
    b"\x02\x00"  # block align
    b"\x10\x00"  # 16 bit
    b"data"
    b"\x02\x00\x00\x00"  # data size
    b"\x00\x00"  # one sample of silence
)


@pytest.fixture()
def app(tmp_path):
    """Create a test app with voices_dir pointing at a temp directory."""
    with patch("voice_service.app.settings") as mock_settings:
        mock_settings.voices_dir = tmp_path
        from voice_service.app import app

        yield app


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# List / Get
# ---------------------------------------------------------------------------


async def test_list_voices_empty(client: AsyncClient):
    resp = await client.get("/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    # "default" is always present as a virtual built-in voice
    assert len(data["data"]) == 1
    assert data["data"][0]["name"] == "default"


async def test_list_voices(client: AsyncClient, tmp_path):
    (tmp_path / "alice.wav").write_bytes(_MINIMAL_WAV)
    (tmp_path / "bob.wav").write_bytes(_MINIMAL_WAV)
    resp = await client.get("/voices")
    assert resp.status_code == 200
    names = [v["name"] for v in resp.json()["data"]]
    # "default" is always first, then disk voices sorted alphabetically
    assert names == ["default", "alice", "bob"]


async def test_get_voice(client: AsyncClient, tmp_path):
    (tmp_path / "alice.wav").write_bytes(_MINIMAL_WAV)
    resp = await client.get("/voices/alice")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "alice"
    assert data["name"] == "alice"


async def test_get_voice_not_found(client: AsyncClient):
    resp = await client.get("/voices/nonexistent")
    assert resp.status_code == 404


async def test_get_default_voice(client: AsyncClient):
    """The 'default' virtual voice should always be accessible."""
    resp = await client.get("/voices/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["voice_id"] == "default"
    assert data["name"] == "default"


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


async def test_create_voice(client: AsyncClient, tmp_path):
    resp = await client.post(
        "/voices",
        data={"name": "newvoice"},
        files={"file": ("sample.wav", _MINIMAL_WAV, "audio/wav")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["voice_id"] == "newvoice"
    assert (tmp_path / "newvoice.wav").exists()


async def test_create_voice_invalid_name(client: AsyncClient):
    resp = await client.post(
        "/voices",
        data={"name": "bad name!"},
        files={"file": ("sample.wav", _MINIMAL_WAV, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "alphanumeric" in resp.json()["detail"]


async def test_create_voice_duplicate(client: AsyncClient, tmp_path):
    (tmp_path / "existing.wav").write_bytes(_MINIMAL_WAV)
    resp = await client.post(
        "/voices",
        data={"name": "existing"},
        files={"file": ("sample.wav", _MINIMAL_WAV, "audio/wav")},
    )
    assert resp.status_code == 409


async def test_create_voice_not_wav(client: AsyncClient):
    resp = await client.post(
        "/voices",
        data={"name": "bad"},
        files={"file": ("sample.mp3", b"\xff\xfb\x90\x00" + b"\x00" * 100, "audio/mpeg")},
    )
    assert resp.status_code == 400
    assert "not a valid WAV" in resp.json()["detail"]


async def test_create_voice_too_small(client: AsyncClient):
    resp = await client.post(
        "/voices",
        data={"name": "tiny"},
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )
    assert resp.status_code == 400
    assert "too small" in resp.json()["detail"].lower()


async def test_create_voice_default_rejected(client: AsyncClient):
    """Creating a voice named 'default' should be rejected (reserved name)."""
    resp = await client.post(
        "/voices",
        data={"name": "default"},
        files={"file": ("sample.wav", _MINIMAL_WAV, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "built-in" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


async def test_delete_voice(client: AsyncClient, tmp_path):
    voice_file = tmp_path / "todelete.wav"
    voice_file.write_bytes(_MINIMAL_WAV)
    resp = await client.delete("/voices/todelete")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] is True
    assert data["voice_id"] == "todelete"
    assert not voice_file.exists()


async def test_delete_voice_not_found(client: AsyncClient):
    resp = await client.delete("/voices/nonexistent")
    assert resp.status_code == 404


async def test_delete_default_voice_rejected(client: AsyncClient):
    """Deleting the 'default' voice should be rejected (built-in)."""
    resp = await client.delete("/voices/default")
    assert resp.status_code == 400
    assert "built-in" in resp.json()["detail"]
