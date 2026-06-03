"""OpenAI TTS backend — key loading, text cleaning, voice validation, HTTP mock."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice import openai_tts  # noqa: E402


# ──────────────── Key loading + priorita ────────────────

def test_key_from_gemma_file_wins_over_env(monkeypatch, tmp_path):
    """~/.gemma-openai-key má přednost před env (dedikovaný test klíč)."""
    kf = tmp_path / ".gemma-openai-key"
    kf.write_text("sk-FROM-FILE")
    kf.chmod(0o600)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-FROM-ENV")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    assert openai_tts.get_api_key() == "sk-FROM-FILE"


def test_key_falls_back_to_env(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ENV-ONLY")
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    assert openai_tts.get_api_key() == "sk-ENV-ONLY"


def test_key_file_world_readable_ignored(monkeypatch, tmp_path):
    kf = tmp_path / ".gemma-openai-key"
    kf.write_text("sk-INSECURE")
    kf.chmod(0o644)  # world readable → ignore
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    assert openai_tts.get_api_key() == ""


def test_key_override_path(monkeypatch, tmp_path):
    kf = tmp_path / "custom-key"
    kf.write_text("sk-CUSTOM")
    kf.chmod(0o600)
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(kf))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert openai_tts.get_api_key() == "sk-CUSTOM"


def test_is_available(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    assert openai_tts.is_available() is False
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert openai_tts.is_available() is True


# ──────────────── Text cleaning (BEZ num2words/fonetiky) ────────────────

def test_clean_keeps_numbers_and_english():
    """OpenAI si poradí s čísly i angličtinou → necháváme je být."""
    out = openai_tts._clean_text("Stojí 9875 Kč, udělej commit.")
    assert "9875" in out          # čísla zůstanou (žádný num2words)
    assert "commit" in out        # angličtina zůstane (žádná fonetika)


def test_clean_strips_markdown():
    out = openai_tts._clean_text("**Tučně** a `kód` a #nadpis")
    assert "*" not in out and "`" not in out and "#" not in out


def test_clean_truncates_long():
    out = openai_tts._clean_text("a" * 9000)
    assert len(out) <= openai_tts._MAX_INPUT_CHARS


# ──────────────── Synth (HTTP mock) ────────────────

def _mock_client(handler):
    captured = []
    def _w(req):
        captured.append(req)
        return handler(req)
    transport = httpx.MockTransport(_w)
    orig = openai_tts.httpx.Client
    class _C(orig):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)
    openai_tts.httpx.Client = _C
    return captured, (lambda: setattr(openai_tts.httpx, "Client", orig))


def test_synth_posts_correct_payload(monkeypatch, tmp_path):
    monkeypatch.setenv = None
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "sk-test")
    cap, restore = _mock_client(lambda r: httpx.Response(200, content=b"RIFFfakewav"))
    try:
        out = tmp_path / "o.wav"
        res = openai_tts.synth_openai_blocking(
            "Ahoj 9875 commit", out, voice="nova", model="gpt-4o-mini-tts", lang="cs"
        )
        assert res == out
        assert out.read_bytes() == b"RIFFfakewav"
        import json
        body = json.loads(cap[0].content)
        assert body["voice"] == "nova"
        assert body["model"] == "gpt-4o-mini-tts"
        assert body["response_format"] == "wav"
        assert "9875" in body["input"] and "commit" in body["input"]
        assert "instructions" in body  # gpt-4o-mini-tts → instructions
        assert cap[0].headers["authorization"] == "Bearer sk-test"
    finally:
        restore()


def test_synth_invalid_voice_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "sk-test")
    cap, restore = _mock_client(lambda r: httpx.Response(200, content=b"wav"))
    try:
        openai_tts.synth_openai_blocking("x", tmp_path / "o.wav", voice="neexistuje")
        import json
        assert json.loads(cap[0].content)["voice"] == openai_tts.DEFAULT_VOICE
    finally:
        restore()


def test_synth_no_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "")
    with pytest.raises(openai_tts.OpenAITTSError, match="OPENAI_API_KEY"):
        openai_tts.synth_openai_blocking("x", tmp_path / "o.wav")


def test_synth_api_error_surfaces(monkeypatch, tmp_path):
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "sk-test")
    cap, restore = _mock_client(
        lambda r: httpx.Response(401, json={"error": {"message": "Invalid key"}})
    )
    try:
        with pytest.raises(openai_tts.OpenAITTSError, match="Invalid key"):
            openai_tts.synth_openai_blocking("x", tmp_path / "o.wav")
    finally:
        restore()


def test_synth_empty_text_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "sk-test")
    assert openai_tts.synth_openai_blocking("   ", tmp_path / "o.wav") is None


def test_synth_cancel_preflight(monkeypatch, tmp_path):
    import threading
    monkeypatch.setattr(openai_tts, "get_api_key", lambda: "sk-test")
    ev = threading.Event(); ev.set()
    assert openai_tts.synth_openai_blocking("x", tmp_path / "o.wav", cancel_event=ev) is None
