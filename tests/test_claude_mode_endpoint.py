"""Unit testy pro Claude mode endpoint (POST /api/turn mode=claude).

Design (2026-05-19 redesign): claude mode = vždy edit, žádný permission
phrase/gate. Bezpečnost = workdir sandbox + claude --permission-mode acceptEdits.

Async endpoint testy s mockovaným adapterem byly removed; reálné integrační
pokrytí žije v src/claude_bridge/tests/integration/test_tmux_comprehensive.py
proti živému claude CLI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _isolated_env(monkeypatch, tmp_path):
    """Setup isolated WORKDIR + XDG_STATE_HOME + HOME tak aby testy neházely
    do reálného user state."""
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(fake_home))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)
    return wd, xdg, fake_home, config, server


# ──────────────── State schema (post-redesign) ────────────────

def test_claude_ui_state_default(monkeypatch, tmp_path):
    """Default state má jen claude_session_id + model. Žádné permission_mode/
    destructive_approved/approved_at (legacy fields odstraněny)."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    state = server._load_claude_ui_state()
    assert state == {"claude_session_id": None, "model": "opus"}


def test_claude_ui_state_persist_round_trip(monkeypatch, tmp_path):
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    new_state = {"claude_session_id": "claude_abc123", "model": "sonnet"}
    server._save_claude_ui_state(new_state)
    loaded = server._load_claude_ui_state()
    assert loaded == new_state


def test_claude_ui_state_corrupt_file(monkeypatch, tmp_path):
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    path = server._claude_ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json")
    state = server._load_claude_ui_state()
    assert state == {"claude_session_id": None, "model": "opus"}


def test_claude_ui_state_strips_legacy_fields(monkeypatch, tmp_path):
    """Pre-redesign state file (s permission_mode/destructive_approved/
    approved_at) je transparentně normalizován na nový schéma při load."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    path = server._claude_ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "permission_mode": "edit",
        "destructive_approved": True,
        "approved_at": 1234567890.0,
        "claude_session_id": "legacy_sid",
        "model": "haiku",
    }
    path.write_text(json.dumps(legacy))
    state = server._load_claude_ui_state()
    # Legacy fields stripped, jen session_id + model survived
    assert state == {"claude_session_id": "legacy_sid", "model": "haiku"}
    assert "permission_mode" not in state
    assert "destructive_approved" not in state
    assert "approved_at" not in state


# ──────────────── State path safety (XDG vs workdir) ────────────────

def test_claude_ui_state_path_sibling_xdg_accepted(monkeypatch, tmp_path):
    """Sibling XDG dir (= mimo workdir) je VALID."""
    _, xdg, _, config, server = _isolated_env(monkeypatch, tmp_path)
    path = server._claude_ui_state_path()
    assert str(xdg) in str(path), f"sibling xdg ignored: path={path}, xdg={xdg}"
    workdir_resolved = config.WORKDIR.resolve()
    try:
        path.resolve().relative_to(workdir_resolved)
        assert False, f"path {path} je v workdir {workdir_resolved}"
    except ValueError:
        pass


def test_claude_ui_state_path_xdg_inside_workdir_rejected(monkeypatch, tmp_path):
    """Pokud XDG_STATE_HOME je UVNITŘ workdir, hardening guard fallbacke mimo."""
    wd = tmp_path / "workdir"
    wd.mkdir()
    bad_xdg = wd / "xdg_inside"
    bad_xdg.mkdir()
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(bad_xdg))
    monkeypatch.setenv("HOME", str(fake_home))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    path = server._claude_ui_state_path()
    workdir_resolved = config.WORKDIR.resolve()
    try:
        path.resolve().relative_to(workdir_resolved)
        assert False, f"hardening failed: path {path} v workdir {workdir_resolved}"
    except ValueError:
        pass
    assert str(fake_home) in str(path) or "/tmp" in str(path)


# ──────────────── Update session_id helper ────────────────

@pytest.mark.asyncio
async def test_update_session_id_merge_preserves_model(monkeypatch, tmp_path):
    """_update_claude_session_id merge-zapíše jen session_id, model zůstane."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    server._save_claude_ui_state({"claude_session_id": None, "model": "sonnet"})
    await server._update_claude_session_id("new_sid_xyz")
    loaded = server._load_claude_ui_state()
    assert loaded["claude_session_id"] == "new_sid_xyz"
    assert loaded["model"] == "sonnet"


@pytest.mark.asyncio
async def test_update_session_id_to_none_clears(monkeypatch, tmp_path):
    """Update s None vymaže session_id (reset flow)."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    server._save_claude_ui_state({"claude_session_id": "existing", "model": "opus"})
    await server._update_claude_session_id(None)
    loaded = server._load_claude_ui_state()
    assert loaded["claude_session_id"] is None
    assert loaded["model"] == "opus"


@pytest.mark.asyncio
async def test_update_session_id_cas_blocks_when_prior_changed(monkeypatch, tmp_path):
    """CAS chrání proti race: pokud reset endpoint mezitím vyčistil state, turn
    NESMÍ overwriteu svým novým session_id zpět (codex HIGH #1, 2026-05-19)."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    # Turn začal s session_id="X"
    server._save_claude_ui_state({"claude_session_id": "X", "model": "opus"})
    # Reset mezitím vyčistil:
    server._save_claude_ui_state({"claude_session_id": None, "model": "opus"})
    # Turn se snaží zapsat NOVÉ Y s expected_prior="X" - CAS musí ZAMÍTNOUT
    written = await server._update_claude_session_id("Y", expected_prior="X")
    assert written is False, "CAS measly accepted overwrite po resetu"
    loaded = server._load_claude_ui_state()
    assert loaded["claude_session_id"] is None, "reset state přepsán"


@pytest.mark.asyncio
async def test_update_session_id_cas_passes_when_prior_matches(monkeypatch, tmp_path):
    """Normální happy path: CAS passuje když state je co turn očekává."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    server._save_claude_ui_state({"claude_session_id": "X", "model": "opus"})
    written = await server._update_claude_session_id("Y", expected_prior="X")
    assert written is True
    loaded = server._load_claude_ui_state()
    assert loaded["claude_session_id"] == "Y"


@pytest.mark.asyncio
async def test_update_session_id_no_cas_unconditional(monkeypatch, tmp_path):
    """Bez expected_prior argumentu (= default sentinel) zapisuje vždy."""
    _, _, _, _, server = _isolated_env(monkeypatch, tmp_path)
    server._save_claude_ui_state({"claude_session_id": "any", "model": "opus"})
    written = await server._update_claude_session_id("Z")
    assert written is True
    loaded = server._load_claude_ui_state()
    assert loaded["claude_session_id"] == "Z"
