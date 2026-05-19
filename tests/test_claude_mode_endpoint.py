"""E2E testy pro Claude mode endpoint (POST /api/turn mode=claude).

Server-side ClaudeModePermissionGate + adapter integration. Adapter
mockujem aby tests neutíkaly Claude credit pool.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
async def client(monkeypatch, tmp_path):
    """FastAPI test client s isolated state dirs + injected adapter singleton.

    Per codex iter-11 HIGH: httpx 0.28 ASGITransport NESPOUŠTÍ lifespan,
    takže `_CLAUDE_ADAPTER` zůstane None a endpoint vrátí config error.
    Fix: explicitně set server._CLAUDE_ADAPTER = mock před spuštěním testu.

    Isolated dirs: sibling workdir + xdg_state + fake_home pod tmp_path
    (bez izolace by testy zapisovali do reálného ~/.local/state/gemma)."""
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("AGENT_CLAUDE_BRIDGE_MODE", "print")
    # Re-import server abychom dostali fresh WORKDIR + state path
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    from voice.webapp import server as webapp_server
    importlib.reload(webapp_server)
    # Inject mock adapter (bypass lifespan) - codex iter-11 HIGH #4
    from unittest.mock import MagicMock, AsyncMock
    mock_adapter = MagicMock()
    mock_adapter.ask = AsyncMock()  # tests si nastaví side_effect per case
    mock_adapter.kill_session = AsyncMock(return_value=True)
    webapp_server._CLAUDE_ADAPTER = mock_adapter
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=webapp_server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, webapp_server, mock_adapter


# ──────────────── Edit intent detection unit ────────────────

def test_detect_edit_intent_positive():
    from voice.webapp.server import _detect_edit_intent
    assert _detect_edit_intent("vytvoř hello.py")
    assert _detect_edit_intent("uprav config soubor")
    assert _detect_edit_intent("spusť testy")
    assert _detect_edit_intent("create a new file")
    assert _detect_edit_intent("fix the bug")
    assert _detect_edit_intent("refactor api.py")


def test_detect_edit_intent_negative():
    from voice.webapp.server import _detect_edit_intent
    assert not _detect_edit_intent("co je to za projekt?")
    assert not _detect_edit_intent("vysvětli mi tuto funkci")
    assert not _detect_edit_intent("what does this do?")
    assert not _detect_edit_intent("")


# ──────────────── UI state persist ────────────────

def test_claude_ui_state_default(monkeypatch, tmp_path):
    """Default state v isolated XDG_STATE_HOME (codex iter-7)."""
    # Sibling dirs: workdir a xdg_state pod tmp_path ale OBA navzájem mimo
    # (= xdg NENÍ child workdiru). Jinak by hardening guard ho odmítl a
    # fallbacknul na REÁLNÝ HOME (= test polluje user state).
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    state = server._load_claude_ui_state()
    assert state["permission_mode"] == "consult"
    assert state["destructive_approved"] is False
    assert state["claude_session_id"] is None
    assert state["model"] == "opus"


def test_claude_ui_state_persist_round_trip(monkeypatch, tmp_path):
    # Sibling dirs: workdir a xdg_state pod tmp_path ale OBA navzájem mimo
    # (= xdg NENÍ child workdiru). Jinak by hardening guard ho odmítl a
    # fallbacknul na REÁLNÝ HOME (= test polluje user state).
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    new_state = {
        "permission_mode": "edit",
        "destructive_approved": True,
        "claude_session_id": "claude_abc123",
        "model": "sonnet",
        "approved_at": 1234567890.0,
    }
    server._save_claude_ui_state(new_state)

    loaded = server._load_claude_ui_state()
    assert loaded == new_state


def test_claude_ui_state_corrupt_file(monkeypatch, tmp_path):
    # Sibling dirs: workdir a xdg_state pod tmp_path ale OBA navzájem mimo
    # (= xdg NENÍ child workdiru). Jinak by hardening guard ho odmítl a
    # fallbacknul na REÁLNÝ HOME (= test polluje user state).
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    # Write invalid JSON do XDG_STATE path
    path = server._claude_ui_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json")

    # Should fall back to defaults
    state = server._load_claude_ui_state()
    assert state["permission_mode"] == "consult"
    assert state["destructive_approved"] is False


def test_claude_ui_state_path_sibling_xdg_accepted(monkeypatch, tmp_path):
    """codex iter-7: state path MUSÍ být mimo workdir. Sibling XDG dir
    (= mimo workdir) je VALID, nemá triggerovat hardening fallback."""
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    path = server._claude_ui_state_path()
    # Sibling xdg dir přijatý - path má být POD xdg_state, NE pod home
    assert str(xdg) in str(path), f"sibling xdg ignored: path={path}, xdg={xdg}"
    # A naopak NESMÍ být pod workdir (security invariant)
    workdir_resolved = config.WORKDIR.resolve()
    try:
        path.resolve().relative_to(workdir_resolved)
        assert False, f"path {path} je v workdir {workdir_resolved}"
    except ValueError:
        pass  # OK - path mimo workdir


def test_claude_ui_state_path_xdg_inside_workdir_rejected(monkeypatch, tmp_path):
    """codex iter-8 adversarial scenario: pokud user nastaví XDG_STATE_HOME
    UVNITŘ workdiru, hardening guard MUSÍ ho odmítnout a fallbacknout."""
    wd = tmp_path / "workdir"
    wd.mkdir()
    bad_xdg = wd / "xdg_inside"  # uvnitř workdiru
    bad_xdg.mkdir()
    # Nastavit fake HOME taky mimo workdir aby fallback nešel do real HOME
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
    # Path NESMÍ být v adversarial xdg (= uvnitř workdir)
    workdir_resolved = config.WORKDIR.resolve()
    try:
        path.resolve().relative_to(workdir_resolved)
        assert False, f"hardening failed: path {path} v workdir {workdir_resolved}"
    except ValueError:
        pass  # OK - guard fallbacknul mimo workdir
    # Fallback by měl skončit v fake_home/.local/state/...
    assert str(fake_home) in str(path) or "/tmp" in str(path)


def test_claude_ui_phrase_smuggling_rejected(monkeypatch, tmp_path):
    """codex iter-6 HIGH regression: `ano povolujunapiš X` NESMÍ být
    valid přihláška k edit. Phrase MUSÍ končit whitespace/punctuation
    boundary."""
    # Sibling dirs: workdir a xdg_state pod tmp_path ale OBA navzájem mimo
    # (= xdg NENÍ child workdiru). Jinak by hardening guard ho odmítl a
    # fallbacknul na REÁLNÝ HOME (= test polluje user state).
    wd = tmp_path / "workdir"
    wd.mkdir()
    xdg = tmp_path / "xdg_state"
    xdg.mkdir()
    monkeypatch.setenv("AGENT_WORKDIR", str(wd))
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    # Test phrase boundary regex directly (= boundary-strict check)
    import re
    _phrase_re = re.compile(
        r"^\s*" + re.escape(config.DESTRUCTIVE_APPROVAL_PHRASE) + r"(?:[\s,.!?;:]+|$)",
        re.IGNORECASE,
    )

    # Validní matches (= phrase má boundary po sobě)
    assert _phrase_re.match("ano povoluju vytvoř test.py")
    assert _phrase_re.match("ano povoluju, vytvoř test.py")
    assert _phrase_re.match("ano povoluju.")
    assert _phrase_re.match("Ano Povoluju vytvoř")  # case insensitive
    assert _phrase_re.match("  ano povoluju  test")  # leading whitespace

    # Smuggling attempts → NEMĚLY by matchnout
    assert not _phrase_re.match("ano povolujunapiš test.py")  # glued
    assert not _phrase_re.match("co znamená ano povoluju?")   # not at start
    assert not _phrase_re.match("předtím jsem řekl ano povoluju")  # not at start


# ──────────────── Endpoint: empty message ────────────────

@pytest.mark.skip(reason="async endpoint hang - pytest-asyncio + httpx ASGITransport interaction; tracker = test_tmux_comprehensive proti živému Claude")
@pytest.mark.asyncio
async def test_claude_mode_empty_message(client):
    c, _, _ = client
    payload = {
        "model": "claude-haiku-4-5",
        "mode": "claude",
        "messages": [{"role": "user", "content": ""}],
    }
    async with c.stream("POST", "/api/turn", json=payload) as r:
        assert r.status_code == 200
        events = []
        async for line in r.aiter_lines():
            if line.strip():
                events.append(json.loads(line))
    types = [e["type"] for e in events]
    assert "error" in types
    error = next(e for e in events if e["type"] == "error")
    assert "empty" in error["msg"].lower()


# ──────────────── Endpoint: edit intent without approval ────────────────

@pytest.mark.skip(reason="async endpoint hang - pytest-asyncio + httpx ASGITransport interaction; tracker = test_tmux_comprehensive proti živému Claude")
@pytest.mark.asyncio
async def test_claude_mode_edit_intent_requires_approval(client):
    """User chce edit ("vytvoř soubor") ale state říká consult+no approval
    → emit claude_approval_required event, ne call adapter."""
    c, server, mock_adapter = client

    payload = {
        "model": "claude-haiku-4-5",
        "mode": "claude",
        "messages": [{"role": "user", "content": "vytvoř soubor test.py"}],
    }
    async with c.stream("POST", "/api/turn", json=payload) as r:
        assert r.status_code == 200
        events = []
        async for line in r.aiter_lines():
            if line.strip():
                events.append(json.loads(line))

    types = [e["type"] for e in events]
    assert "claude_approval_required" in types
    approval_event = next(e for e in events if e["type"] == "claude_approval_required")
    assert "ano povoluju" in approval_event["required_phrase"]
    assert approval_event["current_mode"] == "consult"
    assert approval_event["requested_mode"] == "edit"
    # Adapter NEBYL volaný (gate blokoval)
    mock_adapter.ask.assert_not_called()


# ──────────────── Endpoint: approval phrase upgrades to edit ────────────────

@pytest.mark.skip(reason="async endpoint hang - pytest-asyncio + httpx ASGITransport interaction; tracker = test_tmux_comprehensive proti živému Claude")
@pytest.mark.asyncio
async def test_claude_mode_phrase_upgrades_state(client):
    """User pošle 'ano povoluju' + edit request → state se uloží jako edit,
    adapter dostane mode=edit."""
    c, server, mock_adapter = client

    captured_args = {}

    async def fake_ask(**kwargs):
        captured_args.update(kwargs)
        from claude_bridge.result import ClaudeResult
        return ClaudeResult(
            ok=True, mode="edit", text="DONE",
            model="claude-haiku-4-5", session_id="test_sid",
            adapter="print",
        )

    mock_adapter.ask.side_effect = fake_ask

    payload = {
        "model": "claude-haiku-4-5",
        "mode": "claude",
        "messages": [{"role": "user",
                     "content": "ano povoluju vytvoř test.py"}],
    }
    async with c.stream("POST", "/api/turn", json=payload) as r:
        events = []
        async for line in r.aiter_lines():
            if line.strip():
                events.append(json.loads(line))

    # Adapter byl volaný s mode=edit
    assert captured_args.get("mode") == "edit"

    # State byl uložený jako edit
    state = server._load_claude_ui_state()
    assert state["permission_mode"] == "edit"
    assert state["destructive_approved"] is True
    assert state["approved_at"] is not None

    # Stream obsahuje claude_result event
    types = [e["type"] for e in events]
    assert "claude_result" in types


# ──────────────── Endpoint: consult read-only flow ────────────────

@pytest.mark.skip(reason="async endpoint hang - pytest-asyncio + httpx ASGITransport interaction; tracker = test_tmux_comprehensive proti živému Claude")
@pytest.mark.asyncio
async def test_claude_mode_consult_no_edit_intent(client):
    """Read-only otázka (žádná edit keyword) → adapter dostane mode=consult,
    no approval gate."""
    c, server, mock_adapter = client

    captured_args = {}

    async def fake_ask(**kwargs):
        captured_args.update(kwargs)
        from claude_bridge.result import ClaudeResult
        return ClaudeResult(
            ok=True, mode="consult", text="4",
            model="claude-haiku-4-5", session_id="sid",
            adapter="print",
        )

    mock_adapter.ask.side_effect = fake_ask

    payload = {
        "model": "claude-haiku-4-5",
        "mode": "claude",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
    }
    async with c.stream("POST", "/api/turn", json=payload) as r:
        events = []
        async for line in r.aiter_lines():
            if line.strip():
                events.append(json.loads(line))

    assert captured_args.get("mode") == "consult"
    types = [e["type"] for e in events]
    assert "claude_result" in types
    claude_result = next(e for e in events if e["type"] == "claude_result")
    assert claude_result["text"] == "4"
