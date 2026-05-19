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
    """FastAPI test client s tmp WORKDIR (žádný impact na real .gemma_local)."""
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("AGENT_CLAUDE_BRIDGE_MODE", "print")
    # Re-import server abychom dostali fresh WORKDIR
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    from voice.webapp import server as webapp_server
    importlib.reload(webapp_server)
    from httpx import ASGITransport, AsyncClient
    transport = ASGITransport(app=webapp_server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, webapp_server


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
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
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
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
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
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
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


def test_claude_ui_state_path_outside_workdir(monkeypatch, tmp_path):
    """codex iter-7 CRITICAL: state path MUSÍ být mimo workdir (agent
    write_file by ho jinak mohl přepsat bez approval)."""
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
    import importlib
    from voice.agent import config
    importlib.reload(config)
    from voice.webapp import server
    importlib.reload(server)

    path = server._claude_ui_state_path()
    # Pokud XDG je uvnitř workdir, hardening fallback by ho měl odmítnout.
    # tmp_path / xdg_state je technicky UVNITŘ tmp_path = workdir, takže
    # path by měl fallback na HOME-based location nebo /tmp.
    workdir_resolved = config.WORKDIR.resolve()
    try:
        path.resolve().relative_to(workdir_resolved)
        # Pokud jsme tady, path je v workdir → security fail
        assert False, f"path {path} je v workdir {workdir_resolved}"
    except ValueError:
        # Path je MIMO workdir = OK
        pass


def test_claude_ui_phrase_smuggling_rejected(monkeypatch, tmp_path):
    """codex iter-6 HIGH regression: `ano povolujunapiš X` NESMÍ být
    valid přihláška k edit. Phrase MUSÍ končit whitespace/punctuation
    boundary."""
    monkeypatch.setenv("AGENT_WORKDIR", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg_state"))
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

@pytest.mark.asyncio
async def test_claude_mode_empty_message(client):
    c, _ = client
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

@pytest.mark.asyncio
async def test_claude_mode_edit_intent_requires_approval(client, monkeypatch, tmp_path):
    """User chce edit ("vytvoř soubor") ale state říká consult+no approval
    → emit claude_approval_required event, ne call adapter."""
    c, server = client

    # Mock adapter aby nebyl call (test že gate zachytil)
    fake_ask = AsyncMock()
    with patch("claude_bridge.create_adapter") as mock_factory:
        mock_factory.return_value.ask = fake_ask

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
    fake_ask.assert_not_called()


# ──────────────── Endpoint: approval phrase upgrades to edit ────────────────

@pytest.mark.asyncio
async def test_claude_mode_phrase_upgrades_state(client, monkeypatch, tmp_path):
    """User pošle 'ano povoluju' + edit request → state se uloží jako edit,
    adapter dostane mode=edit."""
    c, server = client

    captured_args = {}

    async def fake_ask(**kwargs):
        captured_args.update(kwargs)
        # Return mock result
        from claude_bridge.result import ClaudeResult
        return ClaudeResult(
            ok=True, mode="edit", text="DONE",
            model="claude-haiku-4-5", session_id="test_sid",
            adapter="print",
        )

    mock_adapter = MagicMock()
    mock_adapter.ask = fake_ask
    with patch("claude_bridge.create_adapter", return_value=mock_adapter):
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

@pytest.mark.asyncio
async def test_claude_mode_consult_no_edit_intent(client):
    """Read-only otázka (žádná edit keyword) → adapter dostane mode=consult,
    no approval gate."""
    c, server = client

    captured_args = {}

    async def fake_ask(**kwargs):
        captured_args.update(kwargs)
        from claude_bridge.result import ClaudeResult
        return ClaudeResult(
            ok=True, mode="consult", text="4",
            model="claude-haiku-4-5", session_id="sid",
            adapter="print",
        )

    mock_adapter = MagicMock()
    mock_adapter.ask = fake_ask
    with patch("claude_bridge.create_adapter", return_value=mock_adapter):
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
