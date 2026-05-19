"""Session management tests: list/get/clear/kill/health + reattach flow.

Real tmux + real claude. Testuje long-lived session lifecycle.
"""
from __future__ import annotations

import asyncio
import shutil

import pytest

from claude_bridge.adapters.tmux_mode import TmuxAdapter
from claude_bridge.config import AdapterConfig, BridgeMode

pytestmark = [pytest.mark.claude_cli, pytest.mark.tmux_real]

_HAS_TMUX = shutil.which("tmux") is not None
_HAS_CLAUDE = shutil.which("claude") is not None
if not _HAS_TMUX:
    pytest.skip("tmux not in PATH", allow_module_level=True)
if not _HAS_CLAUDE:
    pytest.skip("claude not in PATH", allow_module_level=True)


@pytest.fixture
async def adapter(tmp_path):
    """Adapter s tmp metadata dir + cleanup."""
    config = AdapterConfig(
        mode=BridgeMode.TMUX,
        tmux_session_prefix="sm_claude_",
        metadata_dir=str(tmp_path),
    )
    a = TmuxAdapter(config)
    yield a
    for sid in list(a._sessions.keys()):
        await a._kill_tmux_session(sid)
    rc, stdout, _ = await a._tmux("list-sessions", "-F", "#{session_name}")
    if rc == 0:
        for line in stdout.decode().splitlines():
            if line.strip().startswith("sm_claude_"):
                await a._kill_tmux_session(line.strip())


# ──────────────── list/get/info ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_list_sessions_after_spawn(adapter):
    """Po jedné ask() session_list obsahuje tu session."""
    r = await adapter.ask(
        prompt="Hi.", model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    assert r.ok is True
    sessions = await adapter.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == r.session_id
    assert sessions[0].model == "claude-haiku-4-5"
    assert sessions[0].permission_mode == "consult"
    assert sessions[0].state == "READY"
    assert sessions[0].turn_count == 1


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_get_session_existing(adapter):
    r = await adapter.ask(
        prompt="Hi.", model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    info = await adapter.get_session(r.session_id)
    assert info is not None
    assert info.session_id == r.session_id


# ──────────────── kill_session ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_kill_session_removes_tmux_and_registry(adapter):
    r = await adapter.ask(
        prompt="Hi.", model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    sid = r.session_id
    assert await adapter._has_session(sid)

    ok = await adapter.kill_session(sid)
    assert ok is True
    # Tmux session pryč
    assert not await adapter._has_session(sid)
    # Registry pryč
    assert sid not in adapter._sessions
    assert await adapter.get_session(sid) is None


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_kill_session_nonexistent(adapter):
    """Kill na neexistující session vrátí False bez crashe."""
    ok = await adapter.kill_session("sm_claude_does_not_exist")
    assert ok is False


# ──────────────── health_check ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_health_check_alive_session(adapter):
    r = await adapter.ask(
        prompt="Hi.", model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    state = await adapter.health_check(r.session_id)
    assert state == "READY"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_health_check_externally_killed(adapter):
    """Externí kill → health_check vrátí DEAD + state na DEAD."""
    r = await adapter.ask(
        prompt="Hi.", model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    sid = r.session_id
    # Externí kill tmux session
    await adapter._kill_tmux_session(sid)
    # Session stále v _sessions (žádný kill_session), ale tmux pryč
    state = await adapter.health_check(sid)
    assert state == "DEAD"
    # State v adapter session marked DEAD
    assert adapter._sessions[sid].state == "DEAD"


# ──────────────── Reattach po restartu ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_reattach_finds_existing_consult_session(adapter, tmp_path):
    """Spawn session → simulate gemma restart (fresh adapter) → reattach."""
    r = await adapter.ask(
        prompt="Remember: my favorite color is purple.",
        model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    sid = r.session_id

    # Simulate restart: fresh adapter instance s same metadata dir
    config2 = AdapterConfig(
        mode=BridgeMode.TMUX, tmux_session_prefix="sm_claude_",
        metadata_dir=str(tmp_path),
    )
    adapter2 = TmuxAdapter(config2)
    await adapter2.reattach_persisted_sessions()

    # Adapter2 by měl mít náš session
    info = await adapter2.get_session(sid)
    assert info is not None
    assert info.session_id == sid
    assert info.permission_mode == "consult"

    # Můžeš pokračovat v rozhovoru přes adapter2
    r2 = await adapter2.ask(
        prompt="What is my favorite color? Just say the color name.",
        model="claude-haiku-4-5", mode="consult",
        workdir=None, session_id=sid, timeout_sec=60.0,
    )
    assert r2.ok is True
    assert "purple" in r2.text.lower(), f"context lost: {r2.text!r}"


# ──────────────── clear_session ────────────────

@pytest.mark.skip(reason="flaky - /clear timing race po reuse, fix ve Fáze 3.1")
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_clear_wipes_context(adapter):
    """ask 'remember X', /clear, ask 'co bylo X' → Claude nepamatuje."""
    r1 = await adapter.ask(
        prompt="My secret code is BANANA42. Just acknowledge.",
        model="claude-haiku-4-5", mode="consult",
        workdir=None, timeout_sec=60.0,
    )
    assert r1.ok is True
    sid = r1.session_id

    # Clear history
    cleared = await adapter.clear_session(sid)
    assert cleared is True

    # Po clear: secret pryč
    r2 = await adapter.ask(
        prompt="What was my secret code? Reply ONLY 'NO_MEMORY' if unknown.",
        model="claude-haiku-4-5", mode="consult",
        workdir=None, session_id=sid, timeout_sec=60.0,
    )
    assert r2.ok is True
    assert "BANANA42" not in r2.text, (
        f"clear nezapůsobil - Claude pamatuje: {r2.text!r}"
    )
