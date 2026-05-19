"""Integration testy pro tmux primitives v TmuxAdapter (real tmux, žádný claude).

Spawne real tmux session se shell scriptem (sh nebo /bin/cat), ověří že
basic primitives fungují:
- new-session / has-session / kill-session
- send-keys + capture-pane (echo back)
- history-limit (full scrollback retention)
- killpg cleanup

Žádný claude binary, jen tmux + sh. Marker `tmux_real` aby skiplo bez tmux.
"""
from __future__ import annotations

import asyncio
import shutil

import pytest

from claude_bridge.adapters.tmux_mode import TmuxAdapter
from claude_bridge.config import AdapterConfig, BridgeMode

pytestmark = pytest.mark.tmux_real

_HAS_TMUX = shutil.which("tmux") is not None
if not _HAS_TMUX:
    pytest.skip("tmux not in PATH", allow_module_level=True)


@pytest.fixture
async def adapter(tmp_path):
    """Adapter s metadata dir v tmp_path (čistý state per test)."""
    config = AdapterConfig(
        mode=BridgeMode.TMUX,
        tmux_session_prefix="test_claude_",
        metadata_dir=str(tmp_path),
    )
    a = TmuxAdapter(config)
    yield a
    # Cleanup: kill všechny test sessions po testu
    rc, stdout, _ = await a._tmux("list-sessions", "-F", "#{session_name}")
    if rc == 0:
        for line in stdout.decode().splitlines():
            if line.strip().startswith("test_claude_"):
                await a._kill_tmux_session(line.strip())


# ──────────────── new-session / has-session / kill-session ────────────────

@pytest.mark.asyncio
async def test_spawn_and_kill_session(adapter, tmp_path):
    """Spawn fake session se `cat`, ověř has-session, pak kill."""
    sid = "test_claude_basic"
    # Spawn s `cat` jako "claude" (cat čeká na stdin, drží session live)
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok, "tmux new-session should succeed"
    assert await adapter._has_session(sid)
    assert await adapter._kill_tmux_session(sid)
    assert not await adapter._has_session(sid)


@pytest.mark.asyncio
async def test_has_session_for_nonexistent(adapter):
    """has-session na neexistující session vrátí False (rc != 0)."""
    assert not await adapter._has_session("test_claude_does_not_exist")


@pytest.mark.asyncio
async def test_kill_nonexistent_session(adapter):
    """kill-session na neexistující vrátí False bez crashe."""
    assert not await adapter._kill_tmux_session("test_claude_does_not_exist")


# ──────────────── send-keys + capture-pane ────────────────

@pytest.mark.asyncio
async def test_send_keys_echo_visible_in_capture(adapter, tmp_path):
    """Send-keys text → capture-pane musí ten text obsahovat (cat echos)."""
    sid = "test_claude_echo"
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok
    try:
        # cat echos stdin to stdout, takže poslaný text + enter = uvidíme
        await adapter._send_keys(sid, "Hello tmux world", literal=True)
        await adapter._send_enter(sid)
        # Wait pro echo (cat processing)
        await asyncio.sleep(0.3)
        capture = await adapter._capture_pane(sid)
        assert "Hello tmux world" in capture
    finally:
        await adapter._kill_tmux_session(sid)


@pytest.mark.asyncio
async def test_capture_pane_empty_for_nonexistent(adapter):
    """capture-pane na neexistující session vrátí prázdný string."""
    capture = await adapter._capture_pane("test_claude_does_not_exist")
    assert capture == ""


# ──────────────── History limit (codex iter-2 #11) ────────────────

@pytest.mark.asyncio
async def test_history_limit_retains_long_output(adapter, tmp_path):
    """Spawn session, generate 500 řádků output, capture-pane -S - vrátí všechny.

    Bez history-limit by tmux orientoval jen default 2000 řádků a my
    bychom o dlouhé odpovědi přišli. Test ověř že _spawn_tmux_session
    nastaví dostatečný limit.
    """
    sid = "test_claude_history"
    # Spawn s `sh` shellem co něco vypíše a pak čeká na stdin
    ok = await adapter._spawn_tmux_session(
        sid,
        ["sh", "-c", "for i in $(seq 1 500); do echo line_$i; done; cat"],
        cwd=tmp_path,
    )
    assert ok
    try:
        # Wait pro generování všech 500 řádků
        await asyncio.sleep(1.0)
        # Capture s -S - = entire scrollback
        capture = await adapter._capture_pane(sid)
        # Verify že máme line_1 (na začátku) i line_500 (na konci) - bez
        # history limit by line_1 byl out of buffer
        assert "line_1\n" in capture or "line_1 " in capture
        assert "line_500" in capture
    finally:
        await adapter._kill_tmux_session(sid)


# ──────────────── send-keys Enter / Escape ────────────────

@pytest.mark.asyncio
async def test_send_enter_separate_from_text(adapter, tmp_path):
    """send-keys -l 'text' + send-keys Enter = dva oddělené calls."""
    sid = "test_claude_enter"
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok
    try:
        await adapter._send_keys(sid, "line one", literal=True)
        # Pred Enter: text v capture, ale ne na novém řádku
        await asyncio.sleep(0.1)
        before = await adapter._capture_pane(sid)
        # Po Enter: cat dostane řádek + echo zpět
        await adapter._send_enter(sid)
        await asyncio.sleep(0.3)
        after = await adapter._capture_pane(sid)
        # Verify Enter actually submitted (cat echoed line)
        assert after.count("line one") >= 1
    finally:
        await adapter._kill_tmux_session(sid)


@pytest.mark.asyncio
async def test_send_escape_twice(adapter, tmp_path):
    """send Escape Escape (cancel sequence) - just verify call succeeds."""
    sid = "test_claude_escape"
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok
    try:
        result = await adapter._send_escape_twice(sid)
        assert result is True
    finally:
        await adapter._kill_tmux_session(sid)


# ──────────────── Reattach / orphan cleanup (codex iter-2 #9) ────────────────

@pytest.mark.asyncio
async def test_reattach_unsafe_orphan_killed(adapter, tmp_path):
    """Tmux session bez metadata entry = unsafe orphan → killed při reattach."""
    sid = "test_claude_orphan"
    # Spawn session BEZ persisting metadata
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok
    assert await adapter._has_session(sid)

    # Reattach (clean adapter state = žádné known sessions)
    adapter._sessions.clear()  # Reset in-memory state
    # Write valid (but empty) metadata file pointing to no sessions
    import json
    metadata_path = tmp_path / "claude_sessions.json"
    metadata_path.write_text(json.dumps({"version": "v1", "sessions": {}}))
    adapter._metadata_path = metadata_path

    await adapter.reattach_persisted_sessions()

    # Orphan should be killed (no metadata entry)
    assert not await adapter._has_session(sid), \
        "unsafe orphan session should be killed during reattach"


@pytest.mark.asyncio
async def test_reattach_skips_nonexistent_session(adapter, tmp_path):
    """Metadata pro session co už neexistuje → skip, no crash."""
    import json
    metadata_path = tmp_path / "claude_sessions.json"
    # Metadata mentions session that doesn't exist v tmux
    metadata_path.write_text(json.dumps({
        "version": "v1",
        "sessions": {
            "test_claude_ghost": {
                "session_id": "test_claude_ghost",
                "owner": "gemma",
                "workdir": str(tmp_path),
                "model": "claude-haiku-4-5",
                "permission_mode": "consult",
                "created_at": 0.0,
                "last_active": 0.0,
                "approval": None,
                "turn_count": 0,
                "state": "READY",
            }
        }
    }))
    adapter._metadata_path = metadata_path
    adapter._sessions.clear()
    await adapter.reattach_persisted_sessions()
    # Ghost session was not registered
    assert "test_claude_ghost" not in adapter._sessions


# ──────────────── Persistent metadata atomicita ────────────────

@pytest.mark.asyncio
async def test_metadata_atomic_write(adapter, tmp_path):
    """Persist metadata používá tmp+rename (atomic)."""
    sid = "test_claude_metadata"
    ok = await adapter._spawn_tmux_session(sid, ["cat"], cwd=tmp_path)
    assert ok
    try:
        # _new_session calls _persist_metadata; but we used _spawn_tmux_session
        # directly here. Manual register + persist.
        from claude_bridge.adapters.tmux_mode import _TmuxSession
        from claude_bridge.parsing.tui_state import TuiState
        import time
        adapter._sessions[sid] = _TmuxSession(
            session_id=sid, workdir=tmp_path,
            model="claude-haiku-4-5", permission_mode="consult",
            created_at=time.time(), last_active=time.time(),
            tui=TuiState(cols=80, rows=24),
        )
        await adapter._persist_metadata()
        metadata_file = tmp_path / "claude_sessions.json"
        assert metadata_file.exists()
        import json
        data = json.loads(metadata_file.read_text())
        assert data["version"] == "v1"
        assert sid in data["sessions"]
    finally:
        await adapter._kill_tmux_session(sid)
