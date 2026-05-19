"""End-to-end test: TmuxAdapter proti REÁLNÉMU `claude` CLI v tmux session.

Tyto testy spawne live `claude` CLI v tmux pseudo-terminálu, simulujou
real user conversation, ověří že adapter správně:
- Detekuje ready prompt
- Sleduje thinking + tool_use progress
- Extract assistant text
- Cancel, timeout, persistence

POŽADAVKY:
- `tmux` binary (pacman -S tmux / apt install tmux)
- `claude` binary v PATH (Claude Code CLI)
- Aktivní OAuth session (`claude /login` proběhlý)
- Síť dostupná, Anthropic credity

⚠️ TOS NOTE: Tmux/PTY driving je experimentální per docs/plans/claude_tmux_adapter.md.
Spustit jen pro vývoj/testování. Default tmux=off, používáme pouze v test environment.

Marker: claude_cli + tmux_real, takže `pytest -m "claude_cli and tmux_real"`.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from claude_bridge.adapters.tmux_mode import TmuxAdapter
from claude_bridge.config import AdapterConfig, BridgeMode
from claude_bridge.progress import ProgressEvent

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
        tmux_session_prefix="rt_claude_",
        metadata_dir=str(tmp_path),
        default_timeout_sec=180.0,
    )
    a = TmuxAdapter(config)
    yield a
    # Cleanup všech našich test sessions
    for sid in list(a._sessions.keys()):
        await a._kill_tmux_session(sid)
    rc, stdout, _ = await a._tmux("list-sessions", "-F", "#{session_name}")
    if rc == 0:
        for line in stdout.decode().splitlines():
            if line.strip().startswith("rt_claude_"):
                await a._kill_tmux_session(line.strip())


# ──────────────── Basic ask() flow ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_consult_simple_arithmetic(adapter):
    """Simple Q&A: '2+2' → '4'. No FS access. Haiku = levný."""
    result = await adapter.ask(
        prompt="What is 2 + 2? Reply with just the number, nothing else.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=180.0,
    )
    assert result.ok is True, f"failed: {result}"
    assert "4" in result.text, f"expected '4' in: {result.text!r}"
    assert result.adapter == "tmux"
    assert result.session_id is not None


@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_edit_creates_file(adapter, tmp_path):
    """edit mode: Claude vytvoří soubor přes Write tool."""
    wd = tmp_path / "edit_test"
    wd.mkdir()
    target = wd / "hello.py"
    assert not target.exists()

    result = await adapter.ask(
        prompt=(
            "Create a file named hello.py in the current directory containing "
            "exactly: print('hello from tmux claude')\n"
            "Use the Write tool. Then respond with just 'DONE'."
        ),
        model="claude-haiku-4-5",
        mode="edit",
        workdir=wd,
        timeout_sec=180.0,
    )
    assert result.ok is True, f"failed: {result}"
    assert target.exists(), (
        f"hello.py NEVZNIKL v {wd}; obsah: {list(wd.iterdir())}"
    )
    content = target.read_text()
    assert "hello from tmux claude" in content


# ──────────────── Progress events emitted ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_progress_events_emitted(adapter, tmp_path):
    """Adapter MUSÍ emit alespoň 'started' event + tool_use events."""
    wd = tmp_path / "progress_test"
    wd.mkdir()
    (wd / "a.txt").write_text("hello")

    events: list[ProgressEvent] = []

    async def callback(ev: ProgressEvent) -> None:
        events.append(ev)

    result = await adapter.ask(
        prompt=(
            "Read a.txt and tell me its contents. Use Read tool. "
            "Then respond with the content."
        ),
        model="claude-haiku-4-5",
        mode="edit",
        workdir=wd,
        timeout_sec=180.0,
        progress_callback=callback,
    )
    assert result.ok is True, f"failed: {result}"
    stages = [e.stage for e in events]
    assert "started" in stages, f"no 'started' progress event: {stages}"
    # Aspoň jeden tool_use event (Read)
    assert "tool_use" in stages, f"no 'tool_use' event: {stages}"
    tool_use_events = [e for e in events if e.stage == "tool_use"]
    assert any(e.tool_name == "Read" for e in tool_use_events)


# ──────────────── Long-lived session: context continuity ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(360)
async def test_long_session_context_continuity(adapter):
    """3 turny v rámci jedné session - 3. má pamatovat 1. a 2.

    Klíčový test pro long-lived sessions. Bez tmux session re-use by
    Claude každé ask() ztratil context.
    """
    # Turn 1: nastavit context
    r1 = await adapter.ask(
        prompt="My favorite color is purple. Just acknowledge with 'OK'.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=120.0,
    )
    assert r1.ok is True
    sid = r1.session_id
    assert sid is not None

    # Turn 2: dodat víc context, použít stejnou session
    r2 = await adapter.ask(
        prompt="My favorite number is 42. Just acknowledge.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        session_id=sid,
        timeout_sec=120.0,
    )
    assert r2.ok is True
    assert r2.session_id == sid

    # Turn 3: zeptat se na color + number současně
    r3 = await adapter.ask(
        prompt="What is my favorite color and number?",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        session_id=sid,
        timeout_sec=120.0,
    )
    assert r3.ok is True
    # Claude MUSÍ vědět z předchozích turnů
    text_lower = r3.text.lower()
    assert "purple" in text_lower, f"purple not in: {r3.text!r}"
    assert "42" in r3.text, f"42 not in: {r3.text!r}"


# ──────────────── Permission mode immutable (codex iter-2 #8) ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_permission_mode_immutable_across_asks(adapter, tmp_path):
    """ask() s session_id ale jiný mode → AdapterConfigError.

    Per invariant: permission_mode is immutable per Claude process.
    """
    from claude_bridge.exceptions import AdapterConfigError

    # Spawn consult session
    r1 = await adapter.ask(
        prompt="Hi.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=90.0,
    )
    assert r1.ok is True
    sid = r1.session_id

    # Try to switch to edit mode on same session - MUSÍ raise
    with pytest.raises(AdapterConfigError, match="immutable"):
        await adapter.ask(
            prompt="Now create a file.",
            model="claude-haiku-4-5",
            mode="edit",  # different from consult!
            workdir=tmp_path,
            session_id=sid,
            timeout_sec=90.0,
        )


# ──────────────── /clear command ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_clear_wipes_context(adapter):
    """ask 'remember X', /clear, ask 'co bylo X' → Claude nepamatuje."""
    r1 = await adapter.ask(
        prompt="My secret token is ZEBRA42. Just acknowledge.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=120.0,
    )
    sid = r1.session_id
    assert r1.ok is True

    # Clear conversation history
    cleared = await adapter.clear_session(sid)
    assert cleared is True

    # Wait pro clear processing
    await asyncio.sleep(1.0)

    # Po clear: Claude nemá vědět token
    r2 = await adapter.ask(
        prompt="What was my secret token? Reply ONLY with 'NO_MEMORY' if you don't remember, otherwise the token.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        session_id=sid,
        timeout_sec=120.0,
    )
    assert r2.ok is True
    # Claude nesmí mít token v odpovědi
    assert "ZEBRA42" not in r2.text, (
        f"clear nezapůsobil - Claude pořád pamatuje: {r2.text!r}"
    )


# ──────────────── Dead session no recreate (codex iter-2 #13) ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_dead_session_no_silent_recreate(adapter):
    """Externí kill tmux session → next ask() raises SessionDead.

    Adapter NESMÍ silent respawn (kontext by se ztratil bez varování).
    """
    from claude_bridge.exceptions import SessionDead

    r1 = await adapter.ask(
        prompt="Hi.",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=90.0,
    )
    assert r1.ok is True
    sid = r1.session_id

    # Externí kill (simulace crashe / user killed tmux)
    await adapter._kill_tmux_session(sid)
    # Pozor: kill_session normalně sundá z _sessions, my chceme jen tmux kill
    # bez touching _sessions, abychom simulovali external. Re-register.
    from claude_bridge.adapters.tmux_mode import _TmuxSession
    from claude_bridge.parsing.tui_state import TuiState
    import time
    if sid not in adapter._sessions:
        adapter._sessions[sid] = _TmuxSession(
            session_id=sid, workdir=None,
            model="claude-haiku-4-5", permission_mode="consult",
            created_at=time.time(), last_active=time.time(),
            state="READY",  # State říká READY, ale tmux session je mrtvá
            tui=TuiState(cols=80, rows=24),
        )

    # Next ask() s tímto session_id → health_check detect DEAD → raise SessionDead
    with pytest.raises(SessionDead, match="DEAD"):
        await adapter.ask(
            prompt="Hi again.",
            model="claude-haiku-4-5",
            mode="consult",
            workdir=None,
            session_id=sid,
            timeout_sec=60.0,
        )
