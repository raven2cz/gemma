"""Unit testy pro TmuxAdapter (bez real tmux/claude).

Testy které vyžadují live tmux jsou v `tests/integration/test_tmux_real.py`
s `@pytest.mark.tmux_real` markerem (skipnou bez tmux v PATH).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from claude_bridge.config import AdapterConfig, BridgeMode
from claude_bridge.exceptions import AdapterConfigError, SessionNotFound

# Skip celý modul pokud tmux není - zachová unit/integration separaci
# (tyto testy konstrukuju adapter, kde Constructor checkuje tmux v PATH).
_HAS_TMUX = shutil.which("tmux") is not None


# ──────────────── Constructor / fail-fast (codex iter-2 #12) ────────────────

def test_factory_raises_without_tmux(monkeypatch):
    """create_adapter(TMUX) bez tmux → AdapterConfigError s navodným message."""
    from claude_bridge.config import create_adapter

    def _no_which(name):
        return None  # simulate tmux missing
    monkeypatch.setattr(shutil, "which", _no_which)

    config = AdapterConfig(mode=BridgeMode.TMUX)
    with pytest.raises(AdapterConfigError) as exc_info:
        create_adapter(config)
    assert "tmux" in str(exc_info.value).lower()
    assert "PATH" in str(exc_info.value)


def test_factory_raises_without_pyte(monkeypatch):
    """create_adapter(TMUX) bez pyte → AdapterConfigError s installačním message."""
    from claude_bridge.config import create_adapter

    # Simulate tmux present but pyte missing
    monkeypatch.setattr(shutil, "which", lambda n: "/usr/bin/tmux")

    import builtins
    orig_import = builtins.__import__

    def _no_pyte(name, *args, **kwargs):
        if name == "pyte":
            raise ImportError("simulated missing pyte")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pyte)

    config = AdapterConfig(mode=BridgeMode.TMUX)
    with pytest.raises(AdapterConfigError) as exc_info:
        create_adapter(config)
    assert "pyte" in str(exc_info.value).lower()
    assert "install" in str(exc_info.value).lower()


# Skipnout zbytek pokud tmux není (adapter konstruktor by failnul)
pytestmark = pytest.mark.skipif(not _HAS_TMUX, reason="tmux not in PATH")


# ──────────────── Adapter properties ────────────────

def test_adapter_capabilities():
    """Tmux adapter má specific capability flags."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    cap = adapter.capabilities
    assert cap.requires_tty is True
    assert cap.supports_persistent_context is True
    assert cap.supports_session_list is True
    assert cap.supports_clear is True
    assert cap.supports_cost is False   # interactive neukazuje cost
    assert cap.supports_progress is True


def test_adapter_name():
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    assert adapter.name == "tmux"


def test_session_id_format():
    """Generated session_id má prefix + hex."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX, tmux_session_prefix="claude_")
    adapter = TmuxAdapter(config)
    sid = adapter._generate_session_id()
    assert sid.startswith("claude_")
    assert len(sid) == len("claude_") + 16  # 8 bytes = 16 hex chars


# ──────────────── argv construction ────────────────

def test_build_claude_argv_consult():
    """consult mode: no `-p`, --permission-mode plan, --tools empty."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    argv = adapter._build_claude_argv(
        model="claude-haiku-4-5", mode="consult",
        workdir=None, system=None,
    )
    assert argv[0] == "claude"
    assert "-p" not in argv
    assert "--permission-mode" in argv
    assert "plan" in argv
    assert "--tools" in argv


def test_build_claude_argv_edit():
    """edit mode: --permission-mode acceptEdits, --add-dir."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    workdir = Path("/tmp/test_workdir")
    argv = adapter._build_claude_argv(
        model="claude-opus-4-7", mode="edit",
        workdir=workdir, system=None,
    )
    assert "acceptEdits" in argv
    assert "--add-dir" in argv
    assert str(workdir) in argv
    # Tools allowlist
    assert "Read,Edit,Write,Bash,Glob,Grep" in argv


def test_build_claude_argv_edit_requires_workdir():
    """edit mode bez workdir → ValueError."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    with pytest.raises(ValueError, match="workdir"):
        adapter._build_claude_argv(
            model="claude-opus-4-7", mode="edit",
            workdir=None, system=None,
        )


def test_build_claude_argv_with_system_prompt():
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    argv = adapter._build_claude_argv(
        model="claude-haiku-4-5", mode="consult",
        workdir=None, system="You are an expert.",
    )
    assert "--append-system-prompt" in argv
    assert "You are an expert." in argv


# ──────────────── ask() bez tmux running ────────────────

@pytest.mark.asyncio
async def test_ask_empty_prompt():
    """Empty prompt → ok=False, žádný subprocess."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    result = await adapter.ask(
        prompt="", model="claude-haiku-4-5", mode="consult", workdir=None,
    )
    assert result.ok is False
    assert "empty" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_ask_nonexistent_session():
    """ask() s session_id co neexistuje → SessionNotFound."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    with pytest.raises(SessionNotFound):
        await adapter.ask(
            prompt="hi", model="claude-haiku-4-5", mode="consult",
            workdir=None, session_id="claude_nonexistent",
        )


# ──────────────── Session management ────────────────

@pytest.mark.asyncio
async def test_list_sessions_empty():
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    sessions = await adapter.list_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_get_session_none():
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    info = await adapter.get_session("nonexistent")
    assert info is None


@pytest.mark.asyncio
async def test_health_check_unknown_session():
    """Unknown session_id → DEAD."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)
    state = await adapter.health_check("unknown_id")
    assert state == "DEAD"


# ──────────────── Send-keys newline sanitization (sonnet H2) ────────────────

@pytest.mark.asyncio
async def test_send_keys_strips_newlines_in_literal_mode(monkeypatch):
    """Literal text s \\n / \\r\\n musí být sanitizovaný (= replaced spaces),
    jinak by tmux interpretoval newline jako Enter (= predčasný submit)."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)

    captured_args: list[tuple[str, ...]] = []

    async def fake_tmux(*args, **kwargs):
        captured_args.append(args)
        return 0, b"", b""

    monkeypatch.setattr(adapter, "_tmux", fake_tmux)

    await adapter._send_keys("sid", "line1\nline2", literal=True)
    # Args: ("send-keys", "-t", "sid", "-l", "line1 line2")
    assert captured_args[0][-1] == "line1 line2"

    captured_args.clear()
    await adapter._send_keys("sid", "a\r\nb\r\nc", literal=True)
    assert captured_args[0][-1] == "a b c"


@pytest.mark.asyncio
async def test_send_keys_preserves_newlines_in_non_literal_mode(monkeypatch):
    """Non-literal mode (key names) - žádná sanitace (caller posílá `Enter`)."""
    from claude_bridge.adapters.tmux_mode import TmuxAdapter
    config = AdapterConfig(mode=BridgeMode.TMUX)
    adapter = TmuxAdapter(config)

    captured_args: list[tuple[str, ...]] = []

    async def fake_tmux(*args, **kwargs):
        captured_args.append(args)
        return 0, b"", b""

    monkeypatch.setattr(adapter, "_tmux", fake_tmux)

    # literal=False = key name mode, žádný `-l` flag, žádná sanitace
    await adapter._send_keys("sid", "Enter", literal=False)
    assert "Enter" in captured_args[0]
    assert "-l" not in captured_args[0]
