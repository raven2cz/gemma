"""Factory + config pro adapter selection.

User/gemma config volí adapter via BridgeMode enum. Factory check dependencies
před vytvořením, raises AdapterConfigError s navodným error pokud nesplněno.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .exceptions import AdapterConfigError

if TYPE_CHECKING:
    from .base import AbstractClaudeAdapter


class BridgeMode(str, Enum):
    """Adapter selection enum."""
    PRINT = "print"   # `claude -p` per ask, ephemeral
    TMUX = "tmux"     # `claude` v tmux session, long-lived (experimental)


@dataclass
class AdapterConfig:
    """Config pro create_adapter() factory.

    Defaultně print mode (= safest, žádné experimental TOS gray area).
    Tmux mode user explicit opt-in (per codex iter-1 critical #1).
    """
    mode: BridgeMode = BridgeMode.PRINT
    claude_bin: str = "claude"
    tmux_bin: str = "tmux"

    # Per-call defaults (mohou být přepsané v ask() args)
    default_timeout_sec: float = 600.0
    default_model: str = "claude-opus-4-7"

    # Print mode specific
    # (žádný output cap - per user policy "úplně odstranit cap")

    # Tmux mode specific
    tmux_history_limit: int = 100_000  # scrollback per session
    tmux_session_prefix: str = "claude_"
    tmux_idle_kill_hours: int = 24     # auto-cleanup idle sessions (per #3)

    # Persistence (per codex iter-2 critical #2)
    metadata_dir: str = ""  # populated by caller (e.g. ".gemma_local/")

    # Env propagation - which env vars from parent to passthrough do CLI
    env_allowlist: tuple[str, ...] = field(default_factory=lambda: (
        "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TERM",
        "PATH", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
        "XDG_STATE_HOME", "XDG_RUNTIME_DIR",
        "ANTHROPIC_API_KEY",  # CAUTION: pokud set, claude CLI use API billing!
    ))


def create_adapter(config: AdapterConfig) -> "AbstractClaudeAdapter":
    """Factory s fail-fast dependency check (codex iter-2 high #12).

    Pro TMUX mode: ověř že (a) tmux binary v PATH, (b) `pyte` lze importovat.
    Jinak raise AdapterConfigError s navodnou hláškou (= server hláška na UI).

    Pro PRINT mode: žádná special validace, claude CLI check at spawn time.

    Returns:
        Konkrétní adapter instance implementující AbstractClaudeAdapter.

    Raises:
        AdapterConfigError: missing tmux binary, missing pyte, etc.
    """
    if config.mode == BridgeMode.TMUX:
        if shutil.which(config.tmux_bin) is None:
            raise AdapterConfigError(
                f"tmux binary {config.tmux_bin!r} not in PATH. "
                f"Install: apt install tmux / pacman -S tmux / brew install tmux. "
                f"Or switch to BridgeMode.PRINT."
            )
        try:
            import pyte  # noqa: F401
        except ImportError as e:
            raise AdapterConfigError(
                f"pyte library required for tmux adapter (parsing TUI output). "
                f"Install: pip install 'claude_bridge[tmux]'. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        from .adapters.tmux_mode import TmuxAdapter
        return TmuxAdapter(config)

    # Default: print mode
    from .adapters.print_mode import PrintModeAdapter
    return PrintModeAdapter(config)
