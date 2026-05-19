"""Abstract interface pro Claude bridge adaptery.

Dva adaptery splňují tento interface:
  - PrintModeAdapter: spustí `claude -p` per ask() (ephemeral, žádná session)
  - TmuxAdapter:      drive `claude` v tmux session (long-lived, persistent)

Server/UI mohou s adapterem pracovat uniformly bez znalosti backend detailu.
Capability flags signalizují co adapter podporuje (cost tracking, persistent
session, /clear, ...).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal, Protocol

from .progress import ProgressEvent, SessionInfo
from .result import ClaudeResult, Mode

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
SessionState = Literal["READY", "RUNNING", "CANCELING", "DEAD"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """Co adapter podporuje. Server/UI checknou před voláním rozšířených metod."""
    supports_cost: bool             # ClaudeResult.total_cost_usd populated?
    supports_progress: bool         # progress_callback emit eventů?
    supports_persistent_context: bool  # long-lived session across ask() calls?
    supports_session_list: bool     # adapter umí listovat existing sessions?
    supports_clear: bool            # /clear bez restartu session procesu?
    requires_tty: bool              # vyžaduje pseudo-TTY (= tmux nebo pty)?

    @classmethod
    def print_mode(cls) -> "AdapterCapabilities":
        return cls(
            supports_cost=True,
            supports_progress=True,
            supports_persistent_context=False,
            supports_session_list=False,
            supports_clear=False,
            requires_tty=False,
        )

    @classmethod
    def tmux_mode(cls) -> "AdapterCapabilities":
        return cls(
            supports_cost=False,           # interactive claude neukazuje cost
            supports_progress=True,        # best-effort z TUI parsingu
            supports_persistent_context=True,
            supports_session_list=True,
            supports_clear=True,
            requires_tty=True,
        )


class AbstractClaudeAdapter(Protocol):
    """Shared interface pro print_mode i tmux adapter.

    Adapter může uvnitř držet long-lived state (tmux session), z venku
    vypadá vždy stejně. Optional metody (list_sessions, clear_session,
    kill_session, health_check) vrátí no-op / NotSupported pro adaptery
    kde to nedává smysl (= print mode).
    """
    name: str  # "print" | "tmux"
    capabilities: AdapterCapabilities

    async def ask(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str,
        mode: Mode,
        workdir: Path | None,
        session_id: str | None = None,
        timeout_sec: float = 600.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ClaudeResult:
        """Primární operace: pošli prompt, počkej na odpověď, vrať result.

        Args:
            prompt: User dotaz pro Claude.
            system: Optional system prompt (role/persona).
            model: claude-opus-4-7 / claude-sonnet-4-6 / claude-haiku-4-5.
            mode: "consult" (no FS access) | "edit" (workdir tools).
            workdir: Required pro mode="edit"; cwd pro Claude tools.
            session_id: Optional persistent session ID. None = adapter
                rozhodne (print: ephemeral; tmux: spawn novou nebo reuse).
            timeout_sec: Hard kill po této době bez completion.
            cancel_event: asyncio.Event - if set during run, adapter aborts.
            progress_callback: Async callback volaný s ProgressEvent během běhu.

        Returns:
            ClaudeResult s ok=True a text, nebo ok=False s error.

        Raises:
            AdapterConfigError: missing tmux/pyte/auth.
            SessionBusy: konkurentní ask() na running session.
            SessionDead: session je DEAD (jen pokud session_id specified).
        """
        ...

    # Session lifecycle (no-op pro print adapter; meaningful pro tmux)

    async def list_sessions(self) -> list[SessionInfo]:
        """Vrátí všechny adapter-managed sessions. Prázdný list pro print."""
        ...

    async def get_session(self, session_id: str) -> SessionInfo | None:
        """Lookup specific session by ID. None pokud neexistuje."""
        ...

    async def clear_session(self, session_id: str) -> bool:
        """`/clear` semantics - wipe history, keep session alive.
        Vrátí True pokud done; False pokud not supported nebo session DEAD."""
        ...

    async def kill_session(self, session_id: str) -> bool:
        """Permanent kill. Tmux: kill-session. Print: noop (return False).
        Vrátí True pokud killed; False pokud session nenalezena nebo not supported."""
        ...

    async def health_check(self, session_id: str) -> SessionState:
        """Detect dead/orphan sessions. Pro print vrátí vždy READY (žádná state).
        Pro tmux ověř tmux has-session + capture-pane responziveness."""
        ...

    async def close(self) -> None:
        """Adapter shutdown. Print: noop. Tmux: release sessions per config
        (default: NEKILLuje sessions - tmux server běží nezávisle, sessions
        survive gemma restart per #9 invariant)."""
        ...
