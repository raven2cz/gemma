"""claude_bridge - adapter library pro driving Claude Code CLI.

Dvě implementace:
  - PrintModeAdapter: spustí `claude -p` per ask() (= dnešní voice/agent default)
  - TmuxAdapter: drive `claude` v tmux session (experimentální, future)

Public API:
  - AbstractClaudeAdapter (Protocol)
  - ClaudeResult, ProgressEvent, SessionInfo (data models)
  - PrintModeAdapter (concrete)
  - create_adapter(config) factory
  - Exception hierarchie

Usage:
    from claude_bridge import create_adapter, AdapterConfig, BridgeMode

    config = AdapterConfig(mode=BridgeMode.PRINT)
    adapter = create_adapter(config)
    result = await adapter.ask(
        prompt="What is 2+2?",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
    )
    print(result.text)  # "4"
"""
from __future__ import annotations

from .base import (
    AbstractClaudeAdapter,
    AdapterCapabilities,
    ProgressCallback,
    SessionState,
)
from .config import AdapterConfig, BridgeMode, create_adapter
from .exceptions import (
    AdapterConfigError,
    ClaudeBridgeError,
    SessionBusy,
    SessionDead,
    SessionNotFound,
    SubprocessError,
)
from .progress import ProgressEvent, ProgressStage, SessionInfo
from .result import ClaudeResult, Mode

__version__ = "0.1.0"

__all__ = [
    # Interface + capabilities
    "AbstractClaudeAdapter",
    "AdapterCapabilities",
    "SessionState",
    # Data models
    "ClaudeResult",
    "Mode",
    "ProgressEvent",
    "ProgressStage",
    "SessionInfo",
    "ProgressCallback",
    # Factory + config
    "AdapterConfig",
    "BridgeMode",
    "create_adapter",
    # Exceptions
    "ClaudeBridgeError",
    "AdapterConfigError",
    "SessionBusy",
    "SessionDead",
    "SessionNotFound",
    "SubprocessError",
]
