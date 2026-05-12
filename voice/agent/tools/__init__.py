"""Built-in tool registry.

`default_registry()` vrátí ToolRegistry naplněnou všemi dostupnými tooly pro
zvolený `mode` (chat: jen skills; agent: vše)."""
from __future__ import annotations

from voice.agent.tools.base import Tool, ToolRegistry
from voice.agent.tools import echo as _echo_mod


def default_registry(mode: str = "agent") -> ToolRegistry:
    reg = ToolRegistry()
    # Phase 1: jen echo. Další fáze přidají fs/shell/web/home.
    reg.register(_echo_mod.TOOL)
    return reg


__all__ = ["Tool", "ToolRegistry", "default_registry"]
