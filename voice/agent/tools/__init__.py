"""Built-in tool registry.

`default_registry()` vrátí ToolRegistry naplněnou všemi dostupnými tooly pro
zvolený `mode` (chat: jen skills; agent: vše)."""
from __future__ import annotations

from voice.agent.tools.base import Tool, ToolRegistry
from voice.agent.tools import claude as _claude_mod
from voice.agent.tools import echo as _echo_mod
from voice.agent.tools import fs as _fs_mod
from voice.agent.tools import hue as _hue_mod
from voice.agent.tools import shell as _shell_mod
from voice.agent.tools import web as _web_mod


def default_registry(mode: str = "agent") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_echo_mod.TOOL)
    # Phase 2: file-system tooly (read/list/glob/grep/write/edit).
    for tool in _fs_mod.ALL_TOOLS:
        reg.register(tool)
    # Phase 3: bash shell tool.
    reg.register(_shell_mod.TOOL)
    # Phase 4: web tooly (fetch_url, web_search).
    for tool in _web_mod.ALL_TOOLS:
        reg.register(tool)
    # Phase 5: Philips Hue smart-home tooly (light_list, light_set).
    for tool in _hue_mod.ALL_TOOLS:
        reg.register(tool)
    # Phase 6: Claude bridge (ask_claude).
    for tool in _claude_mod.ALL_TOOLS:
        reg.register(tool)
    return reg


__all__ = ["Tool", "ToolRegistry", "default_registry"]
