"""Built-in tool registry.

`default_registry()` vrátí ToolRegistry naplněnou všemi dostupnými tooly pro
zvolený `mode` (chat: jen skills; agent: vše).

POZN (Fáze 6): `ask_claude` tool ODSTRANĚN z registry. Claude přístupný JEN
přes mode=claude (server endpoint /api/turn s mode="claude"). Clean separation:
- agent mode = Gemma + lokální nástroje (fs, shell, web, hue)
- claude mode = direct dialog s Claude přes claude_bridge adapter
Aktivační fráze "použij Opus" v agent módu se nyní triggeruje mode switch
na claude (frontend voice intent detection v app.js).
"""
from __future__ import annotations

from voice.agent.tools.base import Tool, ToolRegistry
from voice.agent.tools import echo as _echo_mod
from voice.agent.tools import fs as _fs_mod
from voice.agent.tools import hue as _hue_mod
from voice.agent.tools import shell as _shell_mod
from voice.agent.tools import web as _web_mod
from voice.agent.tools import hotovo as _hotovo_mod


# MCP discovered tools — populated z webapp lifespan startupu (async),
# čtené sync z default_registry(). Pokud žádný MCP server neběží, list je prázdný.
_MCP_DISCOVERED_TOOLS: list[Tool] = []


def set_mcp_tools(tools: list[Tool]) -> None:
    """Setter pro async MCP discovery. Webapp lifespan startup volá po
    discover_and_register(); subsequent default_registry() calls vidí tooly."""
    global _MCP_DISCOVERED_TOOLS
    _MCP_DISCOVERED_TOOLS = list(tools)


def _build_hotovo_tools() -> list[Tool]:
    """HOTOVO REST tooly. Registrují se JEN když HOTOVO_API_URL je nastaveno,
    jinak prázdný list. Lazy providers čtou config při každém execute → umožňuje
    runtime token reload (např. po `chmod 600 ~/.hotovo-api`)."""
    from voice.agent import config as _cfg

    if not _cfg.HOTOVO_API_URL:
        return []

    def _url(): return _cfg.HOTOVO_API_URL
    def _token(): return _cfg.get_hotovo_token()
    def _timeout(): return _cfg.HOTOVO_HTTP_TIMEOUT_SEC

    tools = _hotovo_mod.build_tools(
        base_url_provider=_url,
        token_provider=_token,
        timeout_provider=_timeout,
    )
    # complete_task má speciální body shape — apply post-build patch
    for i, t in enumerate(tools):
        if t.name == "hotovo_complete_task":
            tools[i] = _hotovo_mod.fix_complete_task_execute(
                t,
                base_url_provider=_url,
                token_provider=_token,
                timeout_provider=_timeout,
            )
            break
    return tools


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
    # HOTOVO todo-list REST tooly (pokud HOTOVO_API_URL je nastaveno).
    for tool in _build_hotovo_tools():
        reg.register(tool)
    # MCP discovered tools — async-discovered při startu webapp pro LOKÁLNÍ
    # MCP servery (HOTOVO není MCP, je REST).
    for tool in _MCP_DISCOVERED_TOOLS:
        reg.register(tool)
    return reg


__all__ = ["Tool", "ToolRegistry", "default_registry", "set_mcp_tools"]
