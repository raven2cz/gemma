"""MCP tool discovery + wrap → gemma Tool.

Při startu webapp se pro každý nakonfigurovaný MCP server (voice.agent.config
MCP_SERVERS):
1. Probe health (HTTP nebo skip pokud bez URL)
2. Pokud probe OK → spawn subprocess + initialize + tools/list
3. Pro každý discovered tool vyrobíme `Tool(name=<server>_<tool>, ...)`
4. Zaregistrujeme do default_registry
5. Subprocess pak žije s idle timeout (lazy re-spawn na další call)

Tool names jsou prefixované jménem serveru aby nedošlo ke kolizi (hotovo_get_state,
hotovo_create_task, …). Model dostává tool description z MCP serveru přímo
(autoritativní = jakkoli ho MCP server popíše).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from voice.agent.mcp import McpClient, McpError, McpServerConfig, health_probe
from voice.agent.tools.base import ExecuteContext, Tool

log = logging.getLogger("agent-mcp")


def _make_execute(client: McpClient, mcp_tool_name: str):
    """Vrátí execute callback pro daný MCP tool. Closes over client + tool name."""
    async def _execute(args: dict, ctx: ExecuteContext) -> dict:
        try:
            mcp_result = await client.call_tool(mcp_tool_name, args)
        except McpError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            log.exception("mcp tool call %s failed", mcp_tool_name)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

        # MCP result shape: {"content":[{"type":"text","text":"..."}], "isError":bool}
        # Extrahujeme text content + parsneme JSON pokud možno.
        is_error = bool(mcp_result.get("isError"))
        content = mcp_result.get("content") or []
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
        raw_text = "\n".join(text_parts)

        # Pokud text vypadá jako JSON, parsne; jinak vrať jako "text".
        payload: dict[str, Any] = {"ok": not is_error}
        if raw_text:
            try:
                parsed = json.loads(raw_text)
                payload["result"] = parsed
            except (ValueError, TypeError):
                payload["text"] = raw_text
        if is_error:
            payload["error"] = raw_text or "MCP tool returned isError without message"
        return payload

    return _execute


async def discover_and_register(
    config: McpServerConfig,
) -> tuple[McpClient | None, list[Tool]]:
    """Probe + discover + wrap. Vrátí (client, tools) nebo (None, []).

    None client = server unavailable nebo crashed během discovery. Tooly se
    nezaregistrujou. Po startup respawn nebude — user musí restartovat webapp
    nebo přidat retry policy (TODO později).
    """
    if not await health_probe(config):
        log.info("mcp %s health probe failed — skipping registration", config.name)
        return None, []

    client = McpClient(config)
    try:
        mcp_tools = await client.list_tools()
    except McpError as e:
        log.warning("mcp %s tools/list failed: %s", config.name, e)
        await client.shutdown()
        return None, []

    gemma_tools: list[Tool] = []
    for mcp_def in mcp_tools:
        mcp_name = mcp_def.get("name")
        if not isinstance(mcp_name, str) or not mcp_name:
            continue
        description = mcp_def.get("description") or f"MCP tool {mcp_name}"
        schema = mcp_def.get("inputSchema") or {"type": "object", "properties": {}}
        # Prefixujeme jménem serveru — chrání před kolizí (hotovo_get_state, …)
        gemma_name = f"{config.name}_{mcp_name}"
        gemma_tools.append(Tool(
            name=gemma_name,
            description=description,
            parameters_schema=schema,
            execute=_make_execute(client, mcp_name),
        ))
    log.info("mcp %s registered %d tools", config.name, len(gemma_tools))
    return client, gemma_tools


# Module-level registry: server name → McpClient (pro shutdown na exit).
_MCP_CLIENTS: dict[str, McpClient] = {}


def get_active_clients() -> dict[str, McpClient]:
    return dict(_MCP_CLIENTS)


def remember_client(name: str, client: McpClient) -> None:
    _MCP_CLIENTS[name] = client


async def shutdown_all() -> None:
    """Volá se z webapp lifespan shutdown."""
    for name, client in list(_MCP_CLIENTS.items()):
        try:
            await client.shutdown()
            log.info("mcp %s shut down", name)
        except Exception as e:
            log.warning("mcp %s shutdown error: %s", name, e)
    _MCP_CLIENTS.clear()
