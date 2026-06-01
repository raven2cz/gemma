"""Model Context Protocol (MCP) klient nad subprocess + stdio JSON-RPC 2.0.

Spec: https://spec.modelcontextprotocol.io/specification/2024-11-05/

Design:
- Generic — používá ho i HOTOVO integrace, ale není na ní vázaný
- Lazy spawn: subprocess startne až při prvním tool callu, ne při registry
  build (registry probe je HTTP-only nebo skip)
- Idle timeout: po N minutách bez requestu subprocess umírá, na další call
  respawn (ušetří RAM když user MCP nepoužívá)
- Per-server lock: souběžné tool calls serializované, JSON-RPC stdio nemá
  multiplexing přes id (mohli bychom, ale serialization je jednodušší a
  pro UI-driven agentní tool calls naprosto stačí)
- Crash recovery: pokud subprocess vrátí non-zero nebo stdout EOF mid-request,
  poznamenáme failed, příští call respawn

Bezpečnost:
- env scrub: jen allowlist + per-server explicit env (auth tokens, PORT, …),
  žádný leak parent env do potenciálně 3rd-party MCP serveru
- argv list (shell=False), command z trusted config (ne user input)
- stdout line cap (8 MiB per line) — chrání před runaway adversarial server
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent-mcp")

# JSON-RPC stdio: 8 MiB per line cap (large get_state responses + safety)
_STDIO_LINE_LIMIT = 8 * 1024 * 1024

# Default env allowlist (passed to subprocess unless overriden per-server)
_DEFAULT_ENV_ALLOWLIST = (
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
    "PATH", "TMPDIR", "NODE_OPTIONS",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
)

# JSON-RPC error codes (per MCP spec subset)
_ERR_PARSE = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_INTERNAL = -32603


@dataclass(frozen=True)
class McpServerConfig:
    """Static config jednoho MCP serveru (loaded from voice.agent.config)."""
    name: str                            # interní jméno, prefix pro Tool name
    command: tuple[str, ...]             # argv list, shell=False
    env: dict[str, str] = field(default_factory=dict)  # extra env (+ allowlist parent)
    env_allowlist: tuple[str, ...] = _DEFAULT_ENV_ALLOWLIST
    cwd: str | None = None
    # Tool classifier hints:
    auto_tools: frozenset[str] = field(default_factory=frozenset)
    requires_explicit_tools: frozenset[str] = field(default_factory=frozenset)
    # Health probe (volitelné) — pokud nedostupný, registry skip tooly.
    # Aktuálně podporujeme jen HTTP probe ("http://host:port/path").
    health_probe_url: str | None = None
    health_probe_timeout_sec: float = 1.5
    # Idle timeout: po této době bez requestu subprocess umírá. None = vždy on.
    idle_timeout_sec: float | None = 300.0
    # Tool call timeout (per JSON-RPC request, ne celý subprocess).
    request_timeout_sec: float = 30.0


class McpError(Exception):
    """Generická MCP chyba (subprocess crash, JSON-RPC error, timeout)."""


class McpClient:
    """Stateful MCP client — jeden per server config.

    Lifecycle:
        client = McpClient(config)
        tools = await client.list_tools()  # spawn + initialize + tools/list
        result = await client.call_tool("get_state", {})
        await client.shutdown()
    """

    def __init__(self, config: McpServerConfig):
        self.config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()
        self._cached_tools: list[dict] | None = None
        self._last_activity = 0.0
        self._idle_task: asyncio.Task | None = None

    # ─── Public API ────────────────────────────────────────────────────────

    async def list_tools(self) -> list[dict]:
        """Vrátí seznam tool definic od MCP serveru (cached po prvním fetch).

        Each entry: {"name": str, "description": str, "inputSchema": dict}
        """
        if self._cached_tools is not None:
            return self._cached_tools
        async with self._lock:
            if self._cached_tools is not None:
                return self._cached_tools
            await self._ensure_started()
            result = await self._request("tools/list", {})
            tools = result.get("tools") or []
            if not isinstance(tools, list):
                raise McpError(f"tools/list returned non-list: {type(tools).__name__}")
            self._cached_tools = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        """Pošli tools/call s daným tool name + args, vrať raw MCP result.

        Result shape (per MCP spec):
            {"content": [{"type":"text","text":"..."}], "isError": false}
        Caller si extrahuje text obsah sám (každý tool/server může mít jiný
        shape — fields vrácené v `content[0].text` jsou typicky JSON).
        """
        async with self._lock:
            await self._ensure_started()
            return await self._request("tools/call", {
                "name": name, "arguments": arguments,
            })

    async def shutdown(self) -> None:
        """Zabij subprocess (graceful → SIGTERM → SIGKILL po 2s)."""
        async with self._lock:
            await self._shutdown_locked()

    # ─── Internal ──────────────────────────────────────────────────────────

    async def _ensure_started(self) -> None:
        """Spawnne subprocess + pošle `initialize` pokud ještě neběží."""
        if self._proc is not None and self._proc.returncode is None:
            self._touch_activity()
            return

        # Předchozí proces mrtvý nebo nikdy nespawnnutý — full restart.
        await self._shutdown_locked(silent=True)

        env = self._build_env()
        log.info("mcp spawn server=%s argv=%s", self.config.name, list(self.config.command))
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self.config.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd,
                start_new_session=True,
                limit=_STDIO_LINE_LIMIT,
            )
        except FileNotFoundError as e:
            raise McpError(
                f"MCP server '{self.config.name}' binary not found: {self.config.command[0]} ({e})"
            ) from e
        except Exception as e:
            raise McpError(
                f"MCP server '{self.config.name}' spawn failed: {type(e).__name__}: {e}"
            ) from e

        # Spustíme stdout reader task (parse JSON-RPC responses → futures)
        self._reader_task = asyncio.create_task(self._stdout_reader())
        # Drain stderr do log (ne kritické, jen warn)
        asyncio.create_task(self._stderr_drain())

        # Initialize handshake (MCP spec)
        try:
            init_result = await asyncio.wait_for(
                self._request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "gemma-agent", "version": "1.0.0"},
                }),
                timeout=self.config.request_timeout_sec,
            )
            log.info(
                "mcp initialized server=%s proto=%s server_info=%s",
                self.config.name,
                init_result.get("protocolVersion"),
                init_result.get("serverInfo"),
            )
        except Exception as e:
            await self._shutdown_locked(silent=True)
            raise McpError(
                f"MCP server '{self.config.name}' initialize failed: {type(e).__name__}: {e}"
            ) from e

        self._touch_activity()
        if self.config.idle_timeout_sec is not None and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_watchdog())

    def _build_env(self) -> dict[str, str]:
        """Scrubbed env: allowlist parent env + per-server explicit env."""
        env: dict[str, str] = {}
        for key in self.config.env_allowlist:
            v = os.environ.get(key)
            if v is not None:
                env[key] = v
        # LC_* prefix wildcard
        for k, v in os.environ.items():
            if k.startswith("LC_"):
                env[k] = v
        # Per-server explicit (auth tokens, PORT, …)
        env.update(self.config.env)
        return env

    async def _request(self, method: str, params: dict) -> dict:
        """Pošli JSON-RPC request, počkej na response. Caller drží self._lock."""
        if self._proc is None or self._proc.stdin is None:
            raise McpError("subprocess not running")
        req_id = self._next_id
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        line = json.dumps(request, ensure_ascii=False) + "\n"

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future

        try:
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            self._pending.pop(req_id, None)
            raise McpError(f"stdin write failed (subprocess crashed?): {e}") from e

        self._touch_activity()
        try:
            response = await asyncio.wait_for(future, timeout=self.config.request_timeout_sec)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise McpError(
                f"MCP {method!r} request timed out after {self.config.request_timeout_sec}s"
            )
        if "error" in response:
            err = response["error"]
            raise McpError(
                f"MCP {method!r} error [{err.get('code', '?')}]: {err.get('message', 'unknown')}"
            )
        return response.get("result") or {}

    async def _stdout_reader(self) -> None:
        """Read line-delimited JSON-RPC responses → match to pending futures."""
        assert self._proc is not None and self._proc.stdout is not None
        try:
            while True:
                try:
                    line = await self._proc.stdout.readline()
                except (asyncio.LimitOverrunError, ValueError) as e:
                    log.warning(
                        "mcp %s stdout line exceeded %d bytes, skip: %s",
                        self.config.name, _STDIO_LINE_LIMIT, e,
                    )
                    continue
                if not line:  # EOF
                    break
                try:
                    msg = json.loads(line.decode("utf-8", errors="replace").rstrip())
                except (ValueError, UnicodeDecodeError):
                    continue
                if not isinstance(msg, dict):
                    continue
                msg_id = msg.get("id")
                if msg_id is None:
                    # Notification (no id) — ignore for now
                    continue
                future = self._pending.pop(msg_id, None)
                if future is None or future.done():
                    continue
                future.set_result(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("mcp %s stdout_reader crashed: %s", self.config.name, e)
        finally:
            # Subprocess EOF → reject všechny pending futures
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(McpError("subprocess EOF before response"))
            self._pending.clear()

    async def _stderr_drain(self) -> None:
        """Drain stderr do log (warn level), prevent pipe fill."""
        assert self._proc is not None and self._proc.stderr is not None
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                txt = line.decode("utf-8", errors="replace").rstrip()
                if txt:
                    log.info("mcp %s stderr: %s", self.config.name, txt)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _touch_activity(self) -> None:
        self._last_activity = asyncio.get_event_loop().time()

    async def _idle_watchdog(self) -> None:
        """Po idle_timeout_sec ticha zabij subprocess."""
        timeout = self.config.idle_timeout_sec
        assert timeout is not None
        try:
            while True:
                await asyncio.sleep(timeout / 4)
                now = asyncio.get_event_loop().time()
                if self._proc is None or self._proc.returncode is not None:
                    break
                if now - self._last_activity > timeout:
                    log.info(
                        "mcp %s idle %.0fs > %.0fs, shutting down",
                        self.config.name, now - self._last_activity, timeout,
                    )
                    async with self._lock:
                        await self._shutdown_locked(silent=False)
                    break
        except asyncio.CancelledError:
            pass

    async def _shutdown_locked(self, *, silent: bool = False) -> None:
        """Caller drží self._lock."""
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
        if self._idle_task is not None and not self._idle_task.done():
            self._idle_task.cancel()
        self._reader_task = None
        self._idle_task = None

        proc = self._proc
        self._proc = None
        self._cached_tools = None
        self._next_id = 1
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(McpError("client shutdown"))
        self._pending.clear()

        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                if not silent:
                    log.warning("mcp %s subprocess survived SIGKILL", self.config.name)


async def health_probe(config: McpServerConfig) -> bool:
    """HTTP-based health probe — pokud config.health_probe_url je set, GET ho.

    Vrátí True když server odpověděl 2xx, jinak False. Bez probe URL → True
    (= unconditional registration; lazy spawn pozná chybu sám).
    """
    if not config.health_probe_url:
        return True
    try:
        import httpx
        async with httpx.AsyncClient(timeout=config.health_probe_timeout_sec) as c:
            resp = await c.get(config.health_probe_url)
            return 200 <= resp.status_code < 300
    except Exception as e:
        log.info("mcp %s health probe failed: %s", config.name, e)
        return False
