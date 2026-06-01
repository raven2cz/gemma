"""HOTOVO todo-list integrace přes přímé REST API volání.

Mapuje 8 toolů (get_state, list_projects, create_project, list_tasks,
create_task, update_task, complete_task, delete_task) na HTTP endpointy
HOTOVO serveru (https://github.com/raven2cz/todo-list).

Důvod proč ne MCP: HOTOVO `server/mcp.js` má hardcoded `BASE = http://127.0.0.1:PORT`,
takže nepodporuje remote backend (typický deployment: Raspberry Pi přes
nginx reverse proxy). Direct REST je 1:1 s tím, co MCP server stejně dělá —
jen vynecháme node subprocess.

Konfigurace:
    HOTOVO_API_URL          — base URL serveru (např. https://fishlive.org:17854)
    HOTOVO_API_TOKEN        — Bearer token (vytvořený v UI: Nastavení → AI Agenti)
    HOTOVO_API_TOKEN_FILE   — alternativa k env var, soubor 0600 (~/.hotovo-api)
    HOTOVO_HTTP_TIMEOUT_SEC — request timeout (default 10s)

Bezpečnost:
- HTTPS s TLS verifikací (httpx default)
- World/group-readable token file → odmítnout + warn (jako BRAVE_SEARCH_API_KEY)
- Per-tool timeout (cancel_event respect)
- Failure modes: server 5xx / network error → return ok=False s text msg
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from voice.agent.tools.base import ExecuteContext, Tool

log = logging.getLogger("agent-hotovo")

# Loopback hosty kde tolerujeme http:// (token neopustí stroj).
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

# ──────────────── Helpers ────────────────


def _validate_base_url(base_url: str) -> str | None:
    """Vrátí chybovou hlášku pokud base_url není bezpečné, jinak None.

    Bearer token NESMÍ jít přes plaintext http:// na remote host (codex review).
    Povolujeme https:// kamkoli + http:// jen na loopback.
    """
    try:
        u = urlparse(base_url)
    except Exception:
        return f"neplatné HOTOVO_API_URL: {base_url!r}"
    if u.scheme == "https":
        return None
    if u.scheme == "http":
        host = (u.hostname or "").lower()
        if host in _LOOPBACK_HOSTS:
            return None
        return (
            f"HOTOVO_API_URL používá http:// na remote host {host!r} — Bearer "
            "token by šel v plaintextu. Použij https:// (nebo http jen na localhost)."
        )
    return f"HOTOVO_API_URL musí být http(s)://, dostal {u.scheme!r}"


def _headers(token: str) -> dict[str, str]:
    h = {"Accept": "application/json", "Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _drop_empty(params: dict) -> dict:
    """Vyhodí None / prázdné values — REST API přijímá jen non-empty filtry."""
    return {k: v for k, v in params.items() if v not in (None, "")}


async def _request(
    method: str,
    base_url: str,
    path: str,
    token: str,
    timeout: float,
    *,
    params: dict | None = None,
    body: dict | None = None,
    cancel_event: asyncio.Event | None = None,
) -> dict:
    """Sjednocený REST call → vrací gemma-friendly dict {ok, result|error}."""
    url = f"{base_url.rstrip('/')}{path}"
    # POZN: cancel_event NEPOUŽÍVÁME přes .wait() — turn_state cancel_event je
    # threading.Event (ne asyncio.Event), takže asyncio.create_task(ev.wait())
    # by sync-blocking voláním zmrazilo celý event loop (stejný bug jako měl
    # claude bridge). Cancelaci řeší volající loop přes asyncio.wait_for kolem
    # tool.execute() + httpx vlastní `timeout`. Pre-flight is_set() check stačí.
    if cancel_event is not None and cancel_event.is_set():
        return {"ok": False, "error": "canceled"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(
                method, url,
                headers=_headers(token),
                params=_drop_empty(params or {}),
                json=body,
            )
    except httpx.TimeoutException:
        return {"ok": False, "error": f"HOTOVO timeout po {timeout}s ({method} {path})"}
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"HOTOVO HTTP chyba: {type(e).__name__}: {e}"}
    except Exception as e:
        log.exception("hotovo request failed")
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Parsování body — JSON nebo plain text
    raw_text = resp.text
    parsed: Any = None
    if raw_text:
        try:
            parsed = json.loads(raw_text)
        except (ValueError, TypeError):
            parsed = raw_text

    if 200 <= resp.status_code < 300:
        return {"ok": True, "result": parsed}
    # Server vrací {"error": "..."} JSON — vyextrahuj message
    err_msg = None
    if isinstance(parsed, dict):
        err_msg = parsed.get("error") or parsed.get("message")
    if not err_msg:
        err_msg = f"HTTP {resp.status_code}"
    return {"ok": False, "error": str(err_msg), "status": resp.status_code}


# ──────────────── Tool execute factory ────────────────


def _make_tool(
    *,
    gemma_name: str,
    description: str,
    parameters_schema: dict,
    method: str,
    path_template: str,                  # např. "/api/tasks/{id}"
    body_keys: tuple[str, ...] = (),     # které args jdou do body (POST/PUT)
    query_keys: tuple[str, ...] = (),    # které do query string (GET filter, ?confirm)
    nullable_body_keys: tuple[str, ...] = (),  # keys kde explicit None SE POSÍLÁ
    static_body: dict | None = None,     # vždy přidané do body (např. status:completed)
    base_url_provider,                   # callable → str (lazy, lze monkeypatch)
    token_provider,                      # callable → str
    timeout_provider,                    # callable → float
) -> Tool:
    """Vyrobí Tool s execute callbackem, který sestaví REST request z args.

    `nullable_body_keys`: pro update_task — `due_date: null` (smaž termín) a
    `parent_id: null` (odpoj rodiče) jsou sémanticky platné, takže explicit
    None se MUSÍ poslat (ne dropnout jako u create_task). Codex review.
    """
    async def _execute(args: dict, ctx: ExecuteContext) -> dict:
        base_url = base_url_provider()
        token = token_provider()
        timeout = timeout_provider()
        if not base_url:
            return {"ok": False, "error": "HOTOVO_API_URL není nastavený (export HOTOVO_API_URL=...)"}
        url_err = _validate_base_url(base_url)
        if url_err:
            return {"ok": False, "error": url_err}

        # Path s {id}, {list_id} substitucí
        path = path_template
        for ph in ("id", "list_id", "parent_id"):
            token_marker = "{" + ph + "}"
            if token_marker in path:
                val = args.get(ph)
                if not isinstance(val, str) or not val:
                    return {"ok": False, "error": f"chybí povinný parametr '{ph}'"}
                # URL-encode (chrání před path traversal)
                from urllib.parse import quote
                path = path.replace(token_marker, quote(val, safe=""))

        # Body: nenull keys jen pokud non-None; nullable keys i s explicit None.
        body: dict = {}
        for k in body_keys:
            if k not in args:
                continue
            if args[k] is None and k not in nullable_body_keys:
                continue  # drop implicit/empty None (create_task default behavior)
            body[k] = args[k]
        if static_body:
            body.update(static_body)
        body = body or None

        params = {k: args[k] for k in query_keys if k in args and args[k] is not None}

        return await _request(
            method, base_url, path, token, timeout,
            params=params, body=body,
            cancel_event=ctx.cancel_event,
        )

    return Tool(
        name=gemma_name,
        description=description,
        parameters_schema=parameters_schema,
        execute=_execute,
    )


# ──────────────── Tool definitions ────────────────


def build_tools(
    *, base_url_provider, token_provider, timeout_provider,
) -> list[Tool]:
    """Sestaví 8 HOTOVO toolů. Providers jsou lazy (čte se per call) — umožňuje
    runtime override (testy, settings reload).
    """
    return [
        _make_tool(
            gemma_name="hotovo_get_state",
            description=(
                "Vrátí snapshot HOTOVO: všechny projekty (lists) + úkoly s parent_id "
                "+ počty. Zavolej PRVNÍ — model dostane ID projektů potřebné pro "
                "create_task/update_task."
            ),
            parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
            method="GET", path_template="/api/agent/state",
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_list_projects",
            description="Vypíše HOTOVO projekty (lists). Každý má id, name, color.",
            parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
            method="GET", path_template="/api/lists",
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_create_project",
            description=(
                "Vytvoří nový HOTOVO projekt. Povinné: name (string). "
                "Volitelné: color (hex string, např. #4ade80)."
            ),
            parameters_schema={
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "description": "Název projektu."},
                    "color": {"type": "string", "description": "Hex barva, např. #4ade80."},
                },
                "additionalProperties": False,
            },
            method="POST", path_template="/api/lists",
            body_keys=("name", "color"),
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_list_tasks",
            description=(
                "Vypíše úkoly. Volitelné filtry (vše string): list_id (UUID projektu), "
                "status (pending|completed), priority (low|medium|high|urgent), "
                "due_date (YYYY-MM-DD), search (fulltext), tag, "
                "due (today|week|overdue)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "completed"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "due_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "search": {"type": "string"},
                    "tag": {"type": "string"},
                    "due": {"type": "string", "enum": ["today", "week", "overdue"]},
                },
                "additionalProperties": False,
            },
            method="GET", path_template="/api/tasks",
            query_keys=("list_id", "status", "priority", "due_date", "search", "tag", "due"),
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_create_task",
            description=(
                "Vytvoří nový úkol nebo podúkol. Povinné: title, list_id "
                "(získej z get_state). Volitelné: parent_id (musí být úkol ve "
                "STEJNÉM projektu pro podúkol), description, priority "
                "(low|medium|high|urgent), due_date (YYYY-MM-DD nebo ISO 8601 s TZ; "
                "úkol s due_date se automaticky propíše do Google Kalendáře), "
                "recurrence (daily|weekly|monthly|none), tags (pole stringů)."
            ),
            parameters_schema={
                "type": "object",
                "required": ["title", "list_id"],
                "properties": {
                    "title": {"type": "string"},
                    "list_id": {"type": "string"},
                    "parent_id": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "due_date": {"type": "string"},
                    "recurrence": {"type": "string", "enum": ["daily", "weekly", "monthly", "none"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
            method="POST", path_template="/api/tasks",
            body_keys=("title", "list_id", "parent_id", "description", "priority",
                       "due_date", "recurrence", "tags"),
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_update_task",
            description=(
                "Upraví existující úkol. Povinné: id. Volitelně cokoliv z: "
                "title, description, status (pending|completed), priority, "
                "due_date (null = vymazat termín), list_id (přesun do jiného "
                "projektu), parent_id (null = odpojit od rodiče)."
            ),
            parameters_schema={
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "completed"]},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "due_date": {"type": ["string", "null"]},
                    "list_id": {"type": "string"},
                    "parent_id": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            method="PUT", path_template="/api/tasks/{id}",
            body_keys=("title", "description", "status", "priority",
                       "due_date", "list_id", "parent_id"),
            # due_date:null = smaž termín, parent_id:null = odpoj rodiče — explicit
            # None SE POSÍLÁ (codex review HIGH #2).
            nullable_body_keys=("due_date", "parent_id"),
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_complete_task",
            description=(
                "Označí úkol jako splněný (status=completed). Dokončení rodiče "
                "dokončí i podúkoly; dokončení všech podúkolů dokončí rodiče. "
                "Opakovaný úkol se sám posune na další termín. Povinné: id."
            ),
            parameters_schema={
                "type": "object",
                "required": ["id"],
                "properties": {"id": {"type": "string"}},
                "additionalProperties": False,
            },
            method="PUT", path_template="/api/tasks/{id}",
            # static_body zajistí {"status":"completed"} vždy — žádný externí
            # patch (codex review HIGH #3: build_tools() je teď self-contained).
            static_body={"status": "completed"},
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
        _make_tool(
            gemma_name="hotovo_delete_task",
            description=(
                "Smaže úkol. Má-li podúkoly, předej confirm=true (jinak vrátí "
                "chybu). DESTRUKTIVNÍ — gemma vyžaduje frázi 'ano povoluju'."
            ),
            parameters_schema={
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            method="DELETE", path_template="/api/tasks/{id}",
            query_keys=("confirm",),
            base_url_provider=base_url_provider,
            token_provider=token_provider,
            timeout_provider=timeout_provider,
        ),
    ]
