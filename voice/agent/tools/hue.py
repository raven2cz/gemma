"""Philips Hue smart-home tooly — `light_list` + `light_set`.

Volá přímo Hue Bridge API v2 přes HTTPS (TLS self-signed cert, `verify=False`).
Bridge IP + application key jsou načteny z `~/.openhue/config.yaml` v `config.py`.

Bezpečnostní model:
- Bridge IP je load-time fixní (z config souboru), LLM ho nemůže ovlivnit
  (= no SSRF přes podstrčený host). Stejně tak app key.
- Endpoint paths jsou hard-coded; LLM kontroluje jen `name` (resolved na resource
  ID) a omezené tělo update (on/brightness/color). Žádná raw URL injection.
- Resource ID lookup je name→ID přes server response (in-memory cache na proces).
- TLS: `verify=False` protože Hue Bridge má self-signed cert. Klíč je sdílený
  secret v hlavičce, žádné citlivé creds přes net. (Hue ekosystém je local-only).
- Klíč nikdy nelogujeme ani neukazujeme ve výstupu.

Tooly:
- `light_list` — vrátí seznam světel s id, name, on/off, brightness, room.
- `light_set` — zapne/vypne, nastaví brightness (0-100 %) a/nebo color_name.
"""
from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any

import httpx

from voice.agent.config import (
    HUE_APP_KEY,
    HUE_BRIDGE_IP,
    HUE_OUTPUT_CAP_BYTES,
    HUE_TIMEOUT_SEC,
)
from voice.agent.tools.base import ExecuteContext, Tool


# ----------------------------------------------------------------------
# Color palette — name → xy CIE 1931 chromaticity coords.
# Schválně malá fixed sada, ne free-form RGB (LLM by mohl strobovat).
# ----------------------------------------------------------------------
COLOR_PALETTE: dict[str, tuple[float, float]] = {
    # Teplé / domácí
    "warm": (0.5054, 0.4153),
    "white": (0.3127, 0.3290),
    "cool": (0.3128, 0.3290),
    "daylight": (0.3168, 0.3253),
    # Barvy
    "red": (0.6750, 0.3220),
    "orange": (0.5751, 0.3957),
    "yellow": (0.4385, 0.4825),
    "green": (0.3000, 0.6000),
    "cyan": (0.2070, 0.3030),
    "blue": (0.1532, 0.0475),
    "purple": (0.2755, 0.1191),
    "pink": (0.4404, 0.1980),
    "magenta": (0.3617, 0.1531),
}


def _hue_url(path: str) -> str:
    """Build absolute Hue Bridge URL. `path` musí být čistý relativní suffix
    (např. `light` nebo `light/abc-123`). Fail closed pro `..`, `://`, leading
    `/`, query/fragment a control chars — všechny by mohly přepsat host nebo
    eskapovat z /clip/v2/resource/ pokud by `path` někdy začalo téct z bridge
    response (defense-in-depth, i když dnes je `path` plně callee-controlled).
    """
    if not path or not isinstance(path, str):
        raise ValueError("path must be a non-empty string")
    if "://" in path or ".." in path or path.startswith("/") or path.startswith("\\"):
        raise ValueError(f"unsafe hue path: {path!r}")
    # Reject query string / fragment / control chars / spaces.
    if any(c in path for c in ("?", "#", " ", "\t", "\n", "\r")):
        raise ValueError(f"unsafe hue path: {path!r}")
    return f"https://{HUE_BRIDGE_IP}/clip/v2/resource/{path}"


def _hue_headers() -> dict[str, str]:
    # Accept-Encoding: identity — vypne gzip/deflate/zstd negotiation.
    # Bez toho by httpx auto-dekomprimoval response a decompression-bomb
    # (malý Content-Length, obří expanded payload) by obešel HUE_OUTPUT_CAP_BYTES.
    return {
        "hue-application-key": HUE_APP_KEY,
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }


async def _hue_client() -> httpx.AsyncClient:
    """httpx client targeted at Hue Bridge.

    - verify=False (self-signed cert, local network)
    - trust_env=False (no HTTPS_PROXY hijack)
    - timeout krátký
    - žádný redirect (Hue Bridge nedělá 30x)
    """
    return httpx.AsyncClient(
        verify=False,
        trust_env=False,
        timeout=httpx.Timeout(HUE_TIMEOUT_SEC, connect=4.0),
        follow_redirects=False,
        headers=_hue_headers(),
    )


def _resp_size_ok(body: bytes) -> bool:
    return len(body) <= HUE_OUTPUT_CAP_BYTES


async def _stream_request_capped(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    json_body: Any = None,
) -> tuple[int, bytes, str]:
    """Streamovaný request (GET/PUT/POST) s hard size cap. Vrací
    (status_code, body, error). error="" pokud OK.

    Bezpečnostní guard:
    - Content-Length pre-check > cap → reject brzy (cheap).
    - Reject jakékoli ne-identity Content-Encoding (decompression bomb defense
      — server mohl vrátit malý gzipped payload který expanduje ven cap).
    - Bytes counter v aiter_raw() cyklu → break early při překročení capu.

    httpx auto-dekomprese se aktivuje jen pokud Content-Encoding header
    deklaruje non-identity codec. Tím že odmítáme jakýkoliv ne-identity CE
    *před* čtením těla, `aiter_bytes` nikdy nedostane do rukou kompresor,
    takže se stejně chová jako raw stream.
    """
    try:
        kwargs: dict[str, Any] = {}
        if json_body is not None:
            kwargs["json"] = json_body
        async with client.stream(method, url, **kwargs) as resp:
            # Reject Content-Encoding != identity (server ignored our request).
            ce = (resp.headers.get("content-encoding") or "").strip().lower()
            if ce and ce != "identity":
                return resp.status_code, b"", f"unexpected content-encoding {ce!r}"
            # Pre-check Content-Length if present.
            cl = resp.headers.get("content-length")
            if cl is not None:
                try:
                    cl_int = int(cl)
                except ValueError:
                    cl_int = -1
                if cl_int > HUE_OUTPUT_CAP_BYTES:
                    return resp.status_code, b"", "Hue response too large"
            chunks: list[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > HUE_OUTPUT_CAP_BYTES:
                    return resp.status_code, b"", "Hue response too large"
                chunks.append(chunk)
            return resp.status_code, b"".join(chunks), ""
    except httpx.TimeoutException as e:
        return 0, b"", f"timeout: {e}"
    except httpx.RequestError:
        return 0, b"", "hue request error"


async def _stream_get_capped(client: httpx.AsyncClient, url: str) -> tuple[int, bytes, str]:
    """Backwards-compatible thin wrapper around `_stream_request_capped`."""
    return await _stream_request_capped(client, "GET", url)


def _config_ok() -> tuple[bool, str]:
    if not HUE_BRIDGE_IP or not HUE_APP_KEY:
        return False, (
            "Hue Bridge not configured. Run `openhue setup` to create "
            "`~/.openhue/config.yaml` with `bridge` and `key` fields."
        )
    # Length sanity — Hue v2 application keys jsou 40 chars.
    if len(HUE_APP_KEY) < 20:
        return False, "Hue application key looks invalid (too short)"
    return True, ""


# ----------------------------------------------------------------------
# light_list
# ----------------------------------------------------------------------


def _summarize_light(d: dict) -> dict:
    """Extrahuj user-facing pole z full Hue API light resource."""
    metadata = d.get("metadata") or {}
    on = (d.get("on") or {}).get("on")
    dim = d.get("dimming") or {}
    color = d.get("color") or {}
    xy = (color.get("xy") or {}) if isinstance(color, dict) else {}
    return {
        "id": d.get("id", ""),
        "name": metadata.get("name", ""),
        "archetype": metadata.get("archetype", ""),
        "on": on if isinstance(on, bool) else None,
        "brightness": dim.get("brightness"),
        "color_xy": [xy.get("x"), xy.get("y")] if xy else None,
    }


async def _light_list_exec(args: dict, ctx: ExecuteContext) -> dict:
    """GET /clip/v2/resource/light — vrátí seznam light resourceů."""
    ok, why = _config_ok()
    if not ok:
        return {"ok": False, "error": why}

    start = time.monotonic()
    try:
        async with await _hue_client() as client:
            status, body, err = await _stream_get_capped(client, _hue_url("light"))
    except ValueError as e:
        # _hue_url path sanitization rejected — should be unreachable here.
        return {"ok": False, "error": f"invalid hue path: {e}"}

    if err:
        return {"ok": False, "error": err}

    if status == 401 or status == 403:
        return {"ok": False, "error": "Hue auth error (key invalid?)"}
    if status >= 400:
        return {"ok": False, "error": f"Hue HTTP {status}"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return {"ok": False, "error": f"invalid JSON from Hue: {e}"}

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {"ok": False, "error": "Hue response missing data list"}

    lights = [_summarize_light(item) for item in data if isinstance(item, dict)]
    return {
        "ok": True,
        "count": len(lights),
        "lights": lights,
        "duration_ms": int((time.monotonic() - start) * 1000),
    }


# ----------------------------------------------------------------------
# light_set
# ----------------------------------------------------------------------


async def _resolve_light_id(client: httpx.AsyncClient, name: str) -> tuple[str, str]:
    """Match light by `name` (case-insensitive, exact or substring). Vrátí
    (id, error). Pokud najde víc matchů, vrátí první alfabeticky a do error
    zapíše `multiple matches`. Pokud žádný match, error obsahuje seznam jmen
    pro UX. Stream-čte response s hard size cap (defense vs OOM)."""
    name_low = name.strip().lower()
    if not name_low:
        return "", "empty name"

    status, body, err = await _stream_get_capped(client, _hue_url("light"))
    if err:
        return "", f"Hue lookup error: {err}"
    if status != 200:
        return "", f"Hue lookup HTTP {status}"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return "", f"Hue lookup JSON error: {e}"
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return "", "Hue lookup data missing"

    exact: list[tuple[str, str]] = []
    partial: list[tuple[str, str]] = []
    all_names: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", ""))
        nm = str((item.get("metadata") or {}).get("name", ""))
        if not rid or not nm:
            continue
        all_names.append(nm)
        nm_low = nm.lower()
        if nm_low == name_low:
            exact.append((rid, nm))
        elif name_low in nm_low:
            partial.append((rid, nm))

    if exact:
        return exact[0][0], ""
    if len(partial) == 1:
        return partial[0][0], ""
    if len(partial) > 1:
        names = ", ".join(n for _, n in partial[:5])
        return "", f"multiple matches: {names}"
    sample = ", ".join(sorted(all_names)[:8])
    return "", f"light {name!r} not found. Available: {sample}"


async def _light_set_exec(args: dict, ctx: ExecuteContext) -> dict:
    """PUT /clip/v2/resource/light/<id> — toggle on/off, set brightness/color."""
    ok, why = _config_ok()
    if not ok:
        return {"ok": False, "error": why}

    name = str(args.get("name", "")).strip()
    if not name:
        return {"ok": False, "error": "name required"}

    body: dict[str, Any] = {}

    on_arg = args.get("on", None)
    if on_arg is not None:
        if not isinstance(on_arg, bool):
            return {"ok": False, "error": "on must be boolean"}
        body["on"] = {"on": on_arg}

    br_arg = args.get("brightness", None)
    if br_arg is not None:
        # Reject bool (Python treats bool as int subclass — disallow True/False
        # being mis-interpreted as brightness=1/0).
        if isinstance(br_arg, bool):
            return {"ok": False, "error": "brightness must be number"}
        try:
            br = float(br_arg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "brightness must be number"}
        # NaN/inf bypass: `nan < 0` and `nan > 100` both False, range check
        # would pass. Explicit reject via math.isfinite (covers NaN, +inf, -inf).
        if not math.isfinite(br):
            return {"ok": False, "error": "brightness must be finite number"}
        if br < 0 or br > 100:
            return {"ok": False, "error": "brightness out of range 0..100"}
        # Hue Bridge accepts 0..100, ale brightness=0 by light vypnul a "on"
        # zůstal True (= odpojené stavy). Clamp na 1.0 minimum když chceme set.
        if br <= 0:
            body["on"] = {"on": False}
        else:
            body["dimming"] = {"brightness": max(1.0, br)}

    color_arg = args.get("color_name", None)
    if color_arg is not None:
        cname = str(color_arg).strip().lower()
        if cname not in COLOR_PALETTE:
            palette = ", ".join(sorted(COLOR_PALETTE.keys()))
            return {
                "ok": False,
                "error": f"unknown color_name {cname!r}. Available: {palette}",
            }
        x, y = COLOR_PALETTE[cname]
        body["color"] = {"xy": {"x": x, "y": y}}

    if not body:
        return {
            "ok": False,
            "error": "must specify at least one of: on, brightness, color_name",
        }

    start = time.monotonic()
    rid = ""
    async with await _hue_client() as client:
        rid, lookup_err = await _resolve_light_id(client, name)
        if lookup_err:
            return {"ok": False, "error": lookup_err}
        # Resource ID from Hue bridge SHOULD be UUID-like, but defense in
        # depth: re-sanitize via _hue_url (rejects `..`, scheme, control chars).
        try:
            put_url = _hue_url(f"light/{rid}")
        except ValueError as e:
            return {"ok": False, "error": f"invalid resource id from bridge: {e}", "light_id": rid}
        status, _body_bytes, err = await _stream_request_capped(
            client, "PUT", put_url, json_body=body,
        )

    if err:
        return {"ok": False, "error": err, "light_id": rid}
    if status == 401 or status == 403:
        return {"ok": False, "error": "Hue auth error (key invalid?)"}
    if status >= 400:
        # Don't leak error body (could contain bridge internals).
        return {"ok": False, "error": f"Hue HTTP {status}", "light_id": rid}

    return {
        "ok": True,
        "light_id": rid,
        "name": name,
        "applied": list(body.keys()),
        "duration_ms": int((time.monotonic() - start) * 1000),
    }


# ----------------------------------------------------------------------
# Tool registrations
# ----------------------------------------------------------------------


LIGHT_LIST_TOOL = Tool(
    name="light_list",
    description=(
        "List all Philips Hue lights with their state (on/off, brightness, "
        "color). Returns: lights[{id, name, archetype, on, brightness, "
        "color_xy}], count, duration_ms. No arguments."
    ),
    parameters_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    execute=_light_list_exec,
)


LIGHT_SET_TOOL = Tool(
    name="light_set",
    description=(
        "Set state of a Philips Hue light by name. Args: name (required, "
        "case-insensitive exact or substring match), on (bool, optional), "
        "brightness (0-100 %, optional), color_name (optional, one of: warm, "
        "white, cool, daylight, red, orange, yellow, green, cyan, blue, "
        "purple, pink, magenta). At least one of on/brightness/color_name "
        "must be specified."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Light name (or substring).",
            },
            "on": {
                "type": "boolean",
                "description": "Turn on (true) or off (false).",
            },
            "brightness": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "description": "Brightness in percent (0-100).",
            },
            "color_name": {
                "type": "string",
                "description": "Color name from palette.",
            },
        },
        "required": ["name"],
        "additionalProperties": False,
    },
    execute=_light_set_exec,
)


ALL_TOOLS = (LIGHT_LIST_TOOL, LIGHT_SET_TOOL)
