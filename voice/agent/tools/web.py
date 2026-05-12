"""Web tooly — `fetch_url` (HTTP GET) + `web_search` (Brave Search API).

Bezpečnostní model:

- `fetch_url(url)`:
    * Classifier zkontroloval scheme (http/https) a host (SSRF deny pro
      private/loopback IP literal). Tool dělá second-pass SSRF check **po DNS
      resolve** — runtime check protože classifier nedělá DNS lookup.
    * Redirect chain: ručně iterujeme, KAŽDÝ hop re-validuje scheme + host
      přes _validate_url + DNS check (jinak by 30x → private IP obejel SSRF).
    * Response size cap: zastav stream při `FETCH_URL_MAX_BYTES`.
    * Content-Type filter: text-like → body; binary → jen metadata.
    * User-Agent: explicit agent identity (`gemma-agent/0.1`).
    * Env keys nikdy nevkládáme do požadavku (jen UA + Accept).

- `web_search(query, count?)`:
    * Brave Search API; klíč v Authorization-style header `X-Subscription-Token`.
    * Body cap; timeout; results se sklerotizují na {title, url, snippet}.
    * Žádný shell, žádný FS write.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
import typing
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import httpcore
import httpx
from httpcore._backends.anyio import AnyIOBackend
from httpcore._backends.base import SOCKET_OPTION, AsyncNetworkBackend, AsyncNetworkStream

from voice.agent.config import (
    BRAVE_SEARCH_API_KEY,
    BRAVE_SEARCH_ENDPOINT,
    FETCH_URL_ALLOWED_SCHEMES,
    FETCH_URL_MAX_BYTES,
    FETCH_URL_MAX_REDIRECTS,
    FETCH_URL_TEXT_CONTENT_TYPES,
    FETCH_URL_TIMEOUT_SEC,
    FETCH_URL_USER_AGENT,
    WEB_SEARCH_DEFAULT_COUNT,
    WEB_SEARCH_MAX_COUNT,
    WEB_SEARCH_TIMEOUT_SEC,
)
from voice.agent.permissions import _is_private_or_blocked_host, _validate_url
from voice.agent.tools.base import ExecuteContext, Tool


# ----------------------------------------------------------------------
# SSRF-safe transport (HIGH-1 defense vůči DNS rebinding TOCTOU)
# ----------------------------------------------------------------------
#
# Problém: bez tohoto backendu fetch udělá DNS dvakrát:
#   1) `_resolve_host_safe` přes loop.getaddrinfo (validation)
#   2) httpx → httpcore → anyio.connect_tcp → další getaddrinfo (skutečný connect)
# Mezi (1) a (2) může útočník vrátit jinou IP (DNS rebinding) — connect by
# tedy mohl jet na 169.254.169.254 i když validation viděla 1.2.3.4.
#
# Řešení: jeden custom backend, který:
#   a) udělá SINGLE DNS resolve pro každý connect_tcp call
#   b) zvalidují VŠECHNY vrácené adresy přes _is_private_or_blocked_host
#   c) zavolá inner backend s IP literal (= žádný další DNS lookup)
#   d) host hostname je předán dál jen pro SNI/TLS verify (start_tls dostává
#      `server_hostname` parametr → původní hostname pro cert validation)


class _SSRFGuardBackend(AsyncNetworkBackend):
    """Wrapped backend that resolves DNS once and validates every IP.

    Replaces the host argument in `connect_tcp` with a literal IP that has
    been confirmed public. The original hostname is no longer used for the
    socket connect, but TLS layer (httpcore's `start_tls`) still passes the
    original hostname as `server_hostname` for SNI + cert verification.
    """

    def __init__(self, inner: AsyncNetworkBackend) -> None:
        self._inner = inner

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        # Strip [ ] brackets z IPv6 literal (httpx posílá `[::1]` → host=`::1`).
        h = host
        if h.startswith("[") and h.endswith("]"):
            h = h[1:-1]

        # Pokud je to už IP literal, validuj a předej dál (žádný DNS call).
        try:
            ipaddress.ip_address(h)
            is_literal = True
        except ValueError:
            is_literal = False

        if is_literal:
            blocked, why = _is_private_or_blocked_host(h)
            if blocked:
                raise httpcore.ConnectError(f"SSRF blocked: {why}")
            connect_target = h
        else:
            # DNS resolve — single call, sjednoceno přes loop.getaddrinfo
            # (= stejný resolver jako `_resolve_host_safe` precheck, ale jen JEDNOU).
            loop = asyncio.get_running_loop()
            try:
                infos = await loop.getaddrinfo(
                    h, port, type=socket.SOCK_STREAM,
                )
            except (socket.gaierror, OSError) as e:
                raise httpcore.ConnectError(f"DNS resolution failed: {e}")
            if not infos:
                raise httpcore.ConnectError("DNS returned no addresses")

            # Validuj VŠECHNY adresy. Pokud jakákoli je blokovaná, abort
            # (= konzervativní stance — rebind by mohl swapnout do private).
            chosen_ip: str | None = None
            for info in infos:
                family = info[0]
                sockaddr = info[4]
                addr = sockaddr[0]
                blocked, why = _is_private_or_blocked_host(addr)
                if blocked:
                    raise httpcore.ConnectError(
                        f"SSRF blocked: DNS for {host!r} resolved to {why}"
                    )
                if chosen_ip is None:
                    # Prefer IPv4 simplicity; IPv6 funguje ale brackety níž.
                    chosen_ip = addr
            assert chosen_ip is not None
            connect_target = chosen_ip

        # Pro IPv6 connect target potřebuje brackety (anyio passuje rovnou
        # raw stringu). Některé backendy si to dotahují samy, ale konzistence:
        try:
            parsed = ipaddress.ip_address(connect_target)
            if isinstance(parsed, ipaddress.IPv6Address):
                connect_target_for_inner = connect_target  # anyio.connect_tcp
                # akceptuje raw `::1` bez brackets.
            else:
                connect_target_for_inner = connect_target
        except ValueError:
            connect_target_for_inner = connect_target

        return await self._inner.connect_tcp(
            connect_target_for_inner,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: typing.Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        # Unix sockety nikdy nepovolujeme z agent web tools (SSRF na docker.sock).
        raise httpcore.ConnectError("unix sockets not permitted")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def _build_safe_client(
    timeout: httpx.Timeout, headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """httpx.AsyncClient s SSRF-safe transportem + bez compression/proxy/env."""
    transport = httpx.AsyncHTTPTransport()
    # Reach into transport internals to swap pool's network_backend.
    pool = transport._pool  # type: ignore[attr-defined]
    pool._network_backend = _SSRFGuardBackend(AnyIOBackend())
    return httpx.AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=False,
        headers=headers or {},
        trust_env=False,
    )


# ----------------------------------------------------------------------
# fetch_url
# ----------------------------------------------------------------------


async def _resolve_host_safe(host: str) -> tuple[bool, str]:
    """DNS resolve + per-address SSRF check.

    Vrátí (ok, reason). ok=False = blokovaná IP nebo unresolvable.
    Akceptujeme jen pokud VŠECHNY rozlišené adresy jsou veřejné — pokud kterákoli
    míří na privátní rozsah, reject (DNS rebinding defense).
    """
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(
            host, None, type=socket.SOCK_STREAM,
        )
    except (socket.gaierror, OSError) as e:
        return False, f"dns: {e}"
    if not infos:
        return False, "dns: no addresses"
    for info in infos:
        sockaddr = info[4]
        addr = sockaddr[0]
        blocked, why = _is_private_or_blocked_host(addr)
        if blocked:
            return False, why
    return True, ""


async def _fetch_once(
    client: httpx.AsyncClient, url: str, max_bytes: int,
) -> tuple[httpx.Response, bytes, bool, str | None]:
    """Single GET. Streamuje response s size cap.

    Vrátí (response, body_bytes, truncated, encoding_error).
    encoding_error != None pokud server vrátil non-identity Content-Encoding
    (= gzip/zstd bomb defense — HIGH-2). Tělo decompresí stejně neulekáme,
    aby cap platil; rovnou abortujeme.
    """
    truncated = False
    buf = bytearray()
    async with client.stream("GET", url) as resp:
        # HIGH-2: defense vůči compression bomby. Žádali jsme `Accept-Encoding:
        # identity`, takže server *neměl* compress posílat. Pokud poslal anyway,
        # rovnou aborte — buď je to nekonformní server nebo pokus o exploit.
        ct_enc = resp.headers.get("Content-Encoding", "").strip().lower()
        if ct_enc and ct_enc not in ("identity", ""):
            return resp, b"", False, f"server sent disallowed Content-Encoding {ct_enc!r}"

        async for chunk in resp.aiter_bytes():
            remaining = max_bytes - len(buf)
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                buf.extend(chunk[:remaining])
                truncated = True
                break
            buf.extend(chunk)
        # Drain not needed — aclose on exit cancels remaining stream.
    return resp, bytes(buf), truncated, None


def _is_text_content_type(ct: str) -> bool:
    """True pokud content-type začíná některým allowed text prefixem."""
    if not ct:
        return False
    ct_low = ct.lower().split(";", 1)[0].strip()
    return any(ct_low.startswith(p) for p in FETCH_URL_TEXT_CONTENT_TYPES)


async def _fetch_url_exec(args: dict, ctx: ExecuteContext) -> dict:
    """Fetch URL s SSRF defense + redirect chain re-validation.

    Důležité: httpx default follow_redirects=False — my redirecty řešíme
    manuálně, abychom mohli každý hop ověřit (SSRF přes 30x na private IP).
    """
    url = str(args.get("url", "")).strip()

    # Re-validate (defense in depth — classifier už ověřil).
    scheme, host, err = _validate_url(url)
    if err:
        return {"ok": False, "error": err}

    # UX fail-fast DNS check (Layer 1).
    # Reálná SSRF garance přichází z `_SSRFGuardBackend` (Layer 2) — ten dělá
    # DNS resolve + validation + IP-pinned connect přímo před socketem.
    # Mezi Layer 1 a Layer 2 by rebind mohl swapnout DNS odpověď, ale Layer 2
    # to chytne (nezávislý lookup) a stejně by connect šel na první pinned IP.
    ok, why = await _resolve_host_safe(host)
    if not ok:
        return {"ok": False, "error": f"ssrf: {why}"}

    start = time.monotonic()
    headers = {
        "User-Agent": FETCH_URL_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.7",
        "Accept-Language": "en,cs;q=0.8",
        # HIGH-2: vyhneme se compression bomb — žádný gzip/zstd, vidíme wire-bytes.
        "Accept-Encoding": "identity",
    }

    current_url = url
    redirect_count = 0
    timeout = httpx.Timeout(FETCH_URL_TIMEOUT_SEC, connect=10.0)
    redirect_chain: list[str] = []

    try:
        async with _build_safe_client(timeout, headers) as client:
            while True:
                if redirect_count > FETCH_URL_MAX_REDIRECTS:
                    return {
                        "ok": False,
                        "error": f"too many redirects (>{FETCH_URL_MAX_REDIRECTS})",
                        "redirect_chain": redirect_chain,
                    }

                # Cancel check.
                if ctx.cancel_event is not None and ctx.cancel_event.is_set():
                    return {"ok": False, "error": "cancelled"}

                resp, body, truncated, enc_err = await _fetch_once(
                    client, current_url, FETCH_URL_MAX_BYTES,
                )
                if enc_err is not None:
                    return {
                        "ok": False,
                        "error": enc_err,
                        "redirect_chain": redirect_chain,
                    }

                # Redirect?
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("Location", "")
                    if not loc:
                        # 30x bez Location — break jako final.
                        break
                    next_url = urljoin(current_url, loc)
                    nscheme, nhost, nerr = _validate_url(next_url)
                    if nerr:
                        return {
                            "ok": False,
                            "error": f"redirect to invalid url: {nerr}",
                            "redirect_chain": redirect_chain + [next_url],
                        }
                    # Layer 1 DNS precheck pro next hop — Layer 2 (_SSRFGuardBackend)
                    # by stejně chytlo, ale precheck umožňuje fail-fast + tests.
                    ok2, why2 = await _resolve_host_safe(nhost)
                    if not ok2:
                        return {
                            "ok": False,
                            "error": f"redirect ssrf: {why2}",
                            "redirect_chain": redirect_chain + [next_url],
                        }
                    redirect_chain.append(next_url)
                    current_url = next_url
                    redirect_count += 1
                    continue

                # Final response.
                content_type = resp.headers.get("Content-Type", "")
                is_text = _is_text_content_type(content_type)
                body_text = ""
                if is_text:
                    # Encoding hint: httpx response.encoding (Content-Type charset)
                    # fallback na utf-8 s replace.
                    enc = resp.encoding or "utf-8"
                    try:
                        body_text = body.decode(enc, errors="replace")
                    except (LookupError, TypeError):
                        body_text = body.decode("utf-8", errors="replace")

                return {
                    "ok": True,
                    "url": current_url,
                    "final_url": str(resp.url),
                    "status": resp.status_code,
                    "content_type": content_type,
                    "size_bytes": len(body),
                    "truncated": truncated,
                    "is_text": is_text,
                    "body": body_text if is_text else "",
                    "redirect_chain": redirect_chain,
                    "duration_ms": int((time.monotonic() - start) * 1000),
                }

            # If we broke out of the loop without returning (rare 30x no-location):
            return {
                "ok": False,
                "error": "redirect without Location header",
                "redirect_chain": redirect_chain,
            }
    except httpx.TimeoutException as e:
        return {"ok": False, "error": f"timeout: {e}"}
    except httpx.DecodingError as e:
        # HIGH-2: server poslal compressed response (Content-Encoding) i přes
        # naše Accept-Encoding: identity. Either invalid gzip nebo bomb.
        return {
            "ok": False,
            "error": (
                f"disallowed Content-Encoding (compressed response rejected): {e}"
            ),
        }
    except httpx.RequestError as e:
        return {"ok": False, "error": f"request error: {type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001 — defensive catch-all pro LLM input
        return {"ok": False, "error": f"unexpected: {type(e).__name__}: {e}"}


FETCH_URL_TOOL = Tool(
    name="fetch_url",
    description=(
        "HTTP GET a URL and return its body as text (for HTML/JSON/XML/text "
        "content types). Only http/https schemes; private/internal IPs and "
        "localhost are blocked (SSRF defense). Follows up to 5 redirects, each "
        "re-validated. Body capped at ~2 MiB; binary responses return metadata "
        "only. Returns: url, status, content_type, body, size_bytes, truncated, "
        "is_text, redirect_chain, duration_ms."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Absolute http(s) URL.",
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    execute=_fetch_url_exec,
)


# ----------------------------------------------------------------------
# web_search (Brave Search API)
# ----------------------------------------------------------------------


def _brave_extract_results(payload: dict, count: int) -> list[dict]:
    """Vytahuje web results z Brave API response.

    Pozn.: payload struktura: payload["web"]["results"] = list dictů s
    title/url/description (+ další pole jako age, language, profile…).
    """
    web = payload.get("web") if isinstance(payload, dict) else None
    if not isinstance(web, dict):
        return []
    raw = web.get("results")
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw[:count]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        snippet = str(item.get("description", "")).strip()
        if not url:
            continue
        out.append({"title": title, "url": url, "snippet": snippet})
    return out


async def _web_search_exec(args: dict, ctx: ExecuteContext) -> dict:
    """Brave Search API call."""
    query = str(args.get("query", "")).strip()
    count_arg = args.get("count", None)
    if count_arg is None:
        count = WEB_SEARCH_DEFAULT_COUNT
    else:
        try:
            count = int(count_arg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid count"}
    if count < 1 or count > WEB_SEARCH_MAX_COUNT:
        return {"ok": False, "error": f"count out of range 1..{WEB_SEARCH_MAX_COUNT}"}
    if not query:
        return {"ok": False, "error": "empty query"}

    if not BRAVE_SEARCH_API_KEY:
        return {
            "ok": False,
            "error": (
                "BRAVE_SEARCH_API_KEY not configured. Set env var or place "
                "key in ~/.brave-search-api (mode 0600)."
            ),
        }

    start = time.monotonic()
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
    }
    params: dict[str, Any] = {"q": query, "count": str(count)}

    timeout = httpx.Timeout(WEB_SEARCH_TIMEOUT_SEC, connect=5.0)
    try:
        async with _build_safe_client(timeout) as client:
            resp = await client.get(
                BRAVE_SEARCH_ENDPOINT, params=params, headers=headers,
            )
    except httpx.TimeoutException as e:
        return {"ok": False, "error": f"timeout: {e}"}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"request error: {type(e).__name__}: {e}"}

    if resp.status_code == 429:
        return {"ok": False, "error": "rate-limited (HTTP 429)", "status": 429}
    if resp.status_code == 401 or resp.status_code == 403:
        return {"ok": False, "error": f"auth error (HTTP {resp.status_code})"}
    if resp.status_code >= 400:
        return {
            "ok": False,
            "error": f"Brave Search HTTP {resp.status_code}",
            "status": resp.status_code,
        }

    try:
        payload = resp.json()
    except ValueError as e:
        return {"ok": False, "error": f"invalid JSON: {e}"}

    results = _brave_extract_results(payload, count)
    return {
        "ok": True,
        "query": query,
        "count": len(results),
        "results": results,
        "duration_ms": int((time.monotonic() - start) * 1000),
    }


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web via Brave Search and return up to 20 results "
        "(default 5). Each result has {title, url, snippet}. Use this to find "
        "current information, documentation, or sources to follow up with "
        "fetch_url."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (max 400 chars).",
            },
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": WEB_SEARCH_MAX_COUNT,
                "description": f"Number of results (1..{WEB_SEARCH_MAX_COUNT}, default {WEB_SEARCH_DEFAULT_COUNT}).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    execute=_web_search_exec,
)


ALL_TOOLS = (FETCH_URL_TOOL, WEB_SEARCH_TOOL)
