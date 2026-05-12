"""Claude bridge tool — `ask_claude(prompt, system?, max_tokens?)`.

Deleguje na Anthropic Messages API. Použití:
  - když lokální Gemma narazí na limit (kontext / kvalita reasoningu / unknown)
  - jako "expert review" druhý názor v rámci řetězu úvah

Bezpečnostní model:
  - Classifier sanitizuje args (prompt délka, max_tokens range, system délka)
    a vrací ASK medium — tj. user musí explicit approve (nákladová ochrana).
  - SSRF-safe transport: reuse `_build_safe_client` z `web.py`, čímž zabráníme
    DNS rebinding TOCTOU + `trust_env=False` zruší proxy/cert override z prostředí.
  - Streaming response s tvrdým size capem `CLAUDE_OUTPUT_CAP_BYTES`. Content-Length
    pre-check + per-chunk counter (stejný pattern jako hue.py).
  - Accept-Encoding: identity — zabráníme decompression-bombu (response je auto-
    dekódovaná knihovnou před přečtením, takže bombu nelze chytit jen size capem
    na komprimovaných bytes). Identita = `len(response_body)` = ekvivalent uncompressed.
  - API klíč je v `Authorization: Bearer …`-stylu hlavičce `x-api-key`. Pokud klíč
    chybí, vrátíme structured error (žádný stack trace, žádný leak fragmentu env).
  - Žádné FS/shell side-effects. Tool je čistá síťová delegace.
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from voice.agent.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_API_ENDPOINT,
    CLAUDE_API_VERSION,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_MAX_PROMPT_BYTES,
    CLAUDE_MAX_SYSTEM_BYTES,
    CLAUDE_MAX_TOKENS_DEFAULT,
    CLAUDE_MAX_TOKENS_LIMIT,
    CLAUDE_OUTPUT_CAP_BYTES,
    CLAUDE_TIMEOUT_SEC,
)
from voice.agent.tools.base import ExecuteContext, Tool
from voice.agent.tools.web import _build_safe_client


async def _stream_post_capped(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any],
) -> tuple[int, bytes, str | None]:
    """POST s tvrdým size capem na response body.

    Vrací (status_code, body_bytes, error_or_None). Při překročení capu vrací
    error="response too large" a status, ale body je truncated (ne abort).
    Při Content-Length pre-check překročení capu vrátí (status, b"", error)
    a vůbec response body nečte.

    Compression defense: žádný Accept-Encoding != identity, ale i kdyby server
    poslal komprimovaný response, httpx by ho auto-dekódoval PŘED tím, než tady
    spočteme bytes — proto rejectneme jakékoli non-identity Content-Encoding.
    """
    try:
        async with client.stream("POST", url, headers=headers, json=json_body) as resp:
            # Content-Length pre-check (server may not send it; that's fine).
            cl = resp.headers.get("content-length")
            if cl is not None:
                try:
                    cl_int = int(cl)
                except ValueError:
                    return resp.status_code, b"", "invalid content-length"
                if cl_int > CLAUDE_OUTPUT_CAP_BYTES:
                    return resp.status_code, b"", (
                        f"response too large ({cl_int} > {CLAUDE_OUTPUT_CAP_BYTES} bytes)"
                    )
            ce = (resp.headers.get("content-encoding") or "").strip().lower()
            if ce and ce != "identity":
                return resp.status_code, b"", f"unexpected content-encoding {ce!r}"
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > CLAUDE_OUTPUT_CAP_BYTES:
                    # Zahodit přebytek, ale ne raisovat — vrátíme co máme.
                    take = CLAUDE_OUTPUT_CAP_BYTES - (total - len(chunk))
                    if take > 0:
                        chunks.append(chunk[:take])
                    return resp.status_code, b"".join(chunks), (
                        f"response truncated at {CLAUDE_OUTPUT_CAP_BYTES} bytes"
                    )
                chunks.append(chunk)
            return resp.status_code, b"".join(chunks), None
    except httpx.TimeoutException as e:
        return 0, b"", f"timeout: {type(e).__name__}"
    except httpx.RequestError as e:
        return 0, b"", f"request error: {type(e).__name__}"


def _extract_text(payload: dict) -> str:
    """Extract concatenated text from Anthropic Messages API response.

    Response shape: { "content": [{ "type": "text", "text": "..." }, ...], ... }
    """
    parts: list[str] = []
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            t = block.get("text")
            if isinstance(t, str):
                parts.append(t)
    return "".join(parts)


async def _ask_claude_exec(args: dict, ctx: ExecuteContext) -> dict:
    """Volání Anthropic Messages API."""
    # Re-validace (defense in depth — classifier už filtroval).
    prompt_arg = args.get("prompt")
    if not isinstance(prompt_arg, str):
        return {"ok": False, "error": "prompt must be string"}
    prompt = prompt_arg.strip()
    if not prompt:
        return {"ok": False, "error": "empty prompt"}
    try:
        if len(prompt.encode("utf-8")) > CLAUDE_MAX_PROMPT_BYTES:
            return {"ok": False, "error": "prompt too large"}
    except UnicodeEncodeError:
        return {"ok": False, "error": "prompt not valid utf-8"}

    system_arg = args.get("system")
    if system_arg is not None:
        if not isinstance(system_arg, str):
            return {"ok": False, "error": "system must be string"}
        try:
            if len(system_arg.encode("utf-8")) > CLAUDE_MAX_SYSTEM_BYTES:
                return {"ok": False, "error": "system too large"}
        except UnicodeEncodeError:
            return {"ok": False, "error": "system not valid utf-8"}

    max_tokens_arg = args.get("max_tokens")
    if max_tokens_arg is None:
        max_tokens = CLAUDE_MAX_TOKENS_DEFAULT
    else:
        if isinstance(max_tokens_arg, bool):
            return {"ok": False, "error": "max_tokens must be int, not bool"}
        try:
            max_tokens = int(max_tokens_arg)
        except (TypeError, ValueError):
            return {"ok": False, "error": "max_tokens must be integer"}
        if max_tokens < 1 or max_tokens > CLAUDE_MAX_TOKENS_LIMIT:
            return {"ok": False, "error": f"max_tokens out of range 1..{CLAUDE_MAX_TOKENS_LIMIT}"}

    if not ANTHROPIC_API_KEY:
        return {
            "ok": False,
            "error": (
                "ANTHROPIC_API_KEY not configured. Set env var or place key in "
                "~/.anthropic-api-key (mode 0600)."
            ),
        }

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": CLAUDE_API_VERSION,
        "content-type": "application/json",
        "accept": "application/json",
        "accept-encoding": "identity",
    }
    body: dict[str, Any] = {
        "model": CLAUDE_DEFAULT_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_arg is not None:
        body["system"] = system_arg

    timeout = httpx.Timeout(CLAUDE_TIMEOUT_SEC, connect=5.0)
    start = time.monotonic()
    async with _build_safe_client(timeout) as client:
        status, raw, err = await _stream_post_capped(
            client, CLAUDE_API_ENDPOINT, headers=headers, json_body=body,
        )

    duration_ms = int((time.monotonic() - start) * 1000)

    if status == 0 and err:
        return {"ok": False, "error": err, "duration_ms": duration_ms}
    if status == 401 or status == 403:
        return {"ok": False, "error": f"auth error (HTTP {status})", "status": status,
                "duration_ms": duration_ms}
    if status == 429:
        return {"ok": False, "error": "rate-limited (HTTP 429)", "status": 429,
                "duration_ms": duration_ms}
    if status >= 400:
        # API error response — body může obsahovat structured error JSON; vyparseuj
        # pokud lze, jinak vrať jen status.
        msg = f"Anthropic API HTTP {status}"
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8", errors="replace"))
                err_obj = payload.get("error")
                if isinstance(err_obj, dict):
                    api_msg = err_obj.get("message")
                    if isinstance(api_msg, str) and api_msg:
                        # Truncate, ne celé tělo (DoS na chat scrollback).
                        msg = f"Anthropic API HTTP {status}: {api_msg[:400]}"
            except (ValueError, UnicodeDecodeError):
                pass
        return {"ok": False, "error": msg, "status": status, "duration_ms": duration_ms}

    if err:
        # Truncation warning, ale status 200.
        return {"ok": False, "error": err, "status": status, "duration_ms": duration_ms}

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        return {"ok": False, "error": f"invalid JSON response: {type(e).__name__}",
                "duration_ms": duration_ms}

    text = _extract_text(payload)
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "ok": True,
        "model": CLAUDE_DEFAULT_MODEL,
        "text": text,
        "stop_reason": payload.get("stop_reason"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "duration_ms": duration_ms,
    }


ASK_CLAUDE_TOOL = Tool(
    name="ask_claude",
    description=(
        "Delegate a question or task to Anthropic Claude (paid external LLM). "
        "Use this for problems that exceed local model capability: complex "
        "reasoning, expert second opinion, code review, or when context is too "
        "large for local model. Returns Claude's text response. Each call costs "
        "money — use sparingly."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    f"User prompt for Claude. Max {CLAUDE_MAX_PROMPT_BYTES // 1024} KiB UTF-8."
                ),
            },
            "system": {
                "type": "string",
                "description": (
                    "Optional system prompt (role/persona). Max "
                    f"{CLAUDE_MAX_SYSTEM_BYTES // 1024} KiB UTF-8."
                ),
            },
            "max_tokens": {
                "type": "integer",
                "minimum": 1,
                "maximum": CLAUDE_MAX_TOKENS_LIMIT,
                "description": (
                    f"Maximum output tokens (default {CLAUDE_MAX_TOKENS_DEFAULT}, "
                    f"max {CLAUDE_MAX_TOKENS_LIMIT})."
                ),
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    },
    execute=_ask_claude_exec,
)


ALL_TOOLS = (ASK_CLAUDE_TOOL,)
