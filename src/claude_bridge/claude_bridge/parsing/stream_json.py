"""NDJSON stream-json parser pro `claude -p --output-format stream-json`.

Claude Code CLI s `-p --output-format stream-json --include-partial-messages`
emituje per-line JSON events:
  - {"type": "system", "subtype": "init", "session_id": "...", "model": "..."}
  - {"type": "stream_event", "event": {"type": "content_block_start|delta|stop", ...}}
  - {"type": "assistant", "message": {...}}      # accumulated assistant turn
  - {"type": "user", "message": {...}}           # tool_result blocks
  - {"type": "result", "subtype": "success|error", "result": "...", ...}

Parser dispatchne každý event na ProgressEvent payload (nebo None pokud
skip event jako rate_limit_event/mcp_*/heartbeat). Stavový stav (last
emit times, tool_use blocks under construction, accumulated assistant
text) je uložený v state dict co caller předává mezi voláními.

Refactor: dříve `voice/agent/claude_bridge.py::_parse_stream_event` +
helpery. Žádná funkční změna, jen package move.
"""
from __future__ import annotations

import json
import time
from typing import Any

# Throttle pro repetitive progress events (thinking, partial text).
_PROGRESS_THROTTLE_SEC = 1.5


def short_tool_input_summary(name: str, inp: dict | None) -> str:
    """Zkrácený popis tool_use vstupu pro progress event."""
    if not isinstance(inp, dict):
        return name
    if name in ("Edit", "Write", "Read", "NotebookEdit"):
        p = inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
        if isinstance(p, str):
            return f"{name} {p}"
    if name == "Bash":
        cmd = inp.get("command")
        if isinstance(cmd, str):
            return f"Bash: {cmd[:80]}"
    if name in ("Glob", "Grep"):
        pat = inp.get("pattern")
        if isinstance(pat, str):
            return f"{name} {pat}"
    return name


def parse_stream_event(
    obj: dict,
    state: dict,
) -> dict | None:
    """Mapuje jeden stream-json event na progress payload (nebo None).

    `state` mutable dict pro accumulating: `assistant_text`, `last_thinking_emit`,
    `tool_uses` (list str), `blocks` (dict[idx]->{name, input_buf}), `session_id`.

    Returns:
        Progress payload dict (drop-in kompatibilní s dnešním tool_progress
        event tvarem) nebo None pokud event nemá UI relevantní obsah.
    """
    t = obj.get("type")
    if t == "system":
        st = obj.get("subtype")
        if st == "init":
            # Codex iter-7: scalar fields guard - CLI mohl pošle non-string
            # session_id/model. Slice/format na non-string crash.
            sid = obj.get("session_id")
            sid_str = sid if isinstance(sid, str) else ""
            model = obj.get("model")
            model_str = model if isinstance(model, str) else "?"
            state["session_id"] = sid_str
            return {
                "stage": "started",
                "message": f"Claude {model_str} session {sid_str[:8]}",
                "session_id": sid_str,
                "model": model_str,
            }
        return None  # status/heartbeat - skip

    if t == "stream_event":
        # Codex iter-6: nested non-dict guard. `event` MUSÍ být dict.
        ev = obj.get("event")
        if not isinstance(ev, dict):
            return None
        et = ev.get("type")
        if et == "content_block_start":
            cb = ev.get("content_block")
            if not isinstance(cb, dict):
                return None
            cbt = cb.get("type")
            if cbt == "thinking":
                # Throttle: thinking emit max 1× per window
                now = time.monotonic()
                if now - state.get("last_thinking_emit", 0) < _PROGRESS_THROTTLE_SEC:
                    return None
                state["last_thinking_emit"] = now
                return {"stage": "thinking", "message": "přemýšlí…"}
            if cbt == "tool_use":
                # input ještě není kompletní (přijde přes input_json_delta).
                # Uložíme block state pro pozdější emit na content_block_stop.
                name_raw = cb.get("name")
                name = name_raw if isinstance(name_raw, str) else "?"
                idx = ev.get("index")
                blocks = state.setdefault("blocks", {})
                blocks[idx] = {"name": name, "input_buf": ""}
                state.setdefault("tool_uses", []).append(name)
                return None  # počkáme až bude input kompletní
            if cbt == "text":
                state["text_block_started"] = True
                return None
        if et == "content_block_delta":
            # Accumulate tool_use input bytes (Claude streams je incrementálně
            # jako input_json_delta).
            delta = ev.get("delta")
            if not isinstance(delta, dict):
                return None
            if delta.get("type") == "input_json_delta":
                partial = delta.get("partial_json")
                idx = ev.get("index")
                blocks = state.get("blocks") or {}
                blk = blocks.get(idx)
                if blk is not None and isinstance(partial, str):
                    blk["input_buf"] = blk.get("input_buf", "") + partial
            return None
        if et == "content_block_stop":
            # Emit tool_use progress s kompletním inputem.
            idx = ev.get("index")
            blocks = state.get("blocks") or {}
            blk = blocks.pop(idx, None)
            if blk is None:
                return None
            name = blk.get("name", "?")
            buf = blk.get("input_buf", "")
            try:
                inp = json.loads(buf) if buf else None
            except (ValueError, TypeError):
                inp = None
            summary = short_tool_input_summary(name, inp)
            return {
                "stage": "tool_use",
                "message": summary,
                "tool_name": name,
                "input": inp if isinstance(inp, dict) else None,
            }
        return None

    if t == "assistant":
        # Accumulated assistant message; extrahuj text pro finální result.
        msg = obj.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        text_parts = [
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if text_parts:
            state["assistant_text"] = "".join(text_parts)
        return None

    if t == "user":
        # User-role message obsahuje tool_result.
        msg = obj.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                is_err = b.get("is_error") is True
                return {
                    "stage": "tool_result",
                    "message": "tool selhal" if is_err else "tool OK",
                    "ok": not is_err,
                }
        return None

    if t == "result":
        # Finální event - caller ho zpracuje mimo dispatcher (terminal).
        return None

    return None  # rate_limit_event, mcp_*, unknown → skip
