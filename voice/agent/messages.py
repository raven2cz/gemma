"""Conversation history schema pro agent mode.

Používáme OpenAI / Ollama format (kompatibilní s `tools` parametrem):

    {"role": "system", "content": "..."}
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [
        {"id": "tc_1", "type": "function",
         "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"}}
    ]}
    {"role": "tool", "tool_call_id": "tc_1", "name": "echo", "content": "<result>"}

`arguments` je vždy JSON STRING (ne object) — tak to chce Ollama i OpenAI.
`content` v tool message je vždy string (často JSON-encoded result).
"""
from __future__ import annotations

import json
import uuid
from typing import Any


def new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex[:10]}"


def new_approval_id() -> str:
    return f"ap_{uuid.uuid4().hex[:10]}"


def assistant_message(text: str, tool_calls: list[dict] | None = None) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": text or ""}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tool_message(tool_call_id: str, name: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


def normalize_tool_calls(raw: list[dict]) -> list[dict]:
    """Ollama vrací tool_calls jako list[{function: {name, arguments}}].

    Sjednocujeme na OpenAI tvar: arguments JSON string, povinný name, deduplikace
    podle `id` (Ollama může v rámci streamování poslat stejný call víckrát, případně
    inkrementálně po chunkách — bez dedupy by se tool spustil 2×).

    Tool calls bez `function.name` jsou DROPNUTÉ — místo aby agent vykonal něco
    s prázdným jménem. Last-write-wins pro arguments (nejnovější verze chunku).
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = (fn.get("name") or "").strip()
        if not name:
            continue  # malformed — drop
        raw_args = fn.get("arguments")
        if isinstance(raw_args, dict):
            args_str = json.dumps(raw_args, ensure_ascii=False)
        elif isinstance(raw_args, str):
            args_str = raw_args.strip() or "{}"
        else:
            args_str = "{}"
        tcid = tc.get("id")
        if not isinstance(tcid, str) or not tcid:
            tcid = new_tool_call_id()
        entry = {
            "id": tcid,
            "type": "function",
            "function": {"name": name, "arguments": args_str},
        }
        if tcid in seen:
            # Dedup: nejnovější chunk přepíše args/name (typicky stejné).
            seen[tcid] = entry
        else:
            seen[tcid] = entry
            order.append(tcid)
    return [seen[tcid] for tcid in order]


def parse_tool_args(tool_call: dict) -> dict:
    """Vrátí args jako dict, nehledě na jejich formát v history."""
    fn = tool_call.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
