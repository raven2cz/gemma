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

import hashlib
import json
import uuid
from typing import Any


_PARSE_ERROR_KEY = "_parse_error"
_MAX_ARGS_BYTES = 64 * 1024  # 64 KiB strop pro raw arguments string (DoS guard
# + Ollama context budget). Větší input → sentinel `too_large`, hash & length
# zachovány pro forensics.


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


def _make_parse_error_sentinel(raw: Any, reason: str = "invalid_json") -> dict:
    """Vrátí sentinel dict pro malformed tool_call arguments.

    Sentinel obsahuje:
    - `_parse_error`: zkratku důvodu (vždy přítomné v sentinel)
    - `raw_length`: byte délka původního stringu (forensic stopa bez leaku)
    - `raw_hash`: sha256 HEX plného originálního raw stringu před `.strip()`
                  (forensic deduplikace + integrity check). Hash celé,
                  netrimnuté hodnoty proto, aby útočník nemohl jeden
                  malicious payload "schovat" za leading/trailing whitespace
                  a vytvořit duplikát s jiným hashem.

    **NEZAHRNUJEME raw_preview**: model může do args zakódovat secret
    (vyhalucinovaný API token z paměti) nebo cílený exfil text. Preview
    by tento secret přešel z LLM contextu do history + audit logu +
    `tool_call` eventu (= leak do logu/UI/disku). Místo toho: hash +
    délka stačí pro forensické porovnání ("je tohle stejný payload?"),
    a auditní stopu drží `tool_call.id` + timestamp.
    """
    if isinstance(raw, (bytes, bytearray)):
        raw_bytes = bytes(raw)
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8", errors="replace")
    else:
        # Pro non-string typy (list, int, None) reprezentujeme JSON-style
        # ať hash zachytí přesný typ ("[1,2,3]" vs "null" vs "false").
        # Codex+Gemini iter-3 HIGH: i sentinel-tvorba musí být uncaught-free.
        # `RecursionError` (cyklický dict) a libovolná další exception z
        # json.dumps musí dopadnout na `repr` fallback — sentinel nesmí spadnout,
        # protože je to záchranná cesta untrusted vstupu.
        try:
            raw_bytes = json.dumps(raw, ensure_ascii=False).encode("utf-8")
        except Exception:
            try:
                raw_bytes = repr(raw).encode("utf-8", errors="replace")
            except Exception:
                raw_bytes = b"<unrepresentable>"
    return {
        _PARSE_ERROR_KEY: reason,
        "raw_length": len(raw_bytes),
        "raw_hash": hashlib.sha256(raw_bytes).hexdigest(),
    }


def canonicalize_arguments(raw: Any) -> str:
    """Vrátí GARANTOVANĚ validní JSON object string pro `tool_call.function.arguments`.

    Při invalid vstupu (truncated JSON, non-object JSON, non-string non-dict
    typ, příliš velký payload) vrátí serialized sentinel s `_parse_error` key.
    Ollama tak dostane parseable JSON a nevyhodí 400; loop pak v `_execute_one`
    detekuje sentinel přes `is_malformed_args` a tool NESPUSTÍ.

    Bezpečnostně kritické: bez canonicalize by model mohl `{"command": "rm -rf /"`
    (bez `}`) → parse_tool_args fallback `{}` → permission classifier vidí
    prázdné args → tool dostane defaults → destructive akce mimo audit.
    Stejně tak `arguments: []` nebo `arguments: 0` (non-string types) musí
    NEPROJÍT na prázdný `{}` — místo toho sentinel `invalid_type`.

    Typové chování:
    - `dict`               → reserializovaný JSON object string (cap + try-safe)
    - `str` empty / blank  → `"{}"` (model neposlal args; defaults expected)
    - `str` > 64 KiB       → sentinel `too_large` (DoS guard)
    - `str` invalid JSON   → sentinel `invalid_json`
    - `str` non-object JSON (`[1]`, `"x"`, `123`)  → sentinel `non_object_json`
    - `None` / missing     → `"{}"` (model neposlal args; required-fields
                              validace probíhá v Ollama tool schema)
    - jiný typ (`list`, `int`, `bool`, …) → sentinel `invalid_type`
    """
    if isinstance(raw, dict):
        # Codex iter-2 HIGH: dict path NESMÍ bypassovat size cap ani spadnout
        # na uncaught exception při non-JSON-serializable hodnotách / cyklech.
        try:
            serialized = json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return json.dumps(
                _make_parse_error_sentinel(raw, reason="invalid_type"),
                ensure_ascii=False,
            )
        if len(serialized.encode("utf-8", errors="replace")) > _MAX_ARGS_BYTES:
            return json.dumps(
                _make_parse_error_sentinel(serialized, reason="too_large"),
                ensure_ascii=False,
            )
        return serialized
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        # Size cap: před parsováním ať velký payload nevybuchne v json.loads.
        if len(raw.encode("utf-8", errors="replace")) > _MAX_ARGS_BYTES:
            return json.dumps(
                _make_parse_error_sentinel(raw, reason="too_large"),
                ensure_ascii=False,
            )
        s = raw.strip()
        if not s:
            return "{}"
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, RecursionError, ValueError):
            # `raw` (NE `s`) → hash je z původního před-trim payloadu
            # (forensic integrity: whitespace-only diff nesmí kolidovat).
            return json.dumps(
                _make_parse_error_sentinel(raw, reason="invalid_json"),
                ensure_ascii=False,
            )
        if not isinstance(parsed, dict):
            # `arguments` musí být JSON object per OpenAI/Ollama spec; array,
            # number, string atd. jsou malformed → sentinel.
            return json.dumps(
                _make_parse_error_sentinel(raw, reason="non_object_json"),
                ensure_ascii=False,
            )
        # Validní JSON object — re-serializuj kanonicky (drop whitespace).
        return json.dumps(parsed, ensure_ascii=False)
    # Non-string, non-dict, non-None: list, int, bool, …
    # NESMÍ projít na "{}" — to by umožnilo `arguments: []` skrýt skutečný
    # malicious obsah (např. arguments serializovaný adversarialním klientem).
    return json.dumps(
        _make_parse_error_sentinel(raw, reason="invalid_type"),
        ensure_ascii=False,
    )


def is_malformed_args(args: dict | Any) -> bool:
    """True, pokud args je sentinel z canonicalize_arguments (parse_error)."""
    return isinstance(args, dict) and _PARSE_ERROR_KEY in args


def normalize_tool_calls(raw: list[dict]) -> list[dict]:
    """Ollama vrací tool_calls jako list[{function: {name, arguments}}].

    Sjednocujeme na OpenAI tvar: arguments JSON string (vždy validní —
    invalid input → sentinel přes `canonicalize_arguments`), povinný name,
    deduplikace podle `id` (Ollama může v rámci streamování poslat stejný
    call víckrát, případně inkrementálně po chunkách — bez dedupy by se
    tool spustil 2×).

    Tool calls bez `function.name` jsou DROPNUTÉ — místo aby agent vykonal něco
    s prázdným jménem. Last-write-wins pro arguments (nejnovější verze chunku).
    """
    seen: dict[str, dict] = {}
    order: list[str] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        # Codex iter-3 HIGH: `function` může být truthy non-dict (model či
        # adversarial chunk) — `.get` by spadl na AttributeError. Vyžadujeme dict.
        fn_raw = tc.get("function")
        fn = fn_raw if isinstance(fn_raw, dict) else {}
        name_raw = fn.get("name")
        name = (name_raw if isinstance(name_raw, str) else "").strip()
        if not name:
            continue  # malformed — drop
        args_str = canonicalize_arguments(fn.get("arguments"))
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
    """Vrátí args jako dict, nehledě na jejich formát v history.

    Při invalid JSON / non-object JSON / non-string non-dict typu / příliš
    velkém payloadu vrací sentinel s `_parse_error` klíčem (NE prázdný `{}`
    — to by umožnilo modelu schovat malicious args za malformed JSON:
    classifier by viděl `{}` místo `{"command":"rm -rf /"` a tool by se
    spustil s defaults).

    Callsite v `_execute_one` musí `is_malformed_args(args)` detekovat
    a tool NESPUSTIT.

    Mapování typů viz `canonicalize_arguments` — chování je symetrické,
    jen vrací dict místo serialized stringu.
    """
    # Codex iter-3 HIGH: `function` může být truthy non-dict (`[1]`, `123`,
    # custom object) — `.get(...)` by spadl na AttributeError. Vyžadujeme dict.
    fn_raw = tool_call.get("function")
    fn = fn_raw if isinstance(fn_raw, dict) else {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        # Codex iter-2 HIGH: dict path NESMÍ bypassovat size cap.
        # Serialize-and-measure (pokud non-serializable → sentinel).
        try:
            serialized = json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError, RecursionError):
            return _make_parse_error_sentinel(raw, reason="invalid_type")
        if len(serialized.encode("utf-8", errors="replace")) > _MAX_ARGS_BYTES:
            return _make_parse_error_sentinel(serialized, reason="too_large")
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        if len(raw.encode("utf-8", errors="replace")) > _MAX_ARGS_BYTES:
            return _make_parse_error_sentinel(raw, reason="too_large")
        s = raw.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return _make_parse_error_sentinel(raw, reason="invalid_json")
        if isinstance(parsed, dict):
            return parsed
        return _make_parse_error_sentinel(raw, reason="non_object_json")
    return _make_parse_error_sentinel(raw, reason="invalid_type")
