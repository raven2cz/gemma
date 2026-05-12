"""Test conversation history schema (voice/agent/messages.py)."""
from __future__ import annotations

import json

from voice.agent.messages import (
    assistant_message,
    canonicalize_arguments,
    is_malformed_args,
    new_approval_id,
    new_tool_call_id,
    normalize_tool_calls,
    parse_tool_args,
    tool_message,
)


def test_new_tool_call_id_unique():
    a, b = new_tool_call_id(), new_tool_call_id()
    assert a != b
    assert a.startswith("tc_")


def test_new_approval_id_unique():
    a, b = new_approval_id(), new_approval_id()
    assert a != b
    assert a.startswith("ap_")


def test_assistant_message_text_only():
    m = assistant_message("hello")
    assert m == {"role": "assistant", "content": "hello"}


def test_assistant_message_with_tool_calls():
    tcs = [{"id": "tc_1", "type": "function",
            "function": {"name": "echo", "arguments": "{}"}}]
    m = assistant_message("", tool_calls=tcs)
    assert m["role"] == "assistant"
    assert m["content"] == ""
    assert m["tool_calls"] == tcs


def test_tool_message_shape():
    m = tool_message("tc_1", "echo", '{"echoed":"hi"}')
    assert m == {
        "role": "tool",
        "tool_call_id": "tc_1",
        "name": "echo",
        "content": '{"echoed":"hi"}',
    }


def test_normalize_tool_calls_with_dict_args():
    raw = [{"function": {"name": "echo", "arguments": {"text": "hi"}}}]
    out = normalize_tool_calls(raw)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "echo"
    # arguments musí být JSON string, ne dict
    assert isinstance(out[0]["function"]["arguments"], str)
    assert json.loads(out[0]["function"]["arguments"]) == {"text": "hi"}
    assert out[0]["id"].startswith("tc_")


def test_normalize_tool_calls_with_string_args():
    raw = [{"id": "x1", "function": {"name": "echo", "arguments": '{"text":"hi"}'}}]
    out = normalize_tool_calls(raw)
    assert out[0]["id"] == "x1"
    assert json.loads(out[0]["function"]["arguments"]) == {"text": "hi"}


def test_normalize_tool_calls_with_empty_args():
    raw = [{"function": {"name": "echo", "arguments": ""}}]
    out = normalize_tool_calls(raw)
    assert out[0]["function"]["arguments"] == "{}"


def test_normalize_tool_calls_with_missing_args():
    raw = [{"function": {"name": "echo"}}]
    out = normalize_tool_calls(raw)
    assert out[0]["function"]["arguments"] == "{}"


def test_parse_tool_args_dict():
    tc = {"function": {"name": "x", "arguments": {"k": 1}}}
    assert parse_tool_args(tc) == {"k": 1}


def test_parse_tool_args_string():
    tc = {"function": {"name": "x", "arguments": '{"k":1}'}}
    assert parse_tool_args(tc) == {"k": 1}


def test_parse_tool_args_invalid_json_returns_sentinel():
    """Invalid JSON args → sentinel s `_parse_error` (NE prázdný `{}`).

    Bezpečnostně kritické: prázdný `{}` by umožnil modelu schovat malicious
    args za malformed JSON — classifier i tool by viděly prázdné args
    místo skutečného (potenciálně destruktivního) obsahu.
    """
    tc = {"function": {"name": "x", "arguments": "not json"}}
    out = parse_tool_args(tc)
    assert out["_parse_error"] == "invalid_json"
    assert out["raw_length"] == len("not json")
    assert len(out["raw_hash"]) == 64  # sha256 hex
    # Žádný raw_preview — secret leak prevention (gemini+codex review HIGH)
    assert "raw_preview" not in out


def test_parse_tool_args_non_dict_json_returns_sentinel():
    """Validní JSON, ale ne object (např. array) → také sentinel."""
    tc = {"function": {"name": "x", "arguments": "[1,2,3]"}}
    out = parse_tool_args(tc)
    assert out["_parse_error"] == "non_object_json"


def test_parse_tool_args_truncated_object_returns_sentinel():
    """Reálný production bug: model uřízl JSON uprostřed (max_tokens)."""
    raw = '{"command": "rm -rf'
    tc = {"function": {"name": "run_bash", "arguments": raw}}
    out = parse_tool_args(tc)
    assert out["_parse_error"] == "invalid_json"
    # forensic: raw_length + raw_hash zachytí stopu BEZ leaknutí command stringu
    assert out["raw_length"] == len(raw)
    assert len(out["raw_hash"]) == 64
    assert "raw_preview" not in out  # secret leak prevention


def test_canonicalize_arguments_dict_returns_json_string():
    assert canonicalize_arguments({"x": 1}) == '{"x": 1}'


def test_canonicalize_arguments_valid_json_string_reserialized():
    out = canonicalize_arguments('  {"x":1}  ')
    assert json.loads(out) == {"x": 1}


def test_canonicalize_arguments_empty_string_returns_empty_object():
    assert canonicalize_arguments("") == "{}"
    assert canonicalize_arguments("   ") == "{}"


def test_canonicalize_arguments_none_returns_empty_object():
    assert canonicalize_arguments(None) == "{}"


def test_canonicalize_arguments_invalid_json_returns_sentinel():
    raw = '{"path": "/tmp/x'
    out = canonicalize_arguments(raw)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "invalid_json"
    assert decoded["raw_length"] == len(raw)
    assert "raw_preview" not in decoded


def test_canonicalize_arguments_non_object_json_returns_sentinel():
    out = canonicalize_arguments("[1,2,3]")
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "non_object_json"


def test_canonicalize_arguments_non_string_non_dict_returns_sentinel():
    """Codex review HIGH: `arguments: []` / `0` / `False` nesmí projít na `{}`.

    Bez tohoto fixu by mohl adversarial klient v history posunout
    `arguments: []` → canonicalize na "{}" → tool dostane defaults → bypass.
    """
    for bad in ([1, 2], 42, True, False, [], {1: 2} if False else None):
        # None je legitimní (model neposlal args) → skip
        if bad is None:
            continue
        out = canonicalize_arguments(bad)
        decoded = json.loads(out)
        assert decoded["_parse_error"] == "invalid_type", f"failed for {bad!r}: {decoded}"
        assert "raw_hash" in decoded


def test_canonicalize_arguments_too_large_returns_sentinel():
    """DoS guard: arguments > 64 KiB → sentinel `too_large`, neproparseuje."""
    huge = '{"x": "' + "A" * (70 * 1024) + '"}'  # ~70 KiB valid JSON
    out = canonicalize_arguments(huge)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "too_large"
    assert decoded["raw_length"] > 64 * 1024


def test_canonicalize_arguments_dict_too_large_does_not_bypass_cap():
    """Codex iter-2 HIGH: dict path nesmí bypassovat 64 KiB cap. Velký dict
    musí dopadnout na sentinel `too_large` (nikoli projít rovnou do history).
    """
    huge_dict = {"x": "A" * (70 * 1024)}
    out = canonicalize_arguments(huge_dict)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "too_large"


def test_canonicalize_arguments_dict_non_serializable_does_not_crash():
    """Codex iter-2 HIGH: dict s non-JSON-serializable hodnotou (cyklus,
    object, Path …) nesmí vyhodit TypeError z json.dumps — sentinel místo crashe.
    """
    class _Custom:
        pass
    bad = {"k": _Custom()}
    out = canonicalize_arguments(bad)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "invalid_type"


def test_canonicalize_arguments_dict_with_cycle_does_not_crash():
    cycle: dict = {}
    cycle["self"] = cycle
    out = canonicalize_arguments(cycle)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "invalid_type"


def test_parse_tool_args_dict_too_large_returns_sentinel():
    """Symmetric s canonicalize: parse_tool_args musí dict cap respektovat."""
    huge = {"x": "A" * (70 * 1024)}
    tc = {"function": {"name": "x", "arguments": huge}}
    out = parse_tool_args(tc)
    assert out["_parse_error"] == "too_large"


def test_parse_tool_args_dict_non_serializable_returns_sentinel():
    class _Custom:
        pass
    tc = {"function": {"name": "x", "arguments": {"k": _Custom()}}}
    out = parse_tool_args(tc)
    assert out["_parse_error"] == "invalid_type"


def test_parse_tool_args_function_non_dict_does_not_crash():
    """Codex iter-3 HIGH: `tool_call.function` může být truthy non-dict
    (`"x"`, `[1]`, `123`). `.get(...)` by spadl AttributeError.
    Po fixu: vrátí `{}` (= žádné args, schema validation potom zachytí)."""
    for bad_fn in ("not a dict", [1, 2], 123, True):
        tc = {"function": bad_fn}
        out = parse_tool_args(tc)
        assert out == {}, f"failed for function={bad_fn!r}: {out}"


def test_normalize_tool_calls_function_non_dict_does_not_crash():
    """Codex iter-3 HIGH: stejné jako parse_tool_args ale v normalize_tool_calls.
    Tool call bez validního dict pro `function` musí být DROPNUTÝ, ne crash."""
    raw = [
        {"id": "tc_1", "function": "not a dict"},
        {"id": "tc_2", "function": [1, 2, 3]},
        {"id": "tc_3", "function": 42},
        {"id": "tc_ok", "function": {"name": "echo", "arguments": {"text": "hi"}}},
    ]
    out = normalize_tool_calls(raw)
    assert len(out) == 1
    assert out[0]["id"] == "tc_ok"


def test_sentinel_creation_does_not_crash_on_recursive_dict():
    """Codex+Gemini iter-3 HIGH: _make_parse_error_sentinel musí být
    uncaught-free i pro cyclic dict (RecursionError z json.dumps)."""
    cycle: dict = {}
    cycle["self"] = cycle
    # Direct call přes canonicalize → sentinel fallback → musí projít
    out = canonicalize_arguments(cycle)
    decoded = json.loads(out)
    assert decoded["_parse_error"] == "invalid_type"
    # Hash by měl mít validní hex (≥ 0 chars)
    assert isinstance(decoded["raw_hash"], str)
    assert len(decoded["raw_hash"]) == 64


def test_canonicalize_arguments_hash_includes_original_whitespace():
    """raw_hash MUSÍ být z původního před-strip stringu — útočník nesmí
    cestou whitespace prefixu vytvořit "různý" hash pro stejný payload."""
    a = canonicalize_arguments("not json")
    b = canonicalize_arguments("   not json   ")
    da, db = json.loads(a), json.loads(b)
    # Oba sentinely, ale hashe se LIŠÍ (každý hash z exact původního raw).
    assert da["_parse_error"] == "invalid_json"
    assert db["_parse_error"] == "invalid_json"
    assert da["raw_hash"] != db["raw_hash"]


def test_is_malformed_args_true_for_sentinel():
    sentinel = canonicalize_arguments("not json")
    args = json.loads(sentinel)
    assert is_malformed_args(args) is True


def test_is_malformed_args_false_for_normal_dict():
    assert is_malformed_args({"path": "/tmp"}) is False


def test_is_malformed_args_false_for_non_dict():
    assert is_malformed_args(None) is False
    assert is_malformed_args("string") is False
    assert is_malformed_args([1, 2]) is False


def test_normalize_tool_calls_with_malformed_args_produces_sentinel():
    """Bug repro: model emituje truncated JSON v arguments.

    Před fixem: arguments=raw_string → Ollama next round HTTP 400.
    Po fixu: arguments=sentinel JSON object → Ollama parses OK,
    loop detekuje a NESPUSTÍ tool.
    """
    raw_args = '{"command": "rm /tmp'
    raw = [{"id": "tc_1", "function": {"name": "run_bash",
                                       "arguments": raw_args}}]
    out = normalize_tool_calls(raw)
    assert len(out) == 1
    args_str = out[0]["function"]["arguments"]
    decoded = json.loads(args_str)  # MUSÍ být valid JSON
    assert decoded["_parse_error"] == "invalid_json"
    assert decoded["raw_length"] == len(raw_args)
    # Žádný raw_preview → command string ("rm /tmp") NESMÍ leaknout
    assert "raw_preview" not in decoded


def test_normalize_tool_calls_dedupes_by_id():
    """Ollama může streamovat stejný tool call víckrát (chunked / retry).
    Bez dedupy by se tool spustil opakovaně — fatální pro destructive."""
    raw = [
        {"id": "tc_1", "function": {"name": "echo", "arguments": '{"text":"a"}'}},
        {"id": "tc_1", "function": {"name": "echo", "arguments": '{"text":"a"}'}},
        {"id": "tc_2", "function": {"name": "echo", "arguments": '{"text":"b"}'}},
    ]
    out = normalize_tool_calls(raw)
    assert len(out) == 2
    assert [tc["id"] for tc in out] == ["tc_1", "tc_2"]


def test_normalize_tool_calls_drops_unnamed():
    """Tool call bez `function.name` musí být zahozený — jinak by agent
    zkusil vykonat něco s prázdným jménem."""
    raw = [
        {"id": "tc_1", "function": {"name": "", "arguments": "{}"}},
        {"id": "tc_2", "function": {"arguments": "{}"}},  # no name key
        {"id": "tc_3", "function": {"name": "echo", "arguments": "{}"}},
    ]
    out = normalize_tool_calls(raw)
    assert len(out) == 1
    assert out[0]["function"]["name"] == "echo"


def test_normalize_tool_calls_drops_non_dict_items():
    raw = ["not a dict", None, 42, {"id": "tc_1", "function": {"name": "echo"}}]
    out = normalize_tool_calls(raw)
    assert len(out) == 1
    assert out[0]["id"] == "tc_1"
