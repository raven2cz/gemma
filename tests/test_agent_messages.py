"""Test conversation history schema (voice/agent/messages.py)."""
from __future__ import annotations

import json

from voice.agent.messages import (
    assistant_message,
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


def test_parse_tool_args_invalid_json():
    tc = {"function": {"name": "x", "arguments": "not json"}}
    assert parse_tool_args(tc) == {}


def test_parse_tool_args_non_dict_json():
    tc = {"function": {"name": "x", "arguments": "[1,2,3]"}}
    assert parse_tool_args(tc) == {}


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
