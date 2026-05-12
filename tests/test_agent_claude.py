"""Unit testy pro Claude bridge (voice/agent/tools/claude.py).

Mock httpx přes httpx.MockTransport — žádná real Anthropic API. DNS resolve
mockujeme pro SSRF guard backend (api.anthropic.com → veřejná IP).
"""
from __future__ import annotations

import asyncio
import gzip
import json
import socket
from pathlib import Path

import httpx
import pytest

from voice.agent.tools.base import ExecuteContext
from voice.agent.tools import claude as claude_mod
from voice.agent.tools.claude import (
    ASK_CLAUDE_TOOL,
    _ask_claude_exec,
    _extract_text,
)


def _ctx(workdir: Path, *, cancel_event: asyncio.Event | None = None) -> ExecuteContext:
    return ExecuteContext(turn_id="t1", cancel_event=cancel_event, workdir=workdir)


def _patch_dns(monkeypatch, addrs: list[str]):
    async def fake(self, host, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (a, 0)) for a in addrs
        ]

    from asyncio.unix_events import _UnixSelectorEventLoop  # type: ignore[attr-defined]
    monkeypatch.setattr(_UnixSelectorEventLoop, "getaddrinfo", fake)


def _install_mock_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        return original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


# ---------------------------------------------------------------------------
# _extract_text helper
# ---------------------------------------------------------------------------


def test_extract_text_concatenates_blocks():
    payload = {
        "content": [
            {"type": "text", "text": "hello "},
            {"type": "text", "text": "world"},
        ]
    }
    assert _extract_text(payload) == "hello world"


def test_extract_text_skips_non_text_blocks():
    payload = {
        "content": [
            {"type": "tool_use", "name": "foo"},
            {"type": "text", "text": "only this"},
        ]
    }
    assert _extract_text(payload) == "only this"


def test_extract_text_missing_content():
    assert _extract_text({}) == ""
    assert _extract_text({"content": "not a list"}) == ""
    assert _extract_text({"content": [None, 123, {"type": "text"}]}) == ""


# ---------------------------------------------------------------------------
# Tool schema metadata
# ---------------------------------------------------------------------------


def test_tool_metadata():
    assert ASK_CLAUDE_TOOL.name == "ask_claude"
    schema = ASK_CLAUDE_TOOL.parameters_schema
    assert schema["required"] == ["prompt"]
    assert schema["properties"]["prompt"]["type"] == "string"
    assert schema["properties"]["max_tokens"]["type"] == "integer"
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_ask_claude_happy_path(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=json.dumps({
                "id": "msg_1",
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Hi there!"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }).encode("utf-8"),
        )

    _install_mock_transport(monkeypatch, handler)

    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hello"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True, r
    assert r["text"] == "Hi there!"
    assert r["stop_reason"] == "end_turn"
    assert r["input_tokens"] == 10
    assert r["output_tokens"] == 5
    assert captured["body"]["model"] == "claude-opus-4-7"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Hello"}]
    assert captured["body"]["max_tokens"] == 1024  # default
    assert "system" not in captured["body"]
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["headers"]["accept-encoding"] == "identity"


def test_ask_claude_with_system_and_max_tokens(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b'{"content":[{"type":"text","text":"ok"}]}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Q", "system": "You are a pirate.", "max_tokens": 100},
        _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert captured["body"]["system"] == "You are a pirate."
    assert captured["body"]["max_tokens"] == 100


# ---------------------------------------------------------------------------
# Missing / bad API key
# ---------------------------------------------------------------------------


def test_ask_claude_no_key(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "")
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "ANTHROPIC_API_KEY" in r["error"]


def test_ask_claude_auth_error(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-bad")

    def handler(request):
        return httpx.Response(
            401,
            headers={"Content-Type": "application/json"},
            content=b'{"error":{"message":"invalid x-api-key"}}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert r["status"] == 401
    assert "auth error" in r["error"]


def test_ask_claude_rate_limit(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        return httpx.Response(429, content=b'{"error":{"message":"slow down"}}')

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert r["status"] == 429
    assert "rate-limited" in r["error"]


def test_ask_claude_api_error_400(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        return httpx.Response(
            400,
            headers={"Content-Type": "application/json"},
            content=b'{"error":{"message":"max_tokens must be > 0"}}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert r["status"] == 400
    assert "max_tokens must be > 0" in r["error"]


def test_ask_claude_api_error_message_truncated(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    huge = "x" * 5000

    def handler(request):
        return httpx.Response(
            400,
            headers={"Content-Type": "application/json"},
            content=json.dumps({"error": {"message": huge}}).encode("utf-8"),
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    # error string is truncated to <=400 chars message body
    assert len(r["error"]) < 500


def test_ask_claude_api_error_without_json_body(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        return httpx.Response(500, content=b"<html>boom</html>")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert r["status"] == 500
    assert "HTTP 500" in r["error"]


# ---------------------------------------------------------------------------
# Invalid input (defense in depth in execute)
# ---------------------------------------------------------------------------


def test_ask_claude_empty_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec({"prompt": "   "}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "empty prompt" in r["error"]


def test_ask_claude_non_string_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec({"prompt": 123}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "must be string" in r["error"]


def test_ask_claude_oversized_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(claude_mod, "CLAUDE_MAX_PROMPT_BYTES", 16)
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "x" * 100}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "too large" in r["error"]


def test_ask_claude_invalid_max_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hi", "max_tokens": "five"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "max_tokens" in r["error"]


def test_ask_claude_max_tokens_bool_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hi", "max_tokens": True}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "bool" in r["error"]


def test_ask_claude_max_tokens_out_of_range(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hi", "max_tokens": 999999}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "out of range" in r["error"]


def test_ask_claude_system_non_string(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hi", "system": 42}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "system" in r["error"]


def test_ask_claude_system_oversized(tmp_path, monkeypatch):
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(claude_mod, "CLAUDE_MAX_SYSTEM_BYTES", 8)
    r = asyncio.run(_ask_claude_exec(
        {"prompt": "Hi", "system": "x" * 100}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "system too large" in r["error"]


# ---------------------------------------------------------------------------
# Network-level errors
# ---------------------------------------------------------------------------


def test_ask_claude_timeout(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        raise httpx.ReadTimeout("read timeout")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "timeout" in r["error"]


def test_ask_claude_request_error(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        raise httpx.ConnectError("boom")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "request error" in r["error"]


# ---------------------------------------------------------------------------
# Response capping + compression defense
# ---------------------------------------------------------------------------


def test_ask_claude_response_truncated_at_cap(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(claude_mod, "CLAUDE_OUTPUT_CAP_BYTES", 64)

    # Posíláme víc než 64 B
    big_body = json.dumps({
        "content": [{"type": "text", "text": "x" * 1000}]
    }).encode("utf-8")

    def handler(request):
        return httpx.Response(
            200,
            # Žádný content-length → padá do per-chunk capu
            headers={"Content-Type": "application/json"},
            content=big_body,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "truncated" in r["error"] or "too large" in r["error"]


def test_ask_claude_response_content_length_precheck(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(claude_mod, "CLAUDE_OUTPUT_CAP_BYTES", 64)

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json", "Content-Length": "10000"},
            content=b'{"content":[]}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "too large" in r["error"]


def test_ask_claude_rejects_gzip_response(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    valid_gzipped = gzip.compress(b'{"content":[]}')

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
            content=valid_gzipped,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "content-encoding" in r["error"].lower() or "encoding" in r["error"].lower()


def test_ask_claude_accepts_identity_encoding(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json", "Content-Encoding": "identity"},
            content=b'{"content":[{"type":"text","text":"ok"}]}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is True
    assert r["text"] == "ok"


def test_ask_claude_invalid_json_response(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", "sk-ant-test")

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b"not-json",
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "invalid JSON" in r["error"]


# ---------------------------------------------------------------------------
# Key leak prevention
# ---------------------------------------------------------------------------


def test_no_key_in_error_messages(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])
    secret = "sk-ant-supersecret-XYZ-1234567890"
    monkeypatch.setattr(claude_mod, "ANTHROPIC_API_KEY", secret)

    def handler(request):
        return httpx.Response(
            401,
            content=b'{"error":{"message":"unauthorized"}}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_ask_claude_exec({"prompt": "Hi"}, _ctx(tmp_path)))
    assert r["ok"] is False
    full = json.dumps(r)
    assert secret not in full
