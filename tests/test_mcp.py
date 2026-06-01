"""Unit testy pro MCP klient + tool wrapper.

Integration testy proti živému HOTOVO `node server/mcp.js` jsou označené
@pytest.mark.integration a vyžadují, aby todo-list backend běžel na :3000.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from voice.agent.mcp import McpClient, McpError, McpServerConfig, health_probe
from voice.agent.tools.mcp_tool import discover_and_register


# ──────────────── Mock MCP server (Python stdio) ────────────────

_MOCK_SERVER_PY = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Minimal MCP server pro testy. JSON-RPC 2.0 stdio.\"\"\"
    import json, sys

    TOOLS = [
        {
            "name": "ping",
            "description": "Vrátí pong",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "echo",
            "description": "Echo text",
            "inputSchema": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
        {
            "name": "fail",
            "description": "Vždy vrátí isError",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock", "version": "0.1"},
            }})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "ping":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps({"reply": "pong"})}],
                }})
            elif name == "echo":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps({"echoed": args.get("text")})}],
                }})
            elif name == "fail":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "Chyba: něco se rozbilo"}],
                    "isError": True,
                }})
            else:
                send({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32602, "message": f"unknown tool: {name}",
                }})
        else:
            send({"jsonrpc": "2.0", "id": mid, "error": {
                "code": -32601, "message": f"method not found: {method}",
            }})
""")


@pytest.fixture
def mock_server_script(tmp_path):
    script = tmp_path / "mock_mcp.py"
    script.write_text(_MOCK_SERVER_PY)
    script.chmod(0o755)
    return script


def _config(script: Path, **overrides) -> McpServerConfig:
    return McpServerConfig(
        name="mock",
        command=(sys.executable, str(script)),
        idle_timeout_sec=None,  # disable for predictable tests
        request_timeout_sec=overrides.get("request_timeout_sec", 5.0),
        health_probe_url=overrides.get("health_probe_url"),
    )


# ──────────────── Tests ────────────────

@pytest.mark.asyncio
async def test_list_tools_returns_definitions(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        tools = await client.list_tools()
        names = [t["name"] for t in tools]
        assert "ping" in names
        assert "echo" in names
        assert "fail" in names
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_list_tools_cached(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        first = await client.list_tools()
        second = await client.list_tools()
        assert first is second  # identita = cached
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_call_tool_returns_content(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        await client.list_tools()
        result = await client.call_tool("ping", {})
        assert result["content"][0]["text"] == json.dumps({"reply": "pong"})
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_call_tool_with_args(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        await client.list_tools()
        result = await client.call_tool("echo", {"text": "hello"})
        text = result["content"][0]["text"]
        assert json.loads(text) == {"echoed": "hello"}
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_call_tool_isError_surfaces(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        await client.list_tools()
        result = await client.call_tool("fail", {})
        assert result.get("isError") is True
        assert "Chyba" in result["content"][0]["text"]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_unknown_tool_raises_mcperror(mock_server_script):
    client = McpClient(_config(mock_server_script))
    try:
        await client.list_tools()
        with pytest.raises(McpError, match="unknown tool"):
            await client.call_tool("ghost", {})
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_subprocess_not_found_raises(tmp_path):
    config = McpServerConfig(
        name="missing",
        command=("/nonexistent/binary-xyz", "arg"),
        idle_timeout_sec=None,
    )
    client = McpClient(config)
    try:
        with pytest.raises(McpError, match="binary not found"):
            await client.list_tools()
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_env_scrub_drops_secrets(mock_server_script, monkeypatch):
    # Skript pošle dump env do stderr — pro tento test stačí ověřit, že
    # FAKE_SECRET není v built env subprocess.
    monkeypatch.setenv("FAKE_SECRET_API_KEY", "super-secret-123")
    config = McpServerConfig(
        name="mock",
        command=(sys.executable, str(mock_server_script)),
        idle_timeout_sec=None,
    )
    client = McpClient(config)
    try:
        # Use internal _build_env (jednoduchý ověřovací path)
        env = client._build_env()
        assert "FAKE_SECRET_API_KEY" not in env, "secret nesmí být v scrubbed env"
        assert "HOME" in env, "allowlist HOME musí projít"
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_discover_and_register_creates_gemma_tools(mock_server_script):
    config = _config(mock_server_script)
    client, tools = await discover_and_register(config)
    try:
        assert client is not None
        names = [t.name for t in tools]
        # Tool names jsou prefixované server jménem ("mock_ping", …)
        assert "mock_ping" in names
        assert "mock_echo" in names
        assert "mock_fail" in names
    finally:
        if client is not None:
            await client.shutdown()


@pytest.mark.asyncio
async def test_gemma_tool_execute_parses_json_response(mock_server_script):
    config = _config(mock_server_script)
    client, tools = await discover_and_register(config)
    try:
        ping_tool = next(t for t in tools if t.name == "mock_ping")
        from voice.agent.tools.base import ExecuteContext
        ctx = ExecuteContext(turn_id="t", cancel_event=None, workdir=Path("/tmp"))
        result = await ping_tool.execute({}, ctx)
        assert result["ok"] is True
        assert result["result"] == {"reply": "pong"}
    finally:
        if client is not None:
            await client.shutdown()


@pytest.mark.asyncio
async def test_gemma_tool_execute_surfaces_is_error(mock_server_script):
    config = _config(mock_server_script)
    client, tools = await discover_and_register(config)
    try:
        fail_tool = next(t for t in tools if t.name == "mock_fail")
        from voice.agent.tools.base import ExecuteContext
        ctx = ExecuteContext(turn_id="t", cancel_event=None, workdir=Path("/tmp"))
        result = await fail_tool.execute({}, ctx)
        assert result["ok"] is False
        assert "Chyba" in result["error"]
    finally:
        if client is not None:
            await client.shutdown()


@pytest.mark.asyncio
async def test_health_probe_no_url_returns_true():
    config = McpServerConfig(name="x", command=("true",))
    assert await health_probe(config) is True


@pytest.mark.asyncio
async def test_health_probe_unreachable_returns_false():
    config = McpServerConfig(
        name="x", command=("true",),
        health_probe_url="http://127.0.0.1:1/never",
        health_probe_timeout_sec=0.5,
    )
    assert await health_probe(config) is False


@pytest.mark.asyncio
async def test_discover_skips_when_health_probe_fails():
    config = McpServerConfig(
        name="dead",
        command=(sys.executable, "-c", "import sys; sys.exit(0)"),
        health_probe_url="http://127.0.0.1:1/dead",
        health_probe_timeout_sec=0.3,
    )
    client, tools = await discover_and_register(config)
    assert client is None
    assert tools == []


# ──────────────── Permission classifier ────────────────


def test_mcp_classifier_auto_for_read_only():
    from voice.agent.permissions import (
        Decision, _CLASSIFIERS, register_mcp_classifier, decide,
    )
    register_mcp_classifier(
        gemma_tool_name="test_get_state",
        mcp_tool_name="get_state",
        server_name="test",
        auto=True,
        requires_explicit=False,
    )
    try:
        result = decide("test_get_state", {}, Path("/tmp"))
        assert result.decision == Decision.AUTO
        assert result.risk == "low"
    finally:
        _CLASSIFIERS.pop("test_get_state", None)


def test_mcp_classifier_destructive_requires_explicit():
    from voice.agent.permissions import (
        Decision, _CLASSIFIERS, register_mcp_classifier, decide,
    )
    register_mcp_classifier(
        gemma_tool_name="test_delete_task",
        mcp_tool_name="delete_task",
        server_name="test",
        auto=False,
        requires_explicit=True,
    )
    try:
        result = decide("test_delete_task", {"id": "x"}, Path("/tmp"))
        assert result.decision == Decision.ASK
        assert result.requires_explicit is True
        assert result.risk == "destructive"
    finally:
        _CLASSIFIERS.pop("test_delete_task", None)


def test_mcp_classifier_default_mutation_is_ask_medium():
    from voice.agent.permissions import (
        Decision, _CLASSIFIERS, register_mcp_classifier, decide,
    )
    register_mcp_classifier(
        gemma_tool_name="test_create_task",
        mcp_tool_name="create_task",
        server_name="test",
        auto=False,
        requires_explicit=False,
    )
    try:
        result = decide("test_create_task", {}, Path("/tmp"))
        assert result.decision == Decision.ASK
        assert result.risk == "medium"
        assert result.requires_explicit is False
    finally:
        _CLASSIFIERS.pop("test_create_task", None)


# ──────────────── Config loader ────────────────


def test_hotovo_config_skipped_when_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_HOTOVO_MCP_PATH", str(tmp_path / "nope.js"))
    # Re-import config aby vzal nový env
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    configs = agent_config.get_mcp_server_configs()
    assert configs == []


def test_hotovo_config_loaded_when_path_exists(monkeypatch, tmp_path):
    fake_mcp = tmp_path / "mcp.js"
    fake_mcp.write_text("// stub")
    monkeypatch.setenv("AGENT_HOTOVO_MCP_PATH", str(fake_mcp))
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    configs = agent_config.get_mcp_server_configs()
    assert len(configs) == 1
    assert configs[0].name == "hotovo"
    assert "get_state" in configs[0].auto_tools
    assert "delete_task" in configs[0].requires_explicit_tools


def test_hotovo_token_file_world_readable_rejected(monkeypatch, tmp_path):
    fake_mcp = tmp_path / "mcp.js"
    fake_mcp.write_text("// stub")
    token_file = tmp_path / "token"
    token_file.write_text("MY-TOKEN")
    token_file.chmod(0o644)  # world readable — must be ignored
    monkeypatch.setenv("AGENT_HOTOVO_MCP_PATH", str(fake_mcp))
    monkeypatch.setenv("HOTOVO_API_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("HOTOVO_API_TOKEN", raising=False)
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    configs = agent_config.get_mcp_server_configs()
    assert configs[0].env.get("HOTOVO_API_TOKEN") is None  # rejected


def test_hotovo_token_file_secure_perms_loaded(monkeypatch, tmp_path):
    fake_mcp = tmp_path / "mcp.js"
    fake_mcp.write_text("// stub")
    token_file = tmp_path / "token"
    token_file.write_text("MY-TOKEN-123")
    token_file.chmod(0o600)
    monkeypatch.setenv("AGENT_HOTOVO_MCP_PATH", str(fake_mcp))
    monkeypatch.setenv("HOTOVO_API_TOKEN_FILE", str(token_file))
    monkeypatch.delenv("HOTOVO_API_TOKEN", raising=False)
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    configs = agent_config.get_mcp_server_configs()
    assert configs[0].env.get("HOTOVO_API_TOKEN") == "MY-TOKEN-123"
