"""HOTOVO REST tool tests. Mockuje HTTP server přes pytest-httpx jako
respx není installed; používáme httpx.MockTransport.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from voice.agent.tools.base import ExecuteContext
from voice.agent.tools.hotovo import build_tools


def _ctx() -> ExecuteContext:
    return ExecuteContext(turn_id="t", cancel_event=None, workdir=Path("/tmp"))


def _make_mock_tools(handler):
    """Vyrobí 8 tooly s mock httpx transportem. Handler dostane request,
    vrátí (status, body)."""
    captured: list = []

    def _wrapped(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        status, body = handler(request)
        return httpx.Response(status, json=body) if isinstance(body, dict) \
            else httpx.Response(status, content=body)

    transport = httpx.MockTransport(_wrapped)

    # Monkey-patch httpx.AsyncClient to inject our transport.
    import voice.agent.tools.hotovo as hotovo_mod
    original = hotovo_mod.httpx.AsyncClient

    class _Client(original):
        def __init__(self, **kw):
            kw["transport"] = transport
            super().__init__(**kw)

    hotovo_mod.httpx.AsyncClient = _Client

    def _restore():
        hotovo_mod.httpx.AsyncClient = original

    tools = build_tools(
        base_url_provider=lambda: "https://test.local",
        token_provider=lambda: "TEST-TOKEN",
        timeout_provider=lambda: 5.0,
    )
    return tools, captured, _restore


# ──────────────── Tool registry sanity ────────────────


def test_build_tools_returns_8_with_correct_names():
    tools = build_tools(
        base_url_provider=lambda: "x",
        token_provider=lambda: "",
        timeout_provider=lambda: 1.0,
    )
    names = {t.name for t in tools}
    assert names == {
        "hotovo_get_state", "hotovo_list_projects", "hotovo_create_project",
        "hotovo_list_tasks", "hotovo_create_task", "hotovo_update_task",
        "hotovo_complete_task", "hotovo_delete_task",
    }


def test_default_registry_includes_hotovo_when_api_url_set(monkeypatch):
    monkeypatch.setenv("HOTOVO_API_URL", "https://test.local")
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    from voice.agent.tools import default_registry
    reg = default_registry()
    names = {t.name for t in reg.all()}
    assert "hotovo_get_state" in names
    assert "hotovo_delete_task" in names


def test_default_registry_excludes_hotovo_when_url_missing(monkeypatch):
    monkeypatch.delenv("HOTOVO_API_URL", raising=False)
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    from voice.agent.tools import default_registry
    reg = default_registry()
    names = {t.name for t in reg.all()}
    assert not any(n.startswith("hotovo_") for n in names)


# ──────────────── REST happy paths (mocked HTTP) ────────────────


@pytest.mark.asyncio
async def test_get_state_returns_json_result():
    payload = {"counts": {"lists": 2, "tasks": 5},
               "lists": [{"id": "x", "name": "Osobni"}]}
    tools, captured, restore = _make_mock_tools(lambda r: (200, payload))
    try:
        get_state = next(t for t in tools if t.name == "hotovo_get_state")
        result = await get_state.execute({}, _ctx())
        assert result["ok"] is True
        assert result["result"] == payload
        assert captured[0].method == "GET"
        assert str(captured[0].url).endswith("/api/agent/state")
        assert captured[0].headers["authorization"] == "Bearer TEST-TOKEN"
    finally:
        restore()


@pytest.mark.asyncio
async def test_create_task_sends_body():
    tools, captured, restore = _make_mock_tools(
        lambda r: (201, {"id": "new-id", "title": "koupit chleba"})
    )
    try:
        create = next(t for t in tools if t.name == "hotovo_create_task")
        result = await create.execute({
            "title": "koupit chleba",
            "list_id": "list-xyz",
            "priority": "high",
            "due_date": "2026-06-02",
            "tags": ["nákup"],
        }, _ctx())
        assert result["ok"] is True
        assert captured[0].method == "POST"
        body = json.loads(captured[0].content)
        assert body["title"] == "koupit chleba"
        assert body["list_id"] == "list-xyz"
        assert body["priority"] == "high"
        assert body["due_date"] == "2026-06-02"
        assert body["tags"] == ["nákup"]
    finally:
        restore()


@pytest.mark.asyncio
async def test_create_task_drops_none_keys():
    """args.priority=None se NEMÁ poslat (server defaultuje sám)."""
    tools, captured, restore = _make_mock_tools(lambda r: (201, {"id": "x"}))
    try:
        create = next(t for t in tools if t.name == "hotovo_create_task")
        await create.execute({
            "title": "x", "list_id": "y",
            "priority": None, "description": None,
        }, _ctx())
        body = json.loads(captured[0].content)
        assert "priority" not in body
        assert "description" not in body
        assert body["title"] == "x"
    finally:
        restore()


@pytest.mark.asyncio
async def test_list_tasks_with_filters_uses_query_string():
    tools, captured, restore = _make_mock_tools(lambda r: (200, []))
    try:
        list_tasks = next(t for t in tools if t.name == "hotovo_list_tasks")
        await list_tasks.execute({
            "list_id": "abc",
            "status": "pending",
            "search": "chleba",
        }, _ctx())
        url = str(captured[0].url)
        assert "list_id=abc" in url
        assert "status=pending" in url
        assert "search=chleba" in url
    finally:
        restore()


@pytest.mark.asyncio
async def test_update_task_uses_path_param():
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"id": "task-xyz"}))
    try:
        update = next(t for t in tools if t.name == "hotovo_update_task")
        result = await update.execute({
            "id": "task-xyz",
            "title": "nový název",
        }, _ctx())
        assert result["ok"] is True
        assert captured[0].method == "PUT"
        assert "/api/tasks/task-xyz" in str(captured[0].url)
        body = json.loads(captured[0].content)
        assert body == {"title": "nový název"}
    finally:
        restore()


@pytest.mark.asyncio
async def test_update_task_url_encodes_id_for_traversal_safety():
    tools, captured, restore = _make_mock_tools(lambda r: (200, {}))
    try:
        update = next(t for t in tools if t.name == "hotovo_update_task")
        await update.execute({"id": "../../etc/passwd", "title": "x"}, _ctx())
        url = str(captured[0].url)
        # Slash MUSÍ být zakódovaný (chrání před path traversal)
        assert "/api/tasks/..%2F..%2Fetc%2Fpasswd" in url
    finally:
        restore()


@pytest.mark.asyncio
async def test_complete_task_sends_status_completed_body():
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"status": "completed"}))
    try:
        complete = next(t for t in tools if t.name == "hotovo_complete_task")
        result = await complete.execute({"id": "abc"}, _ctx())
        assert result["ok"] is True
        body = json.loads(captured[0].content)
        assert body == {"status": "completed"}
        assert "/api/tasks/abc" in str(captured[0].url)
    finally:
        restore()


@pytest.mark.asyncio
async def test_delete_task_passes_confirm_query():
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"ok": True}))
    try:
        delete = next(t for t in tools if t.name == "hotovo_delete_task")
        await delete.execute({"id": "abc", "confirm": True}, _ctx())
        url = str(captured[0].url)
        assert captured[0].method == "DELETE"
        assert "/api/tasks/abc" in url
        assert "confirm=" in url
    finally:
        restore()


# ──────────────── Error paths ────────────────


@pytest.mark.asyncio
async def test_missing_api_url_returns_clear_error():
    tools = build_tools(
        base_url_provider=lambda: "",  # missing!
        token_provider=lambda: "x",
        timeout_provider=lambda: 1.0,
    )
    result = await tools[0].execute({}, _ctx())
    assert result["ok"] is False
    assert "HOTOVO_API_URL" in result["error"]


@pytest.mark.asyncio
async def test_missing_id_returns_clear_error():
    tools = build_tools(
        base_url_provider=lambda: "https://test.local",
        token_provider=lambda: "x",
        timeout_provider=lambda: 1.0,
    )
    update = next(t for t in tools if t.name == "hotovo_update_task")
    result = await update.execute({"title": "x"}, _ctx())  # no id
    assert result["ok"] is False
    assert "'id'" in result["error"]


@pytest.mark.asyncio
async def test_server_error_surfaces_message():
    tools, _captured, restore = _make_mock_tools(
        lambda r: (500, {"error": "Database is locked"})
    )
    try:
        get_state = next(t for t in tools if t.name == "hotovo_get_state")
        result = await get_state.execute({}, _ctx())
        assert result["ok"] is False
        assert result["error"] == "Database is locked"
        assert result["status"] == 500
    finally:
        restore()


@pytest.mark.asyncio
async def test_unauthorized_surfaces_message():
    tools, _, restore = _make_mock_tools(
        lambda r: (401, {"error": "Chybí API token."})
    )
    try:
        result = await tools[0].execute({}, _ctx())
        assert result["ok"] is False
        assert "API token" in result["error"]
    finally:
        restore()


# ──────────────── Codex review fixes ────────────────


@pytest.mark.asyncio
async def test_http_remote_url_rejected_to_protect_token():
    """Codex HIGH #1: http:// na remote host by poslal Bearer token v plaintextu."""
    tools = build_tools(
        base_url_provider=lambda: "http://fishlive.org:17854",  # http + remote!
        token_provider=lambda: "secret-token",
        timeout_provider=lambda: 1.0,
    )
    result = await tools[0].execute({}, _ctx())
    assert result["ok"] is False
    assert "plaintext" in result["error"].lower() or "https" in result["error"].lower()


@pytest.mark.asyncio
async def test_http_loopback_url_allowed():
    """http:// na localhost je OK (token neopustí stroj)."""
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"ok": True}))
    try:
        # Override base_url provider na http loopback
        import voice.agent.tools.hotovo as hm
        local_tools = build_tools(
            base_url_provider=lambda: "http://127.0.0.1:3000",
            token_provider=lambda: "TEST-TOKEN",
            timeout_provider=lambda: 5.0,
        )
        result = await local_tools[0].execute({}, _ctx())
        # Mock transport vrací 200 → projde scheme validací
        assert result["ok"] is True
    finally:
        restore()


@pytest.mark.asyncio
async def test_https_url_accepted():
    tools, _, restore = _make_mock_tools(lambda r: (200, {"ok": True}))
    try:
        result = await tools[0].execute({}, _ctx())  # base_url = https://test.local
        assert result["ok"] is True
    finally:
        restore()


@pytest.mark.asyncio
async def test_update_task_sends_explicit_null_due_date():
    """Codex HIGH #2: due_date:null musí JÍT do body (= smaž termín),
    ne se dropnout."""
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"id": "x"}))
    try:
        update = next(t for t in tools if t.name == "hotovo_update_task")
        await update.execute({"id": "task-1", "due_date": None}, _ctx())
        body = json.loads(captured[0].content)
        assert "due_date" in body
        assert body["due_date"] is None
    finally:
        restore()


@pytest.mark.asyncio
async def test_update_task_sends_explicit_null_parent_id():
    """parent_id:null = odpoj od rodiče → musí JÍT do body."""
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"id": "x"}))
    try:
        update = next(t for t in tools if t.name == "hotovo_update_task")
        await update.execute({"id": "task-1", "parent_id": None}, _ctx())
        body = json.loads(captured[0].content)
        assert "parent_id" in body
        assert body["parent_id"] is None
    finally:
        restore()


@pytest.mark.asyncio
async def test_create_task_still_drops_null():
    """create_task NEMÁ nullable keys → None se dál dropuje (server defaultuje)."""
    tools, captured, restore = _make_mock_tools(lambda r: (201, {"id": "x"}))
    try:
        create = next(t for t in tools if t.name == "hotovo_create_task")
        await create.execute({
            "title": "x", "list_id": "y", "due_date": None, "parent_id": None,
        }, _ctx())
        body = json.loads(captured[0].content)
        assert "due_date" not in body
        assert "parent_id" not in body
    finally:
        restore()


@pytest.mark.asyncio
async def test_complete_task_self_contained_from_build_tools():
    """Codex HIGH #3: build_tools() vrací funkční complete_task BEZ externího
    patche — static_body {status:completed} je přímo v definici."""
    tools, captured, restore = _make_mock_tools(lambda r: (200, {"status": "completed"}))
    try:
        complete = next(t for t in tools if t.name == "hotovo_complete_task")
        result = await complete.execute({"id": "abc"}, _ctx())
        assert result["ok"] is True
        body = json.loads(captured[0].content)
        assert body == {"status": "completed"}
    finally:
        restore()


# ──────────────── Auth file ────────────────


def test_token_file_world_readable_ignored(monkeypatch, tmp_path):
    tf = tmp_path / "tok"
    tf.write_text("MY-TOKEN")
    tf.chmod(0o644)
    monkeypatch.setenv("HOTOVO_API_TOKEN_FILE", str(tf))
    monkeypatch.delenv("HOTOVO_API_TOKEN", raising=False)
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    assert agent_config.get_hotovo_token() == ""


def test_token_file_secure_perms_loaded(monkeypatch, tmp_path):
    tf = tmp_path / "tok"
    tf.write_text("MY-SECURE-TOKEN")
    tf.chmod(0o600)
    monkeypatch.setenv("HOTOVO_API_TOKEN_FILE", str(tf))
    monkeypatch.delenv("HOTOVO_API_TOKEN", raising=False)
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    assert agent_config.get_hotovo_token() == "MY-SECURE-TOKEN"


def test_env_var_overrides_token_file(monkeypatch, tmp_path):
    tf = tmp_path / "tok"
    tf.write_text("FROM-FILE")
    tf.chmod(0o600)
    monkeypatch.setenv("HOTOVO_API_TOKEN_FILE", str(tf))
    monkeypatch.setenv("HOTOVO_API_TOKEN", "FROM-ENV")
    import importlib
    from voice.agent import config as agent_config
    importlib.reload(agent_config)
    assert agent_config.get_hotovo_token() == "FROM-ENV"


# ──────────────── Classifier ────────────────


def test_classifier_get_state_is_auto():
    from voice.agent.permissions import decide, Decision
    result = decide("hotovo_get_state", {}, Path("/tmp"))
    assert result.decision == Decision.AUTO


def test_classifier_create_task_is_ask_medium():
    from voice.agent.permissions import decide, Decision
    result = decide("hotovo_create_task", {"title": "x"}, Path("/tmp"))
    assert result.decision == Decision.ASK
    assert result.risk == "medium"
    assert result.requires_explicit is False


def test_classifier_delete_task_is_destructive():
    from voice.agent.permissions import decide, Decision
    result = decide("hotovo_delete_task", {"id": "x"}, Path("/tmp"))
    assert result.decision == Decision.ASK
    assert result.risk == "destructive"
    assert result.requires_explicit is True
