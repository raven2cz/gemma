"""Unit testy pro Hue tooly (voice/agent/tools/hue.py).

Testují:
- Config-missing path (žádný bridge / key) → ok=False bez network volání.
- `light_list`: happy path (mock httpx), 401 auth, 5xx server error, invalid JSON,
  response too large, empty data, timeout.
- `light_set`: validace args (name required, on bool, brightness range,
  color_name palette, at least one change), name→ID resolve (exact, partial,
  multi-match, missing), happy path GET+PUT, error propagation.
- COLOR_PALETTE: každá barva má xy v rozsahu 0..1.
- `_hue_url`: sanitizace path (žádný `..`, žádný scheme injection).

Mockujeme httpx přes httpx.MockTransport.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from voice.agent.tools import hue as hue_mod
from voice.agent.tools.base import ExecuteContext
from voice.agent.tools.hue import (
    COLOR_PALETTE,
    LIGHT_LIST_TOOL,
    LIGHT_SET_TOOL,
    _hue_url,
    _light_list_exec,
    _light_set_exec,
    _resolve_light_id,
    _summarize_light,
)


def _ctx(workdir: Path) -> ExecuteContext:
    return ExecuteContext(turn_id="t1", cancel_event=None, workdir=workdir)


def _install_mock_transport(monkeypatch, handler):
    """Force httpx.AsyncClient() to use MockTransport regardless of kwargs."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        # Drop verify=False so it doesn't conflict (MockTransport ignores it).
        return original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


def _set_hue_config(monkeypatch, *, bridge: str = "192.168.1.118", key: str = "X" * 40):
    """Inject Hue config without touching the filesystem."""
    monkeypatch.setattr(hue_mod, "HUE_BRIDGE_IP", bridge)
    monkeypatch.setattr(hue_mod, "HUE_APP_KEY", key)


# ---------------------------------------------------------------------------
# COLOR_PALETTE sanity
# ---------------------------------------------------------------------------


def test_color_palette_values_in_range():
    assert "red" in COLOR_PALETTE
    assert "warm" in COLOR_PALETTE
    for name, (x, y) in COLOR_PALETTE.items():
        assert 0.0 <= x <= 1.0, f"{name}: x={x}"
        assert 0.0 <= y <= 1.0, f"{name}: y={y}"


def test_color_palette_size():
    # Sanity that palette is intentional and small.
    assert 10 <= len(COLOR_PALETTE) <= 25


# ---------------------------------------------------------------------------
# _hue_url path sanitization
# ---------------------------------------------------------------------------


def test_hue_url_basic(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    assert _hue_url("light") == "https://1.2.3.4/clip/v2/resource/light"


def test_hue_url_rejects_leading_slash(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    with pytest.raises(ValueError):
        _hue_url("/light/abc")


def test_hue_url_rejects_traversal(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    with pytest.raises(ValueError):
        _hue_url("../../etc/passwd")


def test_hue_url_rejects_scheme_injection(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    with pytest.raises(ValueError):
        _hue_url("http://evil/x")


def test_hue_url_rejects_query_fragment(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    with pytest.raises(ValueError):
        _hue_url("light?x=1")
    with pytest.raises(ValueError):
        _hue_url("light#frag")


def test_hue_url_rejects_control_chars(monkeypatch):
    _set_hue_config(monkeypatch, bridge="1.2.3.4")
    with pytest.raises(ValueError):
        _hue_url("light\nfoo")
    with pytest.raises(ValueError):
        _hue_url("light bar")


# ---------------------------------------------------------------------------
# _summarize_light
# ---------------------------------------------------------------------------


def test_summarize_light_full_record():
    raw = {
        "id": "abc-123",
        "metadata": {"name": "Kuchyně", "archetype": "ceiling_round"},
        "on": {"on": True},
        "dimming": {"brightness": 75.0},
        "color": {"xy": {"x": 0.5, "y": 0.4}},
    }
    out = _summarize_light(raw)
    assert out["id"] == "abc-123"
    assert out["name"] == "Kuchyně"
    assert out["archetype"] == "ceiling_round"
    assert out["on"] is True
    assert out["brightness"] == 75.0
    assert out["color_xy"] == [0.5, 0.4]


def test_summarize_light_missing_fields():
    out = _summarize_light({"id": "x"})
    assert out["id"] == "x"
    assert out["name"] == ""
    assert out["on"] is None
    assert out["brightness"] is None
    assert out["color_xy"] is None


# ---------------------------------------------------------------------------
# Config missing path
# ---------------------------------------------------------------------------


def test_light_list_no_config(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch, bridge="", key="")
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "not configured" in r["error"].lower() or "configured" in r["error"].lower()


def test_light_set_short_key(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch, bridge="1.2.3.4", key="too_short")
    r = asyncio.run(_light_set_exec({"name": "x", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "invalid" in r["error"].lower() or "short" in r["error"].lower()


# ---------------------------------------------------------------------------
# light_list
# ---------------------------------------------------------------------------


def _mk_light(id_, name, *, on=True, bri=80.0, x=0.4, y=0.4, archetype="bulb"):
    return {
        "id": id_,
        "metadata": {"name": name, "archetype": archetype},
        "on": {"on": on},
        "dimming": {"brightness": bri},
        "color": {"xy": {"x": x, "y": y}},
    }


def test_light_list_happy(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {
        "data": [
            _mk_light("id1", "Obývák"),
            _mk_light("id2", "Kuchyň", on=False, bri=10.0),
        ]
    }

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/clip/v2/resource/light"
        assert req.headers["hue-application-key"] == "X" * 40
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is True, r
    assert r["count"] == 2
    assert r["lights"][0]["id"] == "id1"
    assert r["lights"][0]["name"] == "Obývák"
    assert r["lights"][1]["on"] is False


def test_light_list_auth_error(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(401, json={"errors": ["unauthorized"]})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "auth" in r["error"].lower()


def test_light_list_server_error(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(503, text="busy")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "503" in r["error"]


def test_light_list_invalid_json(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(200, content=b"not-json{")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "json" in r["error"].lower()


def test_light_list_data_missing(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(200, json={"not_data": []})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "data" in r["error"].lower()


def test_light_list_size_cap(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    # Force tiny cap so any payload trips it.
    monkeypatch.setattr(hue_mod, "HUE_OUTPUT_CAP_BYTES", 10)

    def handler(req):
        return httpx.Response(200, json={"data": [_mk_light("id1", "x")]})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "large" in r["error"].lower() or "too" in r["error"].lower()


def test_light_list_content_length_precheck(monkeypatch, tmp_path):
    """Server announces huge Content-Length → we reject BEFORE reading body."""
    _set_hue_config(monkeypatch)
    monkeypatch.setattr(hue_mod, "HUE_OUTPUT_CAP_BYTES", 1024)

    def handler(req):
        # Tiny actual body, but lying Content-Length header.
        return httpx.Response(
            200,
            headers={"Content-Length": str(10 * 1024 * 1024)},
            content=b'{"data":[]}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "large" in r["error"].lower() or "too" in r["error"].lower()


def test_light_list_stream_cap_no_content_length(monkeypatch, tmp_path):
    """Stream-mode: server lies / omits Content-Length, body is large → cap hit
    during chunk loop, response truncated."""
    _set_hue_config(monkeypatch)
    monkeypatch.setattr(hue_mod, "HUE_OUTPUT_CAP_BYTES", 50)

    big = b"x" * (4 * 1024)  # 4 KiB body > 50 B cap

    def handler(req):
        # No Content-Length header so pre-check skips → cap must catch in loop.
        return httpx.Response(200, content=big)

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "large" in r["error"].lower() or "too" in r["error"].lower()


def test_light_list_request_error(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)

    def handler(req):
        raise httpx.ConnectError("connection refused")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "hue request error" in r["error"].lower()
    # Don't leak full exception detail (could include internal host info).
    assert "connection refused" not in r["error"]


# ---------------------------------------------------------------------------
# _resolve_light_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_light_id_exact(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {
        "data": [
            _mk_light("id1", "Obývák"),
            _mk_light("id2", "Kuchyň"),
        ]
    }

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "obývák")
    assert rid == "id1"
    assert err == ""


@pytest.mark.asyncio
async def test_resolve_light_id_partial_unique(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("id1", "Obývák Strop"), _mk_light("id2", "Kuchyň")]}

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "strop")
    assert rid == "id1"
    assert err == ""


@pytest.mark.asyncio
async def test_resolve_light_id_multiple_partial(monkeypatch):
    _set_hue_config(monkeypatch)
    payload = {
        "data": [
            _mk_light("id1", "Obývák Strop"),
            _mk_light("id2", "Kuchyň Strop"),
        ]
    }

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "strop")
    assert rid == ""
    assert "multiple" in err.lower()


@pytest.mark.asyncio
async def test_resolve_light_id_missing(monkeypatch):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("id1", "A"), _mk_light("id2", "B")]}

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "neexistuje")
    assert rid == ""
    assert "not found" in err.lower()
    assert "A" in err and "B" in err  # available names in error message


@pytest.mark.asyncio
async def test_resolve_light_id_empty_name(monkeypatch):
    _set_hue_config(monkeypatch)
    payload = {"data": []}

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "  ")
    assert rid == ""
    assert "empty" in err.lower()


@pytest.mark.asyncio
async def test_resolve_light_id_exact_prefers_over_partial(monkeypatch):
    _set_hue_config(monkeypatch)
    payload = {
        "data": [
            _mk_light("id1", "Kuchyň Strop"),  # partial substring "kuchyň"
            _mk_light("id2", "Kuchyň"),         # exact "kuchyň"
        ]
    }

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    async with httpx.AsyncClient() as client:
        rid, err = await _resolve_light_id(client, "Kuchyň")
    assert rid == "id2"  # exact match wins
    assert err == ""


# ---------------------------------------------------------------------------
# light_set
# ---------------------------------------------------------------------------


def test_light_set_no_args(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "name" in r["error"].lower()


def test_light_set_empty_name(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "   ", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "name" in r["error"].lower()


def test_light_set_no_change(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "obývák"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "at least one" in r["error"].lower()


def test_light_set_on_not_bool(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "obývák", "on": "yes"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "boolean" in r["error"].lower() or "bool" in r["error"].lower()


def test_light_set_brightness_out_of_range(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "obývák", "brightness": 150}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "range" in r["error"].lower()
    r2 = asyncio.run(_light_set_exec({"name": "x", "brightness": -1}, _ctx(tmp_path)))
    assert r2["ok"] is False


def test_light_set_brightness_nan_rejected(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec(
        {"name": "x", "brightness": float("nan")}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "finite" in r["error"].lower() or "number" in r["error"].lower()


def test_light_set_brightness_inf_rejected(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec(
        {"name": "x", "brightness": float("inf")}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_light_set_brightness_bool_rejected(monkeypatch, tmp_path):
    """bool is int subclass — explicit reject (True would coerce to 1.0)."""
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec(
        {"name": "x", "brightness": True}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "number" in r["error"].lower()


def test_light_set_brightness_not_number(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "x", "brightness": "high"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "number" in r["error"].lower()


def test_light_set_unknown_color(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    r = asyncio.run(_light_set_exec({"name": "x", "color_name": "puce"}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "unknown" in r["error"].lower() or "color_name" in r["error"].lower()


def test_light_set_happy_brightness_and_color(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idA", "Obývák")]}
    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/clip/v2/resource/light":
            return httpx.Response(200, json=payload)
        if req.method == "PUT":
            assert req.url.path == "/clip/v2/resource/light/idA"
            captured["body"] = json.loads(req.content)
            assert req.headers["hue-application-key"] == "X" * 40
            return httpx.Response(200, json={"data": [{"rid": "idA"}]})
        raise AssertionError(f"unexpected request {req.method} {req.url}")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec(
        {"name": "Obývák", "brightness": 60, "color_name": "warm"},
        _ctx(tmp_path),
    ))
    assert r["ok"] is True, r
    assert r["light_id"] == "idA"
    assert "dimming" in r["applied"]
    assert "color" in r["applied"]
    body = captured["body"]
    assert body["dimming"]["brightness"] == 60
    x, y = COLOR_PALETTE["warm"]
    assert body["color"]["xy"] == {"x": x, "y": y}


def test_light_set_brightness_zero_turns_off(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idB", "Lampa")]}
    captured = {}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": [{"rid": "idB"}]})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "lampa", "brightness": 0}, _ctx(tmp_path)))
    assert r["ok"] is True
    assert captured["body"] == {"on": {"on": False}}


def test_light_set_brightness_floor(monkeypatch, tmp_path):
    """brightness=0.5 (>0 but <1) → floor at 1.0 (Hue accepts 1..100)."""
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idC", "L")]}
    captured = {}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": []})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "l", "brightness": 0.5}, _ctx(tmp_path)))
    assert r["ok"] is True
    assert captured["body"]["dimming"]["brightness"] == 1.0


def test_light_set_resolves_partial_match(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idX", "Obývák Strop")]}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        return httpx.Response(200, json={"data": []})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "strop", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is True
    assert r["light_id"] == "idX"


def test_light_set_not_found(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idA", "A")]}

    def handler(req):
        return httpx.Response(200, json=payload)

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "neexistuje", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "not found" in r["error"].lower()


def test_light_set_auth_error_on_put(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idA", "A")]}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        return httpx.Response(403, json={"err": "forbidden"})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "A", "on": False}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "auth" in r["error"].lower()


def test_light_set_bridge_400_on_put(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idA", "A")]}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        # Hue might return bridge-internal body on error.
        return httpx.Response(400, json={"errors": [{"description": "internal-debug-info"}]})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "A", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is False
    # Don't leak bridge debug info into error message.
    assert "internal-debug-info" not in r["error"]
    assert "400" in r["error"]


def test_light_set_put_response_size_cap(monkeypatch, tmp_path):
    """PUT response is now ALSO streamed + capped (iter-2 fix). A bridge
    returning a huge PUT response body must not OOM the agent."""
    _set_hue_config(monkeypatch)
    monkeypatch.setattr(hue_mod, "HUE_OUTPUT_CAP_BYTES", 50)
    payload = {"data": [_mk_light("idA", "A")]}

    def handler(req):
        if req.method == "GET":
            # Lookup response (size cap applied here too) — must stay small.
            return httpx.Response(200, content=b'{"data":[{"id":"idA","metadata":{"name":"A"}}]}')
        # PUT: oversized body announced via Content-Length header.
        big = b"x" * 1024
        return httpx.Response(
            200,
            headers={"Content-Length": str(len(big))},
            content=big,
        )

    _install_mock_transport(monkeypatch, handler)
    # Need a slightly larger cap for the GET (lookup body ~ 50 B). Bump cap to
    # 256 B so GET passes but PUT (1024 B) trips the cap.
    monkeypatch.setattr(hue_mod, "HUE_OUTPUT_CAP_BYTES", 256)
    r = asyncio.run(_light_set_exec({"name": "A", "on": True}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "large" in r["error"].lower() or "too" in r["error"].lower()


def test_light_list_rejects_gzip_response(monkeypatch, tmp_path):
    """Compression-bomb defense: any non-identity Content-Encoding is rejected
    before reading body (we send Accept-Encoding: identity)."""
    import gzip
    _set_hue_config(monkeypatch)

    valid_gzipped = gzip.compress(b'{"data":[]}')

    def handler(req):
        # Server ignores our Accept-Encoding:identity request and sends gzip.
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip", "Content-Type": "application/json"},
            content=valid_gzipped,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "encoding" in r["error"].lower() or "content" in r["error"].lower()


def test_light_list_rejects_unknown_encoding(monkeypatch, tmp_path):
    """Even unknown CE codecs (e.g. zstd, br) are rejected outright."""
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(
            200,
            headers={"Content-Encoding": "snappy", "Content-Type": "application/json"},
            content=b'{"data":[]}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "encoding" in r["error"].lower() or "content" in r["error"].lower()


def test_light_list_accepts_identity_encoding(monkeypatch, tmp_path):
    """Content-Encoding: identity is the only allowed value."""
    _set_hue_config(monkeypatch)

    def handler(req):
        return httpx.Response(
            200,
            headers={"Content-Encoding": "identity"},
            json={"data": [_mk_light("id1", "x")]},
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is True


def test_light_list_sends_accept_encoding_identity(monkeypatch, tmp_path):
    """Verify our request advertises Accept-Encoding: identity to the bridge."""
    _set_hue_config(monkeypatch)
    captured = {}

    def handler(req):
        captured["accept_encoding"] = req.headers.get("accept-encoding", "")
        return httpx.Response(200, json={"data": []})

    _install_mock_transport(monkeypatch, handler)
    asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert captured["accept_encoding"] == "identity"


def test_light_set_only_on(monkeypatch, tmp_path):
    _set_hue_config(monkeypatch)
    payload = {"data": [_mk_light("idA", "Strop")]}
    captured = {}

    def handler(req):
        if req.method == "GET":
            return httpx.Response(200, json=payload)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"data": []})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_set_exec({"name": "strop", "on": False}, _ctx(tmp_path)))
    assert r["ok"] is True
    assert captured["body"] == {"on": {"on": False}}


# ---------------------------------------------------------------------------
# Key never leaks
# ---------------------------------------------------------------------------


def test_no_key_in_error_messages(monkeypatch, tmp_path):
    """Ensure no Hue error path echoes the application key."""
    _set_hue_config(monkeypatch, key="SUPERSECRETKEYDONOTLEAK_PADDING_PADDING")

    def handler(req):
        return httpx.Response(500, text="bridge error")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_light_list_exec({}, _ctx(tmp_path)))
    assert r["ok"] is False
    assert "SUPERSECRET" not in json.dumps(r)


# ---------------------------------------------------------------------------
# Tool schema sanity
# ---------------------------------------------------------------------------


def test_light_list_tool_metadata():
    assert LIGHT_LIST_TOOL.name == "light_list"
    assert "lights" in LIGHT_LIST_TOOL.description.lower()
    assert LIGHT_LIST_TOOL.parameters_schema["properties"] == {}


def test_light_set_tool_metadata():
    assert LIGHT_SET_TOOL.name == "light_set"
    props = LIGHT_SET_TOOL.parameters_schema["properties"]
    assert "name" in props and "on" in props and "brightness" in props and "color_name" in props
    assert LIGHT_SET_TOOL.parameters_schema["required"] == ["name"]
    assert LIGHT_SET_TOOL.parameters_schema["additionalProperties"] is False
