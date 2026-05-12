"""Unit testy pro web tooly (voice/agent/tools/web.py).

Testují:
- fetch_url: happy path (mock httpx), redirect chain (re-validates každý hop),
  SSRF (private IP literal, post-DNS check), size cap, timeout, content-type
  filter (text vs binary), schema deny (file://, ftp://), userinfo deny.
- web_search: happy path, missing key, 401/403, 429 rate-limit, count validation.

Mockujeme httpx přes httpx.MockTransport (oficiální httpx way) — nepouštíme
real network. DNS resolve mockujeme přes monkeypatch na getaddrinfo.

Volá _fetch_url_exec / _web_search_exec přímo. Classifier testy jsou v
test_agent_permissions.py.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import httpx
import pytest

from voice.agent.tools.base import ExecuteContext
from voice.agent.tools import web as web_mod
from voice.agent.tools.web import (
    FETCH_URL_TOOL,
    WEB_SEARCH_TOOL,
    _fetch_url_exec,
    _web_search_exec,
    _is_text_content_type,
    _brave_extract_results,
)


def _ctx(workdir: Path, *, cancel_event: asyncio.Event | None = None) -> ExecuteContext:
    return ExecuteContext(turn_id="t1", cancel_event=cancel_event, workdir=workdir)


def _patch_dns(monkeypatch, addrs: list[str]):
    """Mock async DNS resolve. addrs = list IP literals (string).

    Patchuje metodu na třídě loop (`_UnixSelectorEventLoop.getaddrinfo`), takže
    první positional je `self` (loop), druhý `host`.
    """
    async def fake(self, host, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (a, 0)) for a in addrs
        ]

    from asyncio.unix_events import _UnixSelectorEventLoop  # type: ignore[attr-defined]
    monkeypatch.setattr(_UnixSelectorEventLoop, "getaddrinfo", fake)


def _patch_dns_fn(monkeypatch, fn):
    """Mock async DNS resolve s vlastní handler funkcí. fn(host) -> list[str].
    Handler dostane host string a vrátí list IP."""
    async def fake(self, host, *args, **kwargs):
        addrs = fn(host)
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", (a, 0)) for a in addrs
        ]

    from asyncio.unix_events import _UnixSelectorEventLoop  # type: ignore[attr-defined]
    monkeypatch.setattr(_UnixSelectorEventLoop, "getaddrinfo", fake)


# ---------------------------------------------------------------------------
# Helpers: _is_text_content_type, _brave_extract_results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ct,expected", [
    ("text/html", True),
    ("text/html; charset=utf-8", True),
    ("application/json", True),
    ("application/json;charset=utf-8", True),
    ("application/xml", True),
    ("application/xhtml+xml", True),
    ("application/ld+json", True),
    ("application/octet-stream", False),
    ("image/png", False),
    ("video/mp4", False),
    ("application/pdf", False),
    ("", False),
])
def test_is_text_content_type(ct, expected):
    assert _is_text_content_type(ct) is expected


def test_brave_extract_results_normalizes():
    payload = {
        "web": {
            "results": [
                {"title": "T1", "url": "https://a.example", "description": "snip1"},
                {"title": "T2", "url": "https://b.example", "description": "snip2"},
                {"title": "T3", "url": "", "description": "skip"},  # no url
                "not a dict",
                {"title": "T4", "url": "https://c.example"},  # no desc
            ]
        }
    }
    out = _brave_extract_results(payload, 10)
    assert len(out) == 3
    assert out[0] == {"title": "T1", "url": "https://a.example", "snippet": "snip1"}
    assert out[2] == {"title": "T4", "url": "https://c.example", "snippet": ""}


def test_brave_extract_results_respects_count():
    payload = {
        "web": {"results": [{"title": "T", "url": f"https://{i}.x", "description": ""} for i in range(10)]}
    }
    assert len(_brave_extract_results(payload, 3)) == 3


def test_brave_extract_results_missing_web():
    assert _brave_extract_results({}, 5) == []
    assert _brave_extract_results({"web": "not a dict"}, 5) == []
    assert _brave_extract_results({"web": {"results": "not a list"}}, 5) == []


# ---------------------------------------------------------------------------
# fetch_url: happy paths via httpx.MockTransport (no real network)
# ---------------------------------------------------------------------------


def _install_mock_transport(monkeypatch, handler):
    """Monkeypatch httpx.AsyncClient default constructor to use MockTransport."""
    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["transport"] = transport
        return original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", patched_init)


def test_fetch_url_happy_path_html(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/page"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            content=b"<html><body>hello</body></html>",
        )

    _install_mock_transport(monkeypatch, handler)

    r = asyncio.run(_fetch_url_exec(
        {"url": "https://example.com/page"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True, r
    assert r["status"] == 200
    assert r["is_text"] is True
    assert "hello" in r["body"]
    assert r["content_type"].startswith("text/html")
    assert r["truncated"] is False
    assert r["redirect_chain"] == []


def test_fetch_url_json_body(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            content=b'{"key": "value"}',
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://api.example.com/x"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert r["is_text"] is True
    assert r["body"] == '{"key": "value"}'


def test_fetch_url_binary_no_body(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "image/png"},
            content=b"\x89PNG\r\n" + b"\x00" * 100,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://cdn.example.com/x.png"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert r["is_text"] is False
    assert r["body"] == ""
    assert r["size_bytes"] > 0


def test_fetch_url_size_cap_truncates(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    big = b"x" * (3 * 1024 * 1024)  # 3 MiB > FETCH_URL_MAX_BYTES (2 MiB default)

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain"},
            content=big,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://big.example.com/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert r["truncated"] is True
    assert r["size_bytes"] <= 2 * 1024 * 1024


def test_fetch_url_redirect_chain(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "/step2"})
        if request.url.path == "/step2":
            return httpx.Response(301, headers={"Location": "https://example.com/final"})
        if request.url.path == "/final":
            return httpx.Response(
                200, headers={"Content-Type": "text/plain"}, content=b"OK",
            )
        return httpx.Response(404)

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://example.com/start"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert r["status"] == 200
    assert r["body"] == "OK"
    assert len(r["redirect_chain"]) == 2


def test_fetch_url_redirect_loop_rejected(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        # Bounce každý request na sebe.
        return httpx.Response(302, headers={"Location": str(request.url)})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://loop.example.com/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "redirect" in r["error"].lower()


def test_fetch_url_redirect_to_private_ip_blocked(tmp_path, monkeypatch):
    # First hop DNS → public, redirect target hostname → private.
    def resolve(host):
        if "internal" in host:
            return ["10.0.0.1"]
        return ["8.8.8.8"]
    _patch_dns_fn(monkeypatch, resolve)

    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={
                "Location": "https://internal.corp/secret",
            })
        return httpx.Response(200, content=b"x")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://example.com/start"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "ssrf" in r["error"].lower() or "blocked" in r["error"].lower()


def test_fetch_url_redirect_to_file_scheme_blocked(tmp_path, monkeypatch):
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        return httpx.Response(302, headers={"Location": "file:///etc/passwd"})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://example.com/start"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "scheme" in r["error"].lower() or "invalid" in r["error"].lower()


def test_fetch_url_invalid_scheme(tmp_path):
    r = asyncio.run(_fetch_url_exec(
        {"url": "file:///etc/passwd"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "scheme" in r["error"].lower()


def test_fetch_url_invalid_url(tmp_path):
    r = asyncio.run(_fetch_url_exec(
        {"url": ""}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_localhost_blocked(tmp_path):
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://localhost/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "blocked" in r["error"].lower() or "ssrf" in r["error"].lower() or "localhost" in r["error"].lower()


def test_fetch_url_private_ip_literal_blocked(tmp_path):
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://10.0.0.1/admin"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_userinfo_blocked(tmp_path):
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://user:pass@example.com/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "userinfo" in r["error"].lower() or "credentials" in r["error"].lower()


def test_fetch_url_dns_to_private_blocked(tmp_path, monkeypatch):
    # DNS resolves to PRIVATE — should be rejected (DNS rebinding).
    _patch_dns(monkeypatch, ["192.168.1.5"])

    def handler(request):
        # Never called.
        return httpx.Response(200, content=b"x")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://innocent-looking-domain.com/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "ssrf" in r["error"].lower() or "blocked" in r["error"].lower()


def test_fetch_url_dns_mixed_with_private_blocked(tmp_path, monkeypatch):
    # Multi-record DNS s jednou private IP — reject (DNS rebinding defense).
    _patch_dns(monkeypatch, ["8.8.8.8", "127.0.0.1"])

    def handler(request):
        return httpx.Response(200, content=b"x")

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://multi-record.example/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


# ---------------------------------------------------------------------------
# iter-2 regression tests: IPv4-mapped IPv6, CGNAT, compression bomb, port
# ---------------------------------------------------------------------------


def test_fetch_url_ipv4_mapped_ipv6_blocked(tmp_path):
    """IPv4-mapped IPv6 (::ffff:127.0.0.1) musí být blokovaná — Python 3.11
    bug: is_private vrací False bez unwrap."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[::ffff:127.0.0.1]/secret"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    err = r["error"].lower()
    assert "blocked" in err or "ssrf" in err or "127.0.0.1" in err


def test_fetch_url_ipv4_mapped_private_ipv6_blocked(tmp_path):
    """`::ffff:10.0.0.1` — privátní IPv4 v IPv6 wrapperu."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[::ffff:10.0.0.1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_cgnat_blocked(tmp_path):
    """100.64.0.0/10 (RFC 6598 CGNAT) blokujeme — `ipaddress.is_private` to
    nevidí."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://100.64.0.5/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    err = r["error"].lower()
    assert "cgnat" in err or "blocked" in err


def test_fetch_url_invalid_port(tmp_path):
    """Port non-numerický — urlparse.port vyhodí ValueError, musíme catch."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://example.com:notaport/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "port" in r["error"].lower() or "invalid" in r["error"].lower()


def test_fetch_url_port_out_of_range(tmp_path):
    """Port >65535 — odmítnuto, ne crash."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://example.com:999999/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_ipv6_site_local_blocked(tmp_path):
    """fec0::/10 (deprecated site-local) — Python is_private vrací False, takže
    musí být explicit blok přes is_site_local."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[fec0::1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_ipv6_nat64_well_known_blocked(tmp_path):
    """NAT64 well-known prefix `64:ff9b::/96` — translates external→internal IPv4."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[64:ff9b::7f00:1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_ipv6_nat64_local_use_blocked(tmp_path):
    """NAT64 local-use prefix `64:ff9b:1::/48` (RFC 8215)."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[64:ff9b:1::1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_ipv4_translated_blocked(tmp_path):
    """IPv4-translated `::ffff:0:0:0/96` (RFC 2765)."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[::ffff:0:0:1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_ipv4_compatible_blocked(tmp_path):
    """Deprecated IPv4-compatible `::a.b.c.d` form embedding loopback."""
    r = asyncio.run(_fetch_url_exec(
        {"url": "http://[::7f00:1]/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_compression_bomb_rejected(tmp_path, monkeypatch):
    """Server vrátí Content-Encoding: gzip i přes Accept-Encoding: identity →
    reject (HIGH-2: gzip/zstd bomb defense)."""
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        # Server vrací gzip i když jsme nechtěli.
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/plain",
                "Content-Encoding": "gzip",
            },
            content=b"\x1f\x8b\x08\x00" + b"\x00" * 100,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://gzipbomb.example/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "encoding" in r["error"].lower() or "content-encoding" in r["error"].lower()


def test_fetch_url_zstd_rejected(tmp_path, monkeypatch):
    """Stejně tak zstd Content-Encoding → reject."""
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain", "Content-Encoding": "zstd"},
            content=b"x" * 100,
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://zstd.example/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_fetch_url_identity_encoding_accepted(tmp_path, monkeypatch):
    """Content-Encoding: identity (= no compression) musí projít."""
    _patch_dns(monkeypatch, ["8.8.8.8"])

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/plain", "Content-Encoding": "identity"},
            content=b"hello world",
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_fetch_url_exec(
        {"url": "https://identity.example/"}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert "hello world" in r["body"]


def test_fetch_url_accept_encoding_identity_header_sent(tmp_path, monkeypatch):
    """Verify že request posíláme s Accept-Encoding: identity (HIGH-2)."""
    _patch_dns(monkeypatch, ["8.8.8.8"])
    captured: dict = {}

    def handler(request):
        captured["accept_encoding"] = request.headers.get("Accept-Encoding")
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, content=b"ok")

    _install_mock_transport(monkeypatch, handler)
    asyncio.run(_fetch_url_exec(
        {"url": "https://example.com/"}, _ctx(tmp_path),
    ))
    assert captured["accept_encoding"] == "identity"


def test_ssrf_backend_blocks_ip_literal_private():
    """_SSRFGuardBackend přímý test — IP literal private → ConnectError."""
    from voice.agent.tools.web import _SSRFGuardBackend
    import httpcore

    class _NoopInner:
        async def connect_tcp(self, host, port, **kwargs):
            return object()
        async def connect_unix_socket(self, path, **kwargs):
            return object()
        async def sleep(self, s):
            pass

    backend = _SSRFGuardBackend(_NoopInner())

    async def go():
        try:
            await backend.connect_tcp("127.0.0.1", 80)
        except httpcore.ConnectError as e:
            return str(e)
        return None

    err = asyncio.run(go())
    assert err is not None
    assert "ssrf" in err.lower() or "blocked" in err.lower()


def test_ssrf_backend_blocks_dns_to_private(monkeypatch):
    """_SSRFGuardBackend — host name resolves na privátní IP → ConnectError."""
    from voice.agent.tools.web import _SSRFGuardBackend
    import httpcore

    _patch_dns(monkeypatch, ["10.0.0.1"])

    class _NoopInner:
        async def connect_tcp(self, host, port, **kwargs):
            return object()
        async def connect_unix_socket(self, path, **kwargs):
            return object()
        async def sleep(self, s):
            pass

    backend = _SSRFGuardBackend(_NoopInner())

    async def go():
        try:
            await backend.connect_tcp("example.com", 80)
        except httpcore.ConnectError as e:
            return str(e)
        return None

    err = asyncio.run(go())
    assert err is not None
    assert "ssrf" in err.lower() or "blocked" in err.lower() or "10.0.0.1" in err


def test_ssrf_backend_allows_public(monkeypatch):
    """_SSRFGuardBackend — public IP literal projde do inner.connect_tcp."""
    from voice.agent.tools.web import _SSRFGuardBackend

    captured: dict = {}

    class _CapturingInner:
        async def connect_tcp(self, host, port, **kwargs):
            captured["host"] = host
            captured["port"] = port
            return "stream"
        async def connect_unix_socket(self, path, **kwargs):
            return None
        async def sleep(self, s):
            pass

    backend = _SSRFGuardBackend(_CapturingInner())

    async def go():
        return await backend.connect_tcp("8.8.8.8", 443)

    res = asyncio.run(go())
    assert res == "stream"
    assert captured["host"] == "8.8.8.8"
    assert captured["port"] == 443


def test_ssrf_backend_unix_socket_denied():
    """_SSRFGuardBackend — unix socket (docker.sock SSRF vektor) → reject."""
    from voice.agent.tools.web import _SSRFGuardBackend
    import httpcore

    class _NoopInner:
        async def connect_tcp(self, host, port, **kwargs):
            return object()
        async def connect_unix_socket(self, path, **kwargs):
            return object()
        async def sleep(self, s):
            pass

    backend = _SSRFGuardBackend(_NoopInner())

    async def go():
        try:
            await backend.connect_unix_socket("/var/run/docker.sock")
        except httpcore.ConnectError as e:
            return str(e)
        return None

    err = asyncio.run(go())
    assert err is not None
    assert "unix" in err.lower()


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


def test_web_search_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "BSAfake")

    def handler(request):
        # Verify auth header is sent.
        assert request.headers.get("X-Subscription-Token") == "BSAfake"
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={
                "web": {
                    "results": [
                        {"title": "Python docs", "url": "https://docs.python.org/", "description": "Official"},
                        {"title": "PEP 8", "url": "https://peps.python.org/pep-0008/", "description": "Style"},
                    ]
                }
            },
        )

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_web_search_exec(
        {"query": "python tutorial", "count": 2}, _ctx(tmp_path),
    ))
    assert r["ok"] is True
    assert r["count"] == 2
    assert r["results"][0]["title"] == "Python docs"


def test_web_search_no_key(tmp_path, monkeypatch):
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "")
    r = asyncio.run(_web_search_exec(
        {"query": "x"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "BRAVE_SEARCH_API_KEY" in r["error"]


def test_web_search_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "BSAfake")

    def handler(request):
        return httpx.Response(429, json={"error": "rate"})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_web_search_exec(
        {"query": "test"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "429" in r["error"]


def test_web_search_auth_error(tmp_path, monkeypatch):
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "BSAbadkey")

    def handler(request):
        return httpx.Response(401, json={"error": "unauthorized"})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_web_search_exec(
        {"query": "test"}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "401" in r["error"] or "auth" in r["error"].lower()


def test_web_search_invalid_count(tmp_path):
    monkeypatch_count = 0
    r = asyncio.run(_web_search_exec(
        {"query": "x", "count": 50}, _ctx(tmp_path),
    ))
    assert r["ok"] is False
    assert "count" in r["error"].lower()


def test_web_search_count_zero(tmp_path):
    r = asyncio.run(_web_search_exec(
        {"query": "x", "count": 0}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_web_search_empty_query(tmp_path):
    r = asyncio.run(_web_search_exec(
        {"query": ""}, _ctx(tmp_path),
    ))
    assert r["ok"] is False


def test_web_search_default_count(tmp_path, monkeypatch):
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "BSAfake")
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"web": {"results": []}})

    _install_mock_transport(monkeypatch, handler)
    asyncio.run(_web_search_exec({"query": "x"}, _ctx(tmp_path)))
    # default je 5
    assert captured["params"].get("count") == "5"


def test_web_search_does_not_follow_redirects(tmp_path, monkeypatch):
    """Anti-spoofing — Brave odpověď nesmí být ovlivněna 30x (token leak risk)."""
    monkeypatch.setattr(web_mod, "BRAVE_SEARCH_API_KEY", "BSAfake")

    def handler(request):
        return httpx.Response(302, headers={"Location": "https://evil.example/"})

    _install_mock_transport(monkeypatch, handler)
    r = asyncio.run(_web_search_exec({"query": "x"}, _ctx(tmp_path)))
    # 302 počítáme jako non-2xx error (≥400 check ho mine — ale ne-2xx + nečitelný JSON).
    # Actually: 302 < 400 → padá do try parse json → invalid JSON. Result ok=False.
    assert r["ok"] is False


# ---------------------------------------------------------------------------
# Tool metadata sanity
# ---------------------------------------------------------------------------


def test_fetch_url_tool_metadata():
    assert FETCH_URL_TOOL.name == "fetch_url"
    assert "url" in FETCH_URL_TOOL.parameters_schema["properties"]
    assert FETCH_URL_TOOL.parameters_schema["required"] == ["url"]


def test_web_search_tool_metadata():
    assert WEB_SEARCH_TOOL.name == "web_search"
    assert "query" in WEB_SEARCH_TOOL.parameters_schema["properties"]
    assert "count" in WEB_SEARCH_TOOL.parameters_schema["properties"]
