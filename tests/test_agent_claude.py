"""Tests pro `ask_claude` tool a `claude_bridge` subprocess wrapper.

Mockujeme `asyncio.create_subprocess_exec` - žádný reálný `claude` CLI.
Fake subprocess implementuje stdin/stdout/stderr/wait minimum pro bridge.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from voice.agent import claude_bridge
from voice.agent.tools.base import ExecuteContext
from voice.agent.tools.claude import ASK_CLAUDE_TOOL


# ─────────────────────── Fake subprocess infrastructure ───────────────────────


class _FakeStreamReader:
    """Async StreamReader fake - `await read(n)` vrací postupně dané chunky."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._closed = False

    async def read(self, n: int = -1) -> bytes:
        if self._closed or not self._chunks:
            self._closed = True
            return b""
        chunk = self._chunks.pop(0)
        return chunk if n < 0 or len(chunk) <= n else chunk[:n]


class _FakeStreamWriter:
    """Async StreamWriter fake - sbírá zapsané bytes (nikdy neblokuje)."""

    def __init__(self):
        self.buffer = bytearray()
        self._closed = False

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeProcess:
    """Mock pro `asyncio.subprocess.Process`."""

    _next_pid = 100000

    def __init__(
        self,
        *,
        stdout_chunks: list[bytes],
        stderr_chunks: list[bytes],
        returncode: int = 0,
        wait_delay: float = 0.0,
    ):
        _FakeProcess._next_pid += 1
        self.pid = _FakeProcess._next_pid
        self.stdin = _FakeStreamWriter()
        self.stdout = _FakeStreamReader(stdout_chunks)
        self.stderr = _FakeStreamReader(stderr_chunks)
        self._returncode: int | None = None
        self._final_returncode = returncode
        self._wait_delay = wait_delay
        self._terminated = False

    @property
    def returncode(self) -> int | None:
        return self._returncode

    async def wait(self) -> int:
        if self._wait_delay > 0:
            await asyncio.sleep(self._wait_delay)
        if self._terminated:
            self._returncode = -15  # SIGTERM
        else:
            self._returncode = self._final_returncode
        return self._returncode

    def terminate(self) -> None:
        self._terminated = True
        self._returncode = -15


def _patch_subprocess(
    monkeypatch,
    proc: _FakeProcess | None = None,
    *,
    raises: type[BaseException] | None = None,
) -> dict:
    """Mockuj `asyncio.create_subprocess_exec` aby vrátil daný fake proc.
    `raises=FileNotFoundError` simuluje že CLI binary chybí.

    Vrací captured kwargs (argv, cwd, env) z volání pro asserce.
    """
    captured: dict = {}

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        captured["cwd"] = kwargs.get("cwd")
        captured["env"] = kwargs.get("env")
        captured["start_new_session"] = kwargs.get("start_new_session")
        if raises is not None:
            raise raises("fake")
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return captured


def _patch_killpg(monkeypatch) -> list:
    """No-op os.killpg + os.getpgid → returns fake PGID. Vrací list volání."""
    calls: list = []

    def fake_killpg(pgid, sig):
        calls.append(("killpg", pgid, sig))

    def fake_getpgid(pid):
        return pid  # pretend pid==pgid

    monkeypatch.setattr(os, "killpg", fake_killpg)
    monkeypatch.setattr(os, "getpgid", fake_getpgid)
    return calls


def _make_ctx(cancel_event: asyncio.Event | None = None) -> ExecuteContext:
    return ExecuteContext(
        turn_id="test",
        cancel_event=cancel_event,
        workdir=Path("/tmp"),
        resolved_path=None,
    )


# ──────────────────────────── Tool metadata ────────────────────────────


def test_tool_metadata():
    """Tool má správné jméno + schema kontrakt."""
    assert ASK_CLAUDE_TOOL.name == "ask_claude"
    assert "Claude Code CLI subprocess" in ASK_CLAUDE_TOOL.description
    schema = ASK_CLAUDE_TOOL.parameters_schema
    assert "prompt" in schema["required"]
    assert "prompt" in schema["properties"]
    assert "system" in schema["properties"]
    assert "max_tokens" in schema["properties"]


# ──────────────────────────── Args validation ────────────────────────────


@pytest.mark.asyncio
async def test_empty_prompt():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": ""}, _make_ctx())
    assert r["ok"] is False
    assert "empty" in r["error"]


@pytest.mark.asyncio
async def test_whitespace_prompt():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "   \n\t  "}, _make_ctx())
    assert r["ok"] is False


@pytest.mark.asyncio
async def test_non_string_prompt():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": 123}, _make_ctx())
    assert r["ok"] is False
    assert "string" in r["error"]


@pytest.mark.asyncio
async def test_oversized_prompt():
    from voice.agent.config import CLAUDE_MAX_PROMPT_BYTES
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "x" * (CLAUDE_MAX_PROMPT_BYTES + 10)}, _make_ctx(),
    )
    assert r["ok"] is False
    assert "too large" in r["error"]


@pytest.mark.asyncio
async def test_system_non_string():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "system": 42}, _make_ctx())
    assert r["ok"] is False


@pytest.mark.asyncio
async def test_system_oversized():
    from voice.agent.config import CLAUDE_MAX_SYSTEM_BYTES
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "hi", "system": "x" * (CLAUDE_MAX_SYSTEM_BYTES + 10)}, _make_ctx(),
    )
    assert r["ok"] is False
    assert "too large" in r["error"]


@pytest.mark.asyncio
async def test_max_tokens_bool_rejected():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "max_tokens": True}, _make_ctx())
    assert r["ok"] is False


@pytest.mark.asyncio
async def test_max_tokens_invalid_type():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "max_tokens": "abc"}, _make_ctx())
    assert r["ok"] is False


@pytest.mark.asyncio
async def test_max_tokens_out_of_range():
    from voice.agent.config import CLAUDE_MAX_TOKENS_LIMIT
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "hi", "max_tokens": CLAUDE_MAX_TOKENS_LIMIT + 1}, _make_ctx(),
    )
    assert r["ok"] is False


# ───────────────────────── Bridge - happy path ─────────────────────────


@pytest.mark.asyncio
async def test_happy_path(monkeypatch):
    """Validní prompt → subprocess úspěch → JSON parsed → ok=True + text.
    Auth si Claude CLI řeší sám (OAuth/keychain/API key) - bridge se na to
    nedívá. Pokud `claude` v terminálu funguje, funguje i tady."""

    response_json = json.dumps({
        "type": "result", "subtype": "success",
        "result": "Hello from Claude.",
        "session_id": "abc-123",
        "total_cost_usd": 0.0012,
        "is_error": False,
    })
    proc = _FakeProcess(
        stdout_chunks=[response_json.encode("utf-8")],
        stderr_chunks=[],
        returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "Hi Claude"}, _make_ctx())

    assert r["ok"] is True
    assert r["text"] == "Hello from Claude."
    assert r["session_id"] == "abc-123"
    assert r["total_cost_usd"] == 0.0012
    assert r["duration_ms"] >= 0
    # Argv security: prompt MUSÍ NE být v argv (`ps` leak)
    argv_joined = " ".join(captured["argv"])
    assert "Hi Claude" not in argv_joined, "prompt leak v argv!"
    # Bezpečnostní flagy MUSÍ být přítomné. NE `--bare` (zakazuje OAuth/
    # keychain auth, vynucuje API klíč - user by ho jinak nepotřeboval).
    assert "--bare" not in captured["argv"]
    assert "--no-session-persistence" in captured["argv"]
    assert "--permission-mode" in captured["argv"]
    assert "plan" in captured["argv"]
    assert "--tools" in captured["argv"]
    assert "" in captured["argv"]  # --tools ""
    assert captured["start_new_session"] is True
    # Prompt poslán přes stdin
    assert proc.stdin.buffer == b"Hi Claude"
    # Empty cwd (temp dir, ne náš workdir)
    assert captured["cwd"] is not None
    assert "claude_bridge_" in captured["cwd"]


@pytest.mark.asyncio
async def test_system_via_argv(monkeypatch):
    """System prompt jde přes --append-system-prompt argv (CLI 2.1.x nemá file flag)."""
    response = json.dumps({"result": "ok", "is_error": False})
    proc = _FakeProcess(stdout_chunks=[response.encode()], stderr_chunks=[])
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "Hi", "system": "You are a tester"}, _make_ctx(),
    )

    assert r["ok"] is True
    assert "--append-system-prompt" in captured["argv"]
    assert "You are a tester" in captured["argv"]


# ───────────────────────── Bridge - error paths ─────────────────────────


@pytest.mark.asyncio
async def test_cli_not_found(monkeypatch):
    """`claude` binary chybí → FileNotFoundError → user-facing error."""
    _patch_subprocess(monkeypatch, proc=None, raises=FileNotFoundError)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "claude CLI nenalezen" in r["error"]


@pytest.mark.asyncio
async def test_nonzero_exit(monkeypatch):
    """Exit != 0 → ok=False + stderr_preview."""
    proc = _FakeProcess(
        stdout_chunks=[b""],
        stderr_chunks=[b"Error: rate limit exceeded\n"],
        returncode=1,
    )
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert r["exit_code"] == 1
    assert "rate limit" in r["stderr_preview"]


@pytest.mark.asyncio
async def test_invalid_json_output(monkeypatch):
    """CLI vrátil garbled output → invalid JSON error."""
    proc = _FakeProcess(
        stdout_chunks=[b"not valid json {{"],
        stderr_chunks=[],
        returncode=0,
    )
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "invalid JSON" in r["error"]


@pytest.mark.asyncio
async def test_is_error_flag(monkeypatch):
    """JSON má is_error=True → ok=False s message z result."""
    response = json.dumps({
        "result": "API quota exceeded", "is_error": True,
    })
    proc = _FakeProcess(stdout_chunks=[response.encode()], stderr_chunks=[])
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "quota" in r["error"]


# ─────────────────────── Bridge - timeout / cancel ───────────────────────


@pytest.mark.asyncio
async def test_timeout(monkeypatch):
    """Subprocess nedokončí v timeout_sec → killpg + ok=False."""
    monkeypatch.setattr("voice.agent.tools.claude.CLAUDE_TIMEOUT_SEC", 0.2)
    proc = _FakeProcess(
        stdout_chunks=[b""], stderr_chunks=[],
        returncode=0, wait_delay=5.0,  # >> timeout
    )
    _patch_subprocess(monkeypatch, proc)
    killpg_calls = _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "timeout" in r["error"]
    # killpg byl zavolán
    assert any(c[0] == "killpg" for c in killpg_calls)


@pytest.mark.asyncio
async def test_cancel_event(monkeypatch):
    """`cancel_event.set()` během běhu → killpg + ok=False canceled."""
    proc = _FakeProcess(
        stdout_chunks=[b""], stderr_chunks=[],
        returncode=0, wait_delay=2.0,
    )
    _patch_subprocess(monkeypatch, proc)
    killpg_calls = _patch_killpg(monkeypatch)

    cancel_event = asyncio.Event()
    ctx = _make_ctx(cancel_event=cancel_event)

    async def trigger_cancel():
        await asyncio.sleep(0.1)
        cancel_event.set()

    asyncio.create_task(trigger_cancel())
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, ctx)

    assert r["ok"] is False
    assert "canceled" in r["error"]
    assert any(c[0] == "killpg" for c in killpg_calls)


# ───────────────────────── Bridge - output cap ─────────────────────────


@pytest.mark.asyncio
async def test_output_cap_kills_process(monkeypatch):
    """Stdout > cap_bytes → on_overflow zavolá killpg, buffer truncated."""
    monkeypatch.setattr("voice.agent.tools.claude.CLAUDE_OUTPUT_CAP_BYTES", 1024)
    # 2 KiB chunk = překročí 1 KiB cap
    huge = b"X" * (2 * 1024)
    proc = _FakeProcess(stdout_chunks=[huge], stderr_chunks=[])
    _patch_subprocess(monkeypatch, proc)
    killpg_calls = _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "exceeded" in r["error"]
    assert any(c[0] == "killpg" for c in killpg_calls)


# ─────────────────── Env scrubbing (security: no secret leak) ───────────────────


def test_env_scrubbed_keeps_safe_vars(monkeypatch):
    """Env passed to subprocess obsahuje JEN allowlist (HOME/USER/PATH/LANG/
    XDG_*/ANTHROPIC_API_KEY když je), NIC z AGENT_*/BRAVE_*/GH_*/*_TOKEN."""
    monkeypatch.setenv("AGENT_INTERNAL_FLAG", "yes")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brv-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "gh_secret")
    monkeypatch.setenv("MY_API_KEY", "my-secret")
    monkeypatch.setenv("HOME", "/home/user")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env = claude_bridge._build_subprocess_env()

    # MUSÍ obsahovat (allowlist)
    assert env["HOME"] == "/home/user"
    assert env["PATH"] == "/usr/bin:/bin"
    # NESMÍ obsahovat (citlivá data jiných služeb)
    assert "AGENT_INTERNAL_FLAG" not in env
    assert "BRAVE_SEARCH_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "MY_API_KEY" not in env


def test_env_passes_anthropic_key_when_in_env(monkeypatch):
    """Pokud user má ANTHROPIC_API_KEY v env (alternativa k OAuth/keychain),
    projde do subprocess. Bridge ho nikdy NEPRIDÁVÁ - jen propustí, když je."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    env = claude_bridge._build_subprocess_env()
    assert env.get("ANTHROPIC_API_KEY") == "sk-test"


def test_env_no_anthropic_key_when_not_in_env(monkeypatch):
    """Bez klíče v env taky OK - CLI použije OAuth/keychain."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = claude_bridge._build_subprocess_env()
    assert "ANTHROPIC_API_KEY" not in env


def test_env_lc_prefix_passed(monkeypatch):
    """LC_* vars passnou (locale)."""
    monkeypatch.setenv("LC_TIME", "cs_CZ.UTF-8")
    env = claude_bridge._build_subprocess_env()
    assert env.get("LC_TIME") == "cs_CZ.UTF-8"
