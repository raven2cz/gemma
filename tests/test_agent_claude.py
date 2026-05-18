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
    """Async StreamReader fake - `await read(n)` vrací postupně dané chunky.
    `await readline()` vrací postupně řádky (rozdělené `\n` nebo celý chunk)."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._closed = False
        # Pro readline: spojené bytes přerozdělené na řádky.
        self._lines = self._chunks_to_lines(chunks)

    @staticmethod
    def _chunks_to_lines(chunks: list[bytes]) -> list[bytes]:
        joined = b"".join(chunks)
        if not joined:
            return []
        # Zachovat \n na konci každého řádku (jak readline vrací).
        lines: list[bytes] = []
        start = 0
        for i, b in enumerate(joined):
            if b == 0x0A:  # \n
                lines.append(joined[start:i+1])
                start = i + 1
        if start < len(joined):
            lines.append(joined[start:])
        return lines

    async def read(self, n: int = -1) -> bytes:
        if self._closed or not self._chunks:
            self._closed = True
            return b""
        chunk = self._chunks.pop(0)
        if n < 0 or len(chunk) <= n:
            return chunk
        # Codex audit fix: partial read by neměl zahodit zbytek. Vrať prefix,
        # zbytek vrať na začátek queue pro další read().
        self._chunks.insert(0, chunk[n:])
        return chunk[:n]

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


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
        captured["limit"] = kwargs.get("limit")  # readline overflow guard
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
    """Tool má správné jméno + schema kontrakt (v2: mode + model, ne max_tokens)."""
    assert ASK_CLAUDE_TOOL.name == "ask_claude"
    assert "Claude Code CLI subprocess" in ASK_CLAUDE_TOOL.description
    schema = ASK_CLAUDE_TOOL.parameters_schema
    assert "prompt" in schema["required"]
    assert "prompt" in schema["properties"]
    assert "system" in schema["properties"]
    assert "mode" in schema["properties"]
    assert schema["properties"]["mode"]["enum"] == ["consult", "edit"]
    assert "model" in schema["properties"]
    assert set(schema["properties"]["model"]["enum"]) == {"opus", "sonnet", "haiku"}


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
async def test_unknown_mode_rejected():
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "mode": "destroy"}, _make_ctx())
    assert r["ok"] is False
    assert "unknown mode" in r["error"]


@pytest.mark.asyncio
async def test_mode_blank_string_rejected():
    """Codex iter-8: blank/whitespace mode = error (NE default consult).
    Classifier i execute MUSÍ mít stejnou normalizaci."""
    for blank in ("", "   ", "\t\n"):
        r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "mode": blank}, _make_ctx())
        assert r["ok"] is False
        assert "empty" in r["error"], f"blank={blank!r}: {r}"


@pytest.mark.asyncio
async def test_mode_non_string_rejected():
    """Codex iter-4 HIGH: bool/int v `mode` args nesmí shodit tool na
    AttributeError. Vrátit controlled ok=False."""
    for bad in (True, False, 42, [], {"x": 1}):
        r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "mode": bad}, _make_ctx())
        assert r["ok"] is False
        assert "mode must be string" in r["error"], f"bad={bad!r}: {r}"


@pytest.mark.asyncio
async def test_model_non_string_rejected():
    """Codex iter-4 HIGH: stejně pro `model` arg."""
    for bad in (True, 1, [], {}):
        r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "model": bad}, _make_ctx())
        assert r["ok"] is False
        assert "model must be string" in r["error"], f"bad={bad!r}: {r}"


@pytest.mark.asyncio
async def test_model_unknown_string_rejected():
    """Codex iter-9 MEDIUM: random model string mimo allowlist → reject."""
    for bad in ("gpt-4", "gemini-pro", "claude-foo", "x"):
        r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "model": bad}, _make_ctx())
        assert r["ok"] is False
        assert "allowlist" in r["error"], f"bad={bad!r}: {r}"


@pytest.mark.asyncio
async def test_model_full_claude_name_accepted(monkeypatch):
    """Plné claude-* jméno (claude-opus-4-7) musí projít stejně jako alias."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "hi", "model": "claude-opus-4-7"}, _make_ctx(),
    )
    assert r["ok"] is True


def test_env_excludes_pwd(monkeypatch):
    """Codex iter-9 MEDIUM: PWD z parent env NESMÍ propadnout (Claude CLI by
    ho mohl použít místo getcwd() a najít project mimo náš sandbox cwd)."""
    monkeypatch.setenv("PWD", "/some/external/project")
    from voice.agent import claude_bridge
    # Bez argumentu - žádný PWD set
    env = claude_bridge._build_subprocess_env()
    assert "PWD" not in env, "PWD parent env MUSÍ být dropped"


def test_env_sets_pwd_to_cwd(monkeypatch):
    """Při explicit cwd argumentu se PWD nastaví na něj (ne parent)."""
    monkeypatch.setenv("PWD", "/some/external/project")
    from voice.agent import claude_bridge
    env = claude_bridge._build_subprocess_env(cwd="/sandbox/tmp")
    assert env["PWD"] == "/sandbox/tmp", "PWD MUSÍ být cwd, ne parent env"


@pytest.mark.asyncio
async def test_edit_requires_workdir():
    """mode=edit potřebuje ctx.workdir. ExecuteContext v testu má /tmp default."""
    # mode=edit s workdir OK pre-validation (subprocess fail-fast bez claude bin
    # je další path). Pojďme assert že mode rejected pokud ctx.workdir je None
    # - patchnu ctx.
    from pathlib import Path
    bad_ctx = ExecuteContext(
        turn_id="t", cancel_event=None, workdir=None, resolved_path=None,
    )
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "fix bug", "mode": "edit"}, bad_ctx)
    assert r["ok"] is False
    assert "workdir" in r["error"]


# ───────────────────────── Bridge - happy path ─────────────────────────


@pytest.mark.asyncio
async def test_happy_path_consult(monkeypatch):
    """consult mode: stream-json sekvence → ok=True + finální text."""
    # Realistický stream: init → assistant text → result
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "abc-123", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "assistant", "message": {
            "content": [{"type": "text", "text": "Hello from Claude."}]
        }}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "Hello from Claude.",
                    "session_id": "abc-123",
                    "total_cost_usd": 0.0012,
                    "is_error": False,
                    "model": "claude-opus-4-7"}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode("utf-8") for s in stream_lines],
        stderr_chunks=[],
        returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "Hi Claude"}, _make_ctx())

    assert r["ok"] is True, f"failed: {r}"
    assert r["text"] == "Hello from Claude."
    assert r["session_id"] == "abc-123"
    assert r["total_cost_usd"] == 0.0012
    assert r["mode"] == "consult"
    assert r["duration_ms"] >= 0
    # Argv security: prompt MUSÍ NE být v argv (`ps` leak)
    argv_joined = " ".join(captured["argv"])
    assert "Hi Claude" not in argv_joined, "prompt leak v argv!"
    # consult mode flagy
    assert "--no-session-persistence" in captured["argv"]
    assert "--permission-mode" in captured["argv"]
    assert "plan" in captured["argv"]  # consult = plan permission
    assert "--tools" in captured["argv"]
    assert "" in captured["argv"]  # --tools "" (žádné tools)
    assert "--output-format" in captured["argv"]
    assert "stream-json" in captured["argv"]
    assert "--include-partial-messages" in captured["argv"]
    assert captured["start_new_session"] is True
    # Prompt přes stdin
    assert proc.stdin.buffer == b"Hi Claude"
    # consult: empty temp cwd, NE workdir
    assert "claude_bridge_" in captured["cwd"]


@pytest.mark.asyncio
async def test_happy_path_edit(monkeypatch, tmp_path):
    """edit mode: argv obsahuje --add-dir + tools + acceptEdits, cwd=workdir."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "xyz", "model": "claude-sonnet-4-6"}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_start",
            "content_block": {"type": "tool_use", "name": "Write",
                              "input": {"file_path": "hello.txt"}}
        }}) + "\n",
        json.dumps({"type": "assistant", "message": {
            "content": [{"type": "text", "text": "Done."}]
        }}) + "\n",
        json.dumps({"type": "result", "subtype": "success", "result": "Done.",
                    "is_error": False}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[],
        returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    ctx = ExecuteContext(
        turn_id="t", cancel_event=None, workdir=tmp_path, resolved_path=None,
    )
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "Create hello.txt", "mode": "edit"}, ctx,
    )

    assert r["ok"] is True, f"failed: {r}"
    assert r["mode"] == "edit"
    assert r["tool_uses"] == ["Write"]
    # edit mode flagy
    assert "--permission-mode" in captured["argv"]
    assert "acceptEdits" in captured["argv"]
    assert "--add-dir" in captured["argv"]
    assert str(tmp_path) in captured["argv"]
    # --tools subset (NE prázdný string)
    tools_idx = captured["argv"].index("--tools")
    tools_val = captured["argv"][tools_idx + 1]
    assert "Edit" in tools_val and "Write" in tools_val
    # cwd = workdir
    assert captured["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_progress_callback_dispatches(monkeypatch, tmp_path):
    """progress_emitter z ctx dostane payloady pro každý dispatched stream event."""
    # Realistic stream: content_block_start (name only, empty input) →
    # content_block_delta (incremental input_json_delta) → content_block_stop
    # (emit s parsed inputem). Bridge musí poskládat partial_json kousky.
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s1", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "thinking"}
        }}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "name": "Read", "input": {}}
        }}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta",
                      "partial_json": '{"file_path":"'},
        }}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta",
                      "partial_json": 'x.py"}'},
        }}) + "\n",
        json.dumps({"type": "stream_event", "event": {
            "type": "content_block_stop", "index": 1,
        }}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    emitted: list[dict] = []
    async def collector(payload):
        emitted.append(payload)

    ctx = ExecuteContext(
        turn_id="t", cancel_event=None, workdir=tmp_path, resolved_path=None,
        progress_emitter=collector,
    )
    r = await ASK_CLAUDE_TOOL.execute(
        {"prompt": "Read x.py", "mode": "edit"}, ctx,
    )
    assert r["ok"] is True
    stages = [p["stage"] for p in emitted]
    assert "started" in stages
    assert "thinking" in stages
    assert "tool_use" in stages
    # tool_use payload má tool_name + enrichnutý message s file_path (jinak
    # by UI ukázalo jen "Read" bez detailu).
    tu = next(p for p in emitted if p["stage"] == "tool_use")
    assert tu["tool_name"] == "Read"
    assert "x.py" in tu["message"], (
        f"tool_use message neobsahuje file_path: {tu}"
    )
    assert tu.get("input") == {"file_path": "x.py"}


@pytest.mark.asyncio
async def test_model_hint_from_ctx(monkeypatch, tmp_path):
    """Pokud args nemají model, použij ctx.model_hint (router decision)."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-sonnet-4-6"}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False,
                    "model": "claude-sonnet-4-6"}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    ctx = ExecuteContext(
        turn_id="t", cancel_event=None, workdir=tmp_path, resolved_path=None,
        model_hint="sonnet",
    )
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, ctx)
    assert r["ok"] is True
    # Argv obsahuje resolved sonnet alias
    assert "claude-sonnet-4-6" in captured["argv"]


@pytest.mark.asyncio
async def test_args_model_overrides_ctx_hint(monkeypatch, tmp_path):
    """args.model má prioritu nad ctx.model_hint."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-haiku-4-5"}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    ctx = ExecuteContext(
        turn_id="t", cancel_event=None, workdir=tmp_path, resolved_path=None,
        model_hint="sonnet",
    )
    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi", "model": "haiku"}, ctx)
    assert r["ok"] is True
    assert "claude-haiku-4-5" in captured["argv"]
    assert "claude-sonnet-4-6" not in captured["argv"]


@pytest.mark.asyncio
async def test_system_via_argv(monkeypatch):
    """System prompt jde přes --append-system-prompt argv (CLI 2.1.x nemá file flag)."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False}) + "\n",
    ]
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[],
    )
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
async def test_oversized_line_skipped_continues_reading():
    """User policy: bridge nemá hard kill cap. Pokud jeden řádek přesáhne
    asyncio StreamReader limit, _read_stream ho SKIPNE a pokračuje dalším
    řádkem - process běží dál, finální result event se chytí, files psané
    do té doby zůstanou.

    Před změnou (output cap mechanism): overflow_event → killpg → ok=False.
    User said: 'ten cap si nejsem jist, ze je uplne dobry napad' → odstraněn.
    """
    from voice.agent.claude_bridge import _read_stream

    # Real StreamReader s malým per-line limitem (1024 B)
    reader = asyncio.StreamReader(limit=1024)
    big_line = b"X" * 2048 + b"\n"  # překročí per-line limit
    # Po něm validní result event - bridge ho má najít
    good_result = json.dumps({
        "type": "result", "subtype": "success",
        "result": "completed", "is_error": False,
    }).encode() + b"\n"
    reader.feed_data(big_line)
    reader.feed_data(good_result)
    reader.feed_eof()

    class _FakeProc:
        pid = 99999
        stdout = reader

    state = {"assistant_text": "", "tool_uses": []}
    result = await _read_stream(_FakeProc(), progress_cb=None, state=state)
    # Result MUSÍ být chycený přestože jeden řádek se přeskočil
    assert result is not None, "oversized line should not abort reading"
    assert result.get("type") == "result"
    assert result.get("result") == "completed"
    # Žádný overflow state - mechanism odstraněn
    assert not state.get("overflow")


@pytest.mark.asyncio
async def test_parser_skips_nested_non_dict():
    """Codex iter-6: nested non-dict v stream_event/assistant/user → skip,
    ne AttributeError crash. Malicious/buggy CLI by mohl emit
    `{"type":"stream_event","event":[]}` apod."""
    from voice.agent.claude_bridge import _parse_stream_event
    # event jako list místo dict
    state = {"assistant_text": "", "tool_uses": []}
    assert _parse_stream_event(
        {"type": "stream_event", "event": [1, 2]}, state
    ) is None
    # assistant.message jako string místo dict
    assert _parse_stream_event(
        {"type": "assistant", "message": "should be dict"}, state
    ) is None
    # assistant.message.content jako string místo list
    assert _parse_stream_event(
        {"type": "assistant", "message": {"content": "bad"}}, state
    ) is None
    # user.message jako int
    assert _parse_stream_event(
        {"type": "user", "message": 42}, state
    ) is None
    # content_block jako list
    assert _parse_stream_event(
        {"type": "stream_event", "event": {
            "type": "content_block_start", "content_block": [1, 2]
        }}, state,
    ) is None


@pytest.mark.asyncio
async def test_parser_skips_non_dict_json():
    """Codex iter-5 MEDIUM: stream-json line co je validní JSON ale ne dict
    (např. `[1,2,3]` nebo `"string"`) MUSÍ být skipnutá, ne hodit AttributeError."""
    from voice.agent.claude_bridge import _read_stream

    reader = asyncio.StreamReader(limit=128 * 1024)
    # Mix: non-dict líny (skip), valid system init (dispatch), result (terminal)
    lines = [
        b'[1, 2, 3]\n',
        b'"just a string"\n',
        b'42\n',
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "m"}).encode() + b"\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": "ok", "is_error": False}).encode() + b"\n",
    ]
    for ln in lines:
        reader.feed_data(ln)
    reader.feed_eof()

    class _FakeProc:
        pid = 99998
        stdout = reader

    state = {"assistant_text": "", "tool_uses": []}
    result = await _read_stream(_FakeProc(), progress_cb=None, state=state)
    assert result is not None
    assert result.get("type") == "result"
    assert state.get("session_id") == "s", "init event MUSÍ dispatched"


@pytest.mark.asyncio
async def test_large_result_line_within_cap(monkeypatch):
    """Codex iter-3 HIGH: stream-json result line může být > 64 KiB (default
    asyncio readline limit). Bridge nastavuje limit na output_cap+16K, aby
    legitimní velký result prošel. Bez fixu by readline raised ValueError
    a vrátili bychom 'no result event'."""
    big_text = "X" * (100 * 1024)  # 100 KiB text v result
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "result", "subtype": "success",
                    "result": big_text, "is_error": False}) + "\n",
    ]
    # output_cap musí být > big_text + JSON overhead
    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    captured = _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "give me a lot"}, _make_ctx())
    assert r["ok"] is True, f"big result failed: {r}"
    assert len(r["text"]) == 100 * 1024
    # Codex iter-4: ověř že bridge nastavil `limit=` v create_subprocess_exec
    # dostatečně vysoký aby asyncio readline nepadl na default 64KiB limit.
    # Po odstranění output capu používáme fixní 32 MiB per-line limit.
    captured_limit = captured.get("limit")
    assert captured_limit is not None and captured_limit >= 16 * 1024 * 1024, (
        f"per-line stream limit moc nízký: {captured_limit}"
    )


@pytest.mark.asyncio
async def test_no_result_event_is_protocol_error(monkeypatch):
    """Codex audit HIGH: bridge MUSÍ vrátit ok=False pokud stream skončil bez
    result eventu. Mlčení by maskovalo invalid JSON / CLI drift / partial
    výsledek v edit módu."""
    proc = _FakeProcess(
        stdout_chunks=[b"not valid json {{\n", b"another bad line\n"],
        stderr_chunks=[],
        returncode=0,
    )
    _patch_subprocess(monkeypatch, proc)
    _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "hi"}, _make_ctx())
    assert r["ok"] is False
    assert "no result event" in r["error"]


@pytest.mark.asyncio
async def test_is_error_flag(monkeypatch):
    """result event s is_error=True → ok=False s message z result."""
    stream_lines = [
        json.dumps({"type": "system", "subtype": "init",
                    "session_id": "s", "model": "claude-opus-4-7"}) + "\n",
        json.dumps({"type": "result", "subtype": "error_quota",
                    "result": "API quota exceeded",
                    "is_error": True}) + "\n",
    ]
    proc = _FakeProcess(stdout_chunks=[s.encode() for s in stream_lines],
                        stderr_chunks=[])
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
    assert r.get("timeout") is True, f"timeout flag missing: {r}"
    assert "nestihl odpovědět" in r["error"] or "timeout" in r["error"]
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
async def test_no_output_cap_large_stream_completes(monkeypatch):
    """User policy: bridge nemá total-bytes cap. Velký stream prochází bez
    overflow killu, dokud subprocess sám doběhne (result event). Reálný
    Opus implementation task generuje 10-50 MB stream-json - dřívější
    256 KiB resp. 16 MiB cap byl pro to nedostatečný.

    Test: 10 MB stream nepřeruší zpracování, ok=True, result event chycený.
    """
    # 50 řádků × 200 KiB = ~10 MB total stream
    big_text = "Y" * (200 * 1024)
    stream_lines = []
    for _ in range(50):
        # Každý řádek je incremental text delta event
        stream_lines.append(json.dumps({
            "type": "stream_event",
            "event": {"type": "content_block_delta",
                      "delta": {"type": "text_delta", "text": big_text}},
        }) + "\n")
    # Final result event
    stream_lines.append(json.dumps({
        "type": "result", "subtype": "success",
        "result": "completed huge task", "is_error": False,
    }) + "\n")

    proc = _FakeProcess(
        stdout_chunks=[s.encode() for s in stream_lines],
        stderr_chunks=[], returncode=0,
    )
    _patch_subprocess(monkeypatch, proc)
    killpg_calls = _patch_killpg(monkeypatch)

    r = await ASK_CLAUDE_TOOL.execute({"prompt": "huge"}, _make_ctx())
    # Žádný overflow kill - process doběhl normálně
    assert r["ok"] is True, f"large stream selhal: {r}"
    assert r["text"] == "completed huge task"
    assert r.get("overflow") is not True
    # killpg se NEMĚL volat (žádný cap-based kill)
    assert not any(c[0] == "killpg" for c in killpg_calls), (
        f"unexpected killpg call: {killpg_calls}"
    )


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
