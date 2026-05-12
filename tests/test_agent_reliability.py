"""Phase 8 reliability sanity suite — dedikovaná konsolidace invariantů.

Plan `plans/agent_mode.md` § Fáze 8 vyžaduje pro DoD:

  - [ ] Cancellation napříč celým loopem (agent loop, subprocess, Claude bridge)
  - [ ] max_wall_time pojistka proti runaway (default 10 min)
  - [ ] max_output_bytes per tool result (4 MiB)
  - [ ] Structured error recovery: tool error → ok=False tool_result, loop pokračuje
  - [ ] Test suite: symlink traversal, path escape, bash allowlist coverage,
        approval timeout, max_turns enforcement
  - [ ] Audit log per tool call

Jednotlivé invarianty jsou pokryté v unit testech napříč:
test_agent_sandbox.py / test_agent_fs.py / test_agent_shell.py /
test_agent_permissions.py / test_agent_audit.py / test_agent_loop.py.

Tento soubor je **konsolidační smoke** — pro každý DoD bod jedna high-signal
integration assertion proti reálnému AgentLoop / Tool / classifier stacku.
Když selže kterýkoli test tady, fáze 8 ztratila vlastnost; rychlá indikace
regrese před release. NEDUPLIKUJE granulární unit coverage, ZVALIDUJE že
celý pipeline drží invariant end-to-end.

Skupiny:
  A. Sandbox boundary  (symlink + path escape)
  B. Bash allowlist matrix
  C. Wall-time + budget enforcement
  D. Cancellation atomicity
  E. Structured error recovery
  F. Audit log forensic completeness
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from voice.agent import config, permissions
from voice.agent.audit import AuditLog
from voice.agent.loop import AgentLoop
from voice.agent.permissions import Decision, PermissionResult, decide
from voice.agent.tools import default_registry
from voice.agent.tools.base import ExecuteContext, Tool, ToolRegistry


# ---------------------------------------------------------------------------
# Helpers — mock Ollama + loop construction
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, lines: list[str]):
        self.lines = lines
        self.status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for ln in self.lines:
            yield ln

    async def aread(self) -> bytes:
        return b""


class _FakeClient:
    def __init__(self, scripts: list[list[str]]):
        self.scripts = scripts
        self.idx = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method: str, url: str, json=None):  # noqa: ARG002
        lines = self.scripts[self.idx]
        self.idx += 1
        return _FakeStream(lines)


def _ollama_line(content: str = "", tool_calls=None, done: bool = False) -> str:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return json.dumps({"message": msg, "done": done})


def _new_turn_state() -> dict:
    return {"id": "rel_t1", "canceled": False, "cancel_event": asyncio.Event()}


# ---------------------------------------------------------------------------
# Group A — Sandbox boundary
# ---------------------------------------------------------------------------


def test_reliability_path_escape_blocked_by_classifier(tmp_path: Path):
    """Path escape přes `..` mimo workdir musí NEBÝT AUTO (ASK je OK — user může
    schválit). Special files = DENY (viz separátní test). Pokrývá DoD: path escape."""
    perm = decide("write_file", {"path": "../../../etc/passwd", "content": "x"}, tmp_path)
    assert perm.decision != Decision.AUTO, "write escape must require approval"
    perm2 = decide("read_file", {"path": "/var/log/syslog"}, tmp_path)
    assert perm2.decision != Decision.AUTO, "read outside workdir must require approval"


def test_reliability_special_file_blocked(tmp_path: Path):
    """`/proc/self/environ` obsahuje secret env vars → must DENY i pro read.
    Pokrývá DoD bod: path escape (special files)."""
    perm = decide("read_file", {"path": "/proc/self/environ"}, tmp_path)
    assert perm.decision == Decision.DENY


def test_reliability_symlink_to_outside_workdir_blocked(tmp_path: Path):
    """Symlink uvnitř workdir mířící na /etc/passwd → classifier resolve
    follow-uje a containment check ho odmítne. Pokrývá: symlink traversal."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    bait = workdir / "shortcut.txt"
    bait.symlink_to("/etc/passwd")
    perm = decide("read_file", {"path": "shortcut.txt"}, workdir)
    # Buď DENY (escape) nebo ASK (mimo workdir) — kritické je, že není AUTO.
    assert perm.decision != Decision.AUTO


def test_reliability_parent_symlink_swap_blocked(tmp_path: Path):
    """Parent directory je symlink mimo workdir → resolve_safe to chytí.
    Subtilní varianta path escape."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workdir / "evil").symlink_to(outside)
    # Pokus zapsat do workdir/evil/x.txt — real path je outside/x.txt → escape.
    perm = decide("write_file", {"path": "evil/x.txt", "content": "leak"}, workdir)
    assert perm.decision != Decision.AUTO


# ---------------------------------------------------------------------------
# Group B — Bash allowlist matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    "pwd", "ls", "find .", "cat README.md",
    "wc -l README.md", "head README.md", "tail README.md", "rg foo",
])
def test_reliability_bash_known_commands_auto(cmd: str, tmp_path: Path):
    """Allowlistové root commands musí být AUTO bez approval. Pokrývá:
    bash allowlist coverage. (git/echo deliberately NOT in allowlist —
    git má RCE přes .git/config, echo není potřeba.)"""
    perm = decide("run_bash", {"command": cmd}, tmp_path)
    assert perm.decision == Decision.AUTO, f"{cmd!r} should be AUTO, got {perm.decision}"


@pytest.mark.parametrize("cmd", [
    "unknown_xyz_binary foo",
    "ls | wc -l",                # shell metaznaky → ASK
    "cat /etc/hosts > /tmp/x",   # redirect → ASK
    "echo $(date)",              # command substitution → ASK
])
def test_reliability_bash_unknown_or_shell_meta_asks(cmd: str, tmp_path: Path):
    """Cokoli mimo allowlist NEBO se shell metaznaky musí být alespoň ASK
    (nikdy AUTO). Pokrývá: bash allowlist coverage."""
    perm = decide("run_bash", {"command": cmd}, tmp_path)
    assert perm.decision == Decision.ASK, f"{cmd!r} should be ASK, got {perm.decision}"


@pytest.mark.parametrize("cmd", [
    "rm -rf /tmp/foo",
    "sudo ls",
    "chmod 777 /etc/hosts",
    "find . -delete",
    "find . -exec rm {} ;",
])
def test_reliability_bash_destructive_requires_explicit(cmd: str, tmp_path: Path):
    """Destruktivní příkazy MUSÍ vyžadovat explicit „ano povoluju" frázi.
    Pokrývá: bash allowlist coverage (destructive branch)."""
    perm = decide("run_bash", {"command": cmd}, tmp_path)
    assert perm.decision == Decision.ASK
    assert perm.requires_explicit is True, f"{cmd!r} should require explicit phrase"


# ---------------------------------------------------------------------------
# Group C — Wall-time + budget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reliability_wall_time_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """MAX_WALL_TIME_SEC pojistka proti runaway: pokud model produkuje rounds
    do nekonečna a deadline je vypršený, loop vrátí agent_error. Pokrývá:
    max_wall_time enforcement."""
    monkeypatch.setattr(config, "MAX_WALL_TIME_SEC", 0.1)
    # Model emituje tool call každé kolo → bez deadlinu by se točilo navždy.
    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": "x"}}}],
            done=True,
        )],
    ] * 50  # ne dlouho, ale dost na to aby deadline překročil

    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    # Loop konstruktor čte deadline na chvíli, donutíme ho aby už byl vypršený.
    loop.deadline = time.monotonic() - 0.01
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    types = [e["type"] for e in events]
    assert "agent_error" in types
    err = next(e for e in events if e["type"] == "agent_error")
    assert "wall-time" in err["msg"]


@pytest.mark.asyncio
async def test_reliability_max_tool_calls_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """MAX_TOOL_CALLS_PER_TURN: nad limit jsou tcs označeny jako skipped
    a synthetic tool_result je párovaný. Pokrývá: max_turns enforcement."""
    monkeypatch.setattr(config, "MAX_TOOL_CALLS_PER_TURN", 2)
    scripts = [
        [_ollama_line(
            tool_calls=[
                {"id": "tc_a", "function": {"name": "echo", "arguments": {"text": "a"}}},
                {"id": "tc_b", "function": {"name": "echo", "arguments": {"text": "b"}}},
                {"id": "tc_c", "function": {"name": "echo", "arguments": {"text": "c"}}},
            ],
            done=True,
        )],
    ]
    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    tool_results = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_results) == 3  # 2 executed + 1 synthetic
    skipped = [tr for tr in tool_results if not tr["ok"]]
    assert len(skipped) == 1
    assert "limit" in json.loads(skipped[0]["content"])["error"]
    assert events[-1]["type"] == "agent_error"


@pytest.mark.asyncio
async def test_reliability_max_model_rounds_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """MAX_MODEL_ROUNDS_PER_TURN: cap na počet LLM iterací v rámci jednoho turn.
    Pokrývá: max_turns enforcement (model rounds branch)."""
    monkeypatch.setattr(config, "MAX_MODEL_ROUNDS_PER_TURN", 2)
    # Každý round model volá tool → bez capu by se točilo navždy.
    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": str(i)}}}],
            done=True,
        )]
        for i in range(10)
    ]
    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    err = next(e for e in events if e["type"] == "agent_error")
    assert "round limit" in err["msg"]


@pytest.mark.asyncio
async def test_reliability_tool_output_cap_truncates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TOOL_OUTPUT_CAP_BYTES: extrémně velký tool output je oříznut s markerem
    `…[truncated]`. Pokrývá: max_output_bytes per tool result."""
    monkeypatch.setattr(config, "TOOL_OUTPUT_CAP_BYTES", 256)

    async def _huge_exec(args, ctx):
        return "x" * 10_000

    huge = Tool(
        name="huge_unique_for_reliability",
        description="returns huge string",
        parameters_schema={"type": "object", "properties": {}},
        execute=_huge_exec,
    )
    reg = ToolRegistry()
    reg.register(huge)

    @permissions.register_classifier("huge_unique_for_reliability")
    def _cls(args, workdir):
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "huge_unique_for_reliability", "arguments": {}}}],
                done=True,
            )],
            [_ollama_line(content="ok"), _ollama_line(done=True)],
        ]
        turn_state = _new_turn_state()
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]
        tr = next(e for e in events if e["type"] == "tool_result")
        assert tr["ok"] is True
        marker_bytes = len("\n…[truncated]".encode("utf-8"))  # = 15
        assert len(tr["content"].encode("utf-8")) <= 256 + marker_bytes
        assert "…[truncated]" in tr["content"]
    finally:
        permissions._CLASSIFIERS.pop("huge_unique_for_reliability", None)


# ---------------------------------------------------------------------------
# Group D — Cancellation atomicity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reliability_cancel_emits_canceled_event(tmp_path: Path):
    """Cancel flag → loop emituje agent_canceled, neprovede další iterace.
    Pokrývá: cancellation napříč loopem."""
    turn_state = _new_turn_state()

    async def _set_cancel_during_stream():
        await asyncio.sleep(0.05)
        turn_state["canceled"] = True

    scripts = [
        [_ollama_line(content="a"), _ollama_line(content="b"), _ollama_line(done=True)],
    ]
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    turn_state["canceled"] = True  # nejjednodušší: cancel hned na startu
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    assert events[-1]["type"] == "agent_canceled"


@pytest.mark.asyncio
async def test_reliability_bash_subprocess_killed_on_cancel(tmp_path: Path):
    """Cancel během běžícího run_bash MUSÍ zabít subprocess (a celou process group)
    rychle, ne čekat na timeout. Pokrývá: cancellation napříč subprocess.

    Spustíme `sleep 5`, nastavíme cancel_event po 200ms, ověříme že tool
    skončí do 3s (mnohem dřív než 5s naturálních) a vrátí killed=True."""
    from voice.agent.tools.shell import TOOL as bash_tool

    cancel_event = asyncio.Event()
    ctx = ExecuteContext(
        turn_id="rel_d2", cancel_event=cancel_event,
        workdir=tmp_path, resolved_path=None,
    )

    async def _cancel_after_delay():
        await asyncio.sleep(0.2)
        cancel_event.set()

    t0 = time.monotonic()
    canceler = asyncio.create_task(_cancel_after_delay())
    out = await bash_tool.execute({"command": "sleep 5"}, ctx)
    elapsed = time.monotonic() - t0
    await canceler

    # Kill grace = 2s SIGTERM→SIGKILL, takže <3s je signál „rychlý cancel".
    assert elapsed < 3.0, f"subprocess not killed promptly: {elapsed:.2f}s"
    payload = out if isinstance(out, dict) else json.loads(out)
    assert payload.get("killed") is True


# ---------------------------------------------------------------------------
# Group E — Structured error recovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reliability_tool_exception_does_not_crash_loop(tmp_path: Path):
    """Tool raises exception → loop vrátí ok=False tool_result a pokračuje
    k dalšímu modelovému roundu (nezhroutí celý turn). Pokrývá: structured
    error recovery."""

    async def _boom_exec(args, ctx):
        raise RuntimeError("simulated tool crash")

    boom = Tool(
        name="boom_unique_for_reliability",
        description="raises",
        parameters_schema={"type": "object", "properties": {}},
        execute=_boom_exec,
    )
    reg = ToolRegistry()
    reg.register(boom)

    @permissions.register_classifier("boom_unique_for_reliability")
    def _cls(args, workdir):
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "boom_unique_for_reliability", "arguments": {}}}],
                done=True,
            )],
            [_ollama_line(content="recovered"), _ollama_line(done=True)],
        ]
        turn_state = _new_turn_state()
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]

        types = [e["type"] for e in events]
        assert "tool_result" in types
        tr = next(e for e in events if e["type"] == "tool_result")
        assert tr["ok"] is False
        assert "RuntimeError" in json.loads(tr["content"])["error"]
        # Loop pokračoval k dalšímu round + final text.
        assert types[-1] == "agent_done"
        text = "".join(e["delta"] for e in events if e["type"] == "text")
        assert "recovered" in text
    finally:
        permissions._CLASSIFIERS.pop("boom_unique_for_reliability", None)


@pytest.mark.asyncio
async def test_reliability_unknown_tool_does_not_crash(tmp_path: Path):
    """Model halucinuje název toolu → ok=False tool_result, loop pokračuje.
    Pokrývá: structured error recovery (unknown tool branch)."""
    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "totally_made_up_tool", "arguments": {}}}],
            done=True,
        )],
        [_ollama_line(content="oh well"), _ollama_line(done=True)],
    ]
    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False
    assert events[-1]["type"] == "agent_done"


# ---------------------------------------------------------------------------
# Group F — Audit log forensic completeness
# ---------------------------------------------------------------------------


def _read_audit_records(audit_dir: Path) -> list[dict]:
    files = list(audit_dir.glob("*.jsonl"))
    if not files:
        return []
    return [json.loads(ln) for ln in files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.mark.asyncio
async def test_reliability_audit_records_auto_path(tmp_path: Path):
    """AUTO tool call → audit record má decision=auto, approval=None, ok=True.
    Pokrývá: audit log per tool call (AUTO branch)."""
    audit_dir = tmp_path / "audit"
    audit_log = AuditLog(audit_dir)

    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": "hi"}}}],
            done=True,
        )],
        [_ollama_line(content="done"), _ollama_line(done=True)],
    ]
    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state,
        workdir=tmp_path, audit_log=audit_log,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        async for _ in loop.run():
            pass

    # audit task je fire-and-forget shielded — počkáme krátce na drain.
    await asyncio.sleep(0.05)
    records = _read_audit_records(audit_dir)
    assert len(records) == 1
    r = records[0]
    assert r["tool"] == "echo"
    assert r["permission"]["decision"] == "auto"
    assert r["approval"] is None
    assert r["ok"] is True


@pytest.mark.asyncio
async def test_reliability_audit_records_ask_denied(tmp_path: Path):
    """ASK tool + user denial → audit record má approval=denied, ok=False.
    Pokrývá: audit log per tool call (denial branch)."""
    audit_dir = tmp_path / "audit"
    audit_log = AuditLog(audit_dir)

    # Force echo do ASK větve.
    original_cls = permissions._CLASSIFIERS.get("echo")
    permissions._CLASSIFIERS["echo"] = lambda args, wd: PermissionResult(
        decision=Decision.ASK, reason="forced ask",
        summary="echo (test)", risk="medium",
    )
    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "echo", "arguments": {"text": "x"}}}],
                done=True,
            )],
            [_ollama_line(content="ok"), _ollama_line(done=True)],
        ]
        turn_state = _new_turn_state()
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=default_registry("agent"), turn_state=turn_state,
            workdir=tmp_path, audit_log=audit_log,
        )

        async def _deny(approval_id, event):
            return False

        loop.set_approval_resolver(_deny)
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            async for _ in loop.run():
                pass
        await asyncio.sleep(0.05)
    finally:
        if original_cls is None:
            permissions._CLASSIFIERS.pop("echo", None)
        else:
            permissions._CLASSIFIERS["echo"] = original_cls

    records = _read_audit_records(audit_dir)
    assert len(records) >= 1
    echo_rec = next(r for r in records if r["tool"] == "echo")
    assert echo_rec["permission"]["decision"] == "ask"
    assert echo_rec["approval"] == "denied"
    assert echo_rec["ok"] is False


@pytest.mark.asyncio
async def test_reliability_audit_records_dropped_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Tool calls dropped přes MAX_TOOL_CALLS_PER_TURN MUSÍ být v auditu jako
    skipped_limit (jinak by LLM mohl schovat malicious call za rate-limit).
    Pokrývá: audit log per tool call (rate-limit skipped branch)."""
    monkeypatch.setattr(config, "MAX_TOOL_CALLS_PER_TURN", 1)
    audit_dir = tmp_path / "audit"
    audit_log = AuditLog(audit_dir)

    scripts = [
        [_ollama_line(
            tool_calls=[
                {"id": "tc_a", "function": {"name": "echo", "arguments": {"text": "a"}}},
                {"id": "tc_b", "function": {"name": "echo", "arguments": {"text": "b"}}},
            ],
            done=True,
        )],
    ]
    turn_state = _new_turn_state()
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state,
        workdir=tmp_path, audit_log=audit_log,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        async for _ in loop.run():
            pass
    await asyncio.sleep(0.05)

    records = _read_audit_records(audit_dir)
    assert len(records) == 2, f"expected 2 records (1 executed + 1 skipped), got {len(records)}"
    decisions = sorted(r["permission"]["decision"] for r in records)
    assert "auto" in decisions
    assert "skipped_limit" in decisions
    skipped = next(r for r in records if r["permission"]["decision"] == "skipped_limit")
    assert skipped["ok"] is False
    assert "limit" in skipped["error"]
