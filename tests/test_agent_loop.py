"""Unit testy pro agent loop. Ollama je mockovaný — žádné network calls."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice.agent.loop import AgentLoop
from voice.agent.tools import default_registry
from voice.agent.tools.base import Tool, ToolRegistry


class _FakeStream:
    """Fake httpx.AsyncClient.stream() context manager. Emituje NDJSON
    lines podle scriptu."""

    def __init__(self, lines: list[str], status_code: int = 200):
        self.lines = lines
        self.status_code = status_code

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def aiter_lines(self):
        for ln in self.lines:
            yield ln

    async def aread(self) -> bytes:
        return "\n".join(self.lines).encode()


class _FakeClient:
    def __init__(self, scripts: list[list[str]]):
        # scripts[i] = NDJSON lines pro i-tý request
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


@pytest.mark.asyncio
async def test_loop_text_only_no_tools(tmp_path: Path):
    """Model vrátí čistý text bez tool_calls → loop emituje text + agent_done."""
    scripts = [[
        _ollama_line(content="Ahoj "),
        _ollama_line(content="světe!"),
        _ollama_line(done=True),
    ]]

    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "ahoj"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    text = "".join(e["delta"] for e in events if e["type"] == "text")
    assert text == "Ahoj světe!"
    assert events[-1]["type"] == "agent_done"
    # Žádný tool call ani approval
    assert not any(e["type"] in ("tool_call", "tool_result", "approval_required") for e in events)


@pytest.mark.asyncio
async def test_loop_echo_tool_call(tmp_path: Path):
    """Model volá echo, dostane result, pak doplní finalní text."""
    scripts = [
        # Turn 1: tool call
        [
            _ollama_line(
                tool_calls=[{
                    "function": {"name": "echo", "arguments": {"text": "hi"}},
                }],
                done=True,
            ),
        ],
        # Turn 2: final text po tool result
        [
            _ollama_line(content="Hotovo: "),
            _ollama_line(content="hi"),
            _ollama_line(done=True),
        ],
    ]

    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "echo hi"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    types = [e["type"] for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "agent_done"

    tc_event = next(e for e in events if e["type"] == "tool_call")
    assert tc_event["name"] == "echo"
    assert tc_event["args"] == {"text": "hi"}

    tr_event = next(e for e in events if e["type"] == "tool_result")
    assert tr_event["ok"] is True
    result_payload = json.loads(tr_event["content"])
    assert result_payload == {"echoed": "hi", "length": 2}

    # Final text byl emitován v druhé iteraci
    text = "".join(e["delta"] for e in events if e["type"] == "text")
    assert text == "Hotovo: hi"

    # History musí obsahovat assistant s tool_calls + tool message + final assistant
    history = loop.messages
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "tool_calls" in history[1]
    assert history[2]["tool_call_id"] == tc_event["id"]


@pytest.mark.asyncio
async def test_loop_unknown_tool_denied(tmp_path: Path):
    """Model vrátí neznámý tool → loop ho zamítne, model dostane error
    a v druhé iteraci dokončí turn textem."""
    scripts = [
        [
            _ollama_line(
                tool_calls=[{"function": {"name": "no_such_tool", "arguments": {}}}],
                done=True,
            ),
        ],
        [
            _ollama_line(content="Ups, nemůžu."),
            _ollama_line(done=True),
        ],
    ]
    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False
    payload = json.loads(tr["content"])
    assert "denied" in payload["error"] or "policy" in payload["error"]


@pytest.mark.asyncio
async def test_loop_cancellation(tmp_path: Path):
    """Cancel uprostřed streamu → agent_canceled, žádné další iterace."""
    cancel_event = asyncio.Event()
    turn_state = {"id": "t1", "canceled": False, "cancel_event": cancel_event}

    async def slow_lines():
        yield _ollama_line(content="zacatek ")
        turn_state["canceled"] = True
        yield _ollama_line(content="nedoraz")
        yield _ollama_line(done=True)

    class _SlowStream:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def aiter_lines(self):
            async for ln in slow_lines():
                yield ln
        async def aread(self):
            return b""
        status_code = 200

    class _SlowClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def stream(self, *a, **kw):
            return _SlowStream()

    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "ahoj"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_SlowClient()):
        events = [ev async for ev in loop.run()]

    types = [e["type"] for e in events]
    assert "agent_canceled" in types


@pytest.mark.asyncio
async def test_loop_ollama_error(tmp_path: Path):
    """Ollama HTTP 500 → agent_error event."""
    class _ErrStream:
        status_code = 500
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        async def aiter_lines(self):
            return
            yield  # pragma: no cover
        async def aread(self):
            return b"server boom"

    class _ErrClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *a):
            return False
        def stream(self, *a, **kw):
            return _ErrStream()

    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "ahoj"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_ErrClient()):
        events = [ev async for ev in loop.run()]

    err = [e for e in events if e["type"] == "agent_error"]
    assert err
    assert "500" in err[0]["msg"]


@pytest.mark.asyncio
async def test_loop_non_serializable_tool_output_degrades(tmp_path: Path):
    """Tool vrátí ne-JSON-serializovatelný objekt (např. Path).
    Loop nesmí spadnout — musí vrátit ok=False tool_result a pokračovat."""
    from voice.agent import permissions

    async def _bad_exec(args, ctx):
        return {"path": Path("/tmp")}  # Path není JSON-serializable

    bad_tool = Tool(
        name="bad_unique_for_test",
        description="returns non-serializable",
        parameters_schema={"type": "object", "properties": {}},
        execute=_bad_exec,
    )
    reg = ToolRegistry()
    reg.register(bad_tool)

    # AUTO classifier ať tool projde permission checkem.
    @permissions.register_classifier("bad_unique_for_test")
    def _cls(args, workdir):
        return permissions.PermissionResult(
            decision=permissions.Decision.AUTO, reason="test", summary="test",
            risk="low", requires_explicit=False,
        )

    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "bad_unique_for_test", "arguments": {}}}],
            done=True,
        )],
        [_ollama_line(content="ok"), _ollama_line(done=True)],
    ]
    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "go"}],
        registry=reg,
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    types = [e["type"] for e in events]
    assert types[-1] == "agent_done"
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False
    payload = json.loads(tr["content"])
    assert "non-serializable" in payload["error"]


@pytest.mark.asyncio
async def test_loop_tool_call_cap_emits_synthetic_results(tmp_path: Path, monkeypatch):
    """Při překročení MAX_TOOL_CALLS_PER_TURN musí pro dropped tool calls
    vzniknout párový tool_result (jinak by LLM history byla malformed)."""
    from voice.agent import config as cfg
    monkeypatch.setattr(cfg, "MAX_TOOL_CALLS_PER_TURN", 1)

    # Model emituje 3 echo tool calls v jednom kole.
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
    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "echo"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    tool_results = [e for e in events if e["type"] == "tool_result"]
    # 1 executed + 2 synthetic skipped — všechny 3 mají párový tool_result.
    assert len(tool_results) == 3
    ok_count = sum(1 for tr in tool_results if tr["ok"])
    assert ok_count == 1
    skipped = [tr for tr in tool_results if not tr["ok"]]
    for tr in skipped:
        payload = json.loads(tr["content"])
        assert "limit" in payload["error"]

    # History: assistant.tool_calls má 3 položky, následují 3 tool messages,
    # poslední event je agent_error (cap přesažen).
    history = loop.messages
    asst = next(m for m in history if m["role"] == "assistant" and "tool_calls" in m)
    assert len(asst["tool_calls"]) == 3
    tool_msgs = [m for m in history if m["role"] == "tool"]
    assert len(tool_msgs) == 3
    assert events[-1]["type"] == "agent_error"
    assert "tool-call limit" in events[-1]["msg"]


# ----------------------------------------------------------------------
# Fáze 9.3 — Audio filler watchdog
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_audio_filler_emitted_for_slow_tool(tmp_path: Path, monkeypatch):
    """Tool, který trvá déle než AUDIO_FILLER_DELAY_SEC, musí způsobit emit
    `audio_filler` event před `tool_result`."""
    from voice.agent import config as cfg
    monkeypatch.setattr(cfg, "AUDIO_FILLER_DELAY_SEC", 0.05)

    async def _slow_exec(args, ctx):
        await asyncio.sleep(0.2)
        return "done"

    from voice.agent import permissions as perm
    from voice.agent.permissions import Decision, PermissionResult, register_classifier

    slow = Tool(
        name="slow_unique_for_filler",
        description="sleeps",
        parameters_schema={"type": "object", "properties": {}},
        execute=_slow_exec,
    )
    reg = ToolRegistry()
    reg.register(slow)

    @register_classifier("slow_unique_for_filler")
    def _cls(args, workdir):
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "slow_unique_for_filler", "arguments": {}}}],
                done=True,
            )],
            [_ollama_line(content="ok"), _ollama_line(done=True)],
        ]
        turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]
        types = [e["type"] for e in events]
        # audio_filler musí být PŘED tool_result (jinak je k ničemu)
        ai_idx = types.index("audio_filler")
        tr_idx = types.index("tool_result")
        assert ai_idx < tr_idx
        # payload
        af = events[ai_idx]
        assert af["tool"] == "slow_unique_for_filler"
        assert af["tool_call_id"]
    finally:
        perm._CLASSIFIERS.pop("slow_unique_for_filler", None)


@pytest.mark.asyncio
async def test_loop_audio_filler_skipped_for_fast_tool(tmp_path: Path, monkeypatch):
    """Rychlý tool (< AUDIO_FILLER_DELAY_SEC) NESMÍ emit `audio_filler`."""
    from voice.agent import config as cfg
    monkeypatch.setattr(cfg, "AUDIO_FILLER_DELAY_SEC", 0.5)

    scripts = [
        [_ollama_line(
            tool_calls=[{"function": {"name": "echo", "arguments": {"text": "hi"}}}],
            done=True,
        )],
        [_ollama_line(content="ok"), _ollama_line(done=True)],
    ]
    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test", messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"), turn_state=turn_state, workdir=tmp_path,
    )
    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]
    types = [e["type"] for e in events]
    assert "audio_filler" not in types, f"unexpected filler in {types}"


@pytest.mark.asyncio
async def test_loop_audio_filler_no_race_against_finalize(tmp_path: Path, monkeypatch):
    """Regression (Gemini iter-1): pokud tool doběhne těsně PŘED delay vypršením,
    ale finalize_tool_call dělá I/O await (audit log shield, queue.put), starý
    kód (cancel ve vnějším finally) by mohl emit audio_filler PO tool_result.

    Setup: tool sleep 0.1s, delay 0.15s → tool finishes BEFORE delay. Pak v
    cestě k tool_result vložíme umělé `asyncio.sleep(0.2)` (simulace pomalého
    audit logu) — pokud by filler_task nebyl cancelován ihned po execute(),
    stihne v této pauze emitovat. Po fixu musí cancel proběhnout PŘED tímto
    artificialním sleep.
    """
    from voice.agent import config as cfg
    from voice.agent import permissions as perm
    from voice.agent.permissions import Decision, PermissionResult, register_classifier
    from voice.agent.loop import AgentLoop

    monkeypatch.setattr(cfg, "AUDIO_FILLER_DELAY_SEC", 0.15)

    async def _quick_exec(args, ctx):
        await asyncio.sleep(0.1)  # finish PŘED delay vypršením
        return "done"

    quick = Tool(
        name="quick_race_unique",
        description="quick",
        parameters_schema={"type": "object", "properties": {}},
        execute=_quick_exec,
    )
    reg = ToolRegistry()
    reg.register(quick)

    @register_classifier("quick_race_unique")
    def _cls(args, workdir):
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    # Monkeypatch _finalize_tool_call → vlož umělý sleep mezi return execute
    # a queue.put. Po fixu je filler_task už cancelován; před fixem by stihl emit.
    orig_finalize = AgentLoop._finalize_tool_call

    async def _slow_finalize(self, **kw):
        await asyncio.sleep(0.2)
        return await orig_finalize(self, **kw)

    monkeypatch.setattr(AgentLoop, "_finalize_tool_call", _slow_finalize)

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "quick_race_unique", "arguments": {}}}],
                done=True,
            )],
            [_ollama_line(content="ok"), _ollama_line(done=True)],
        ]
        turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]
        types = [e["type"] for e in events]
        # Tool finished BEFORE delay → filler NESMÍ být emitován ani po
        # I/O pauze v finalize. (Bez fixu: filler by se mezitím probudil.)
        assert "audio_filler" not in types, (
            f"filler emitted out-of-order after tool_result: {types}"
        )
    finally:
        perm._CLASSIFIERS.pop("quick_race_unique", None)


@pytest.mark.asyncio
async def test_loop_audio_filler_disabled_when_delay_zero(tmp_path: Path, monkeypatch):
    """AUDIO_FILLER_DELAY_SEC = 0 → filler completely disabled."""
    from voice.agent import config as cfg
    monkeypatch.setattr(cfg, "AUDIO_FILLER_DELAY_SEC", 0)

    async def _slow_exec(args, ctx):
        await asyncio.sleep(0.1)
        return "done"

    from voice.agent import permissions as perm
    from voice.agent.permissions import Decision, PermissionResult, register_classifier

    slow = Tool(
        name="slow2_unique_for_filler",
        description="sleeps",
        parameters_schema={"type": "object", "properties": {}},
        execute=_slow_exec,
    )
    reg = ToolRegistry()
    reg.register(slow)

    @register_classifier("slow2_unique_for_filler")
    def _cls(args, workdir):
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{"function": {"name": "slow2_unique_for_filler", "arguments": {}}}],
                done=True,
            )],
            [_ollama_line(content="ok"), _ollama_line(done=True)],
        ]
        turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]
        types = [e["type"] for e in events]
        assert "audio_filler" not in types
    finally:
        perm._CLASSIFIERS.pop("slow2_unique_for_filler", None)


# ─────────────────── Logging chyb (Ollama 400, agent_error propagace) ───────────────────


class _FakeClient400:
    """FakeClient, který vrátí non-200 status pro první request."""

    def __init__(self, status_code: int, body: str = "model not found"):
        self.status_code = status_code
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def stream(self, method: str, url: str, json=None):  # noqa: ARG002
        return _FakeStream([self.body], status_code=self.status_code)


@pytest.mark.asyncio
async def test_loop_ollama_400_logs_error(tmp_path: Path, caplog):
    """Když Ollama vrátí 400, loop MUSÍ:
    1. zalogovat ERROR s detaily (status, body, payload keys),
    2. zalogovat ERROR „agent loop terminating",
    3. emit agent_error event s informativní zprávou.
    """
    import logging
    caplog.set_level(logging.WARNING, logger="agent-loop")

    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "test"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch(
        "voice.agent.loop.httpx.AsyncClient",
        return_value=_FakeClient400(400, body='{"error":"invalid request"}'),
    ):
        events = [ev async for ev in loop.run()]

    # Event: agent_error obsahuje status code i body
    errs = [e for e in events if e["type"] == "agent_error"]
    assert len(errs) == 1
    assert "ollama 400" in errs[0]["msg"]
    assert "invalid request" in errs[0]["msg"]

    # Logging: ERROR pro non-200 i pro agent loop terminate
    log_msgs = [r.getMessage() for r in caplog.records if r.name == "agent-loop"]
    assert any("non-200" in m and "400" in m for m in log_msgs), \
        f"missing non-200 log; got: {log_msgs}"
    assert any("terminating with error" in m for m in log_msgs), \
        f"missing terminating log; got: {log_msgs}"


# ─────── Malformed tool_call arguments (production bug: Ollama 400 mid-turn) ───────


@pytest.mark.asyncio
async def test_loop_malformed_args_short_circuits_execution(tmp_path: Path):
    """Model emituje truncated JSON v `arguments` (např. max_tokens uprostřed
    `{"command":"rm -rf`).

    Production bug repro: před fixem se raw string propagoval do history
    a další Ollama round vrátil HTTP 400 `Value looks like object, but can't
    find closing '}' symbol`.

    Po fixu (defense in depth):
    1. `normalize_tool_calls` přepíše arguments na sentinel dict
       (validní object → Ollama nevyhodí 400),
    2. `_execute_one` detekuje sentinel přes `is_malformed_args` a tool
       NESPUSTÍ (security: prevence schování destruktivních args za malformed JSON),
    3. emit paired `tool_result` ok=False s `_parse_error` + `raw_hash`,
    4. history obsahuje validní dict arguments → další round projde.
    """
    # Echo tool by execute když bychom dostali args. Pokud test selže
    # (short-circuit nefunguje), echo by vrátil `{"echoed":"","length":0}`
    # místo error payloadu → assertion na _parse_error by failnula.
    truncated = '{"text": "hi'  # uříznutý JSON object
    scripts = [
        [_ollama_line(
            tool_calls=[{
                "id": "tc_malformed",
                "function": {"name": "echo", "arguments": truncated},
            }],
            done=True,
        )],
        # Druhý round: po malformed tool_result agent dokončí turn textem.
        [_ollama_line(content="ok, malformed"), _ollama_line(done=True)],
    ]

    turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
    loop = AgentLoop(
        model="test",
        messages=[{"role": "user", "content": "go"}],
        registry=default_registry("agent"),
        turn_state=turn_state,
        workdir=tmp_path,
    )

    with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
        events = [ev async for ev in loop.run()]

    # tool_call event: args už jsou sentinel (NE empty {} a NE raw string)
    tc_ev = next(e for e in events if e["type"] == "tool_call")
    assert tc_ev["args"]["_parse_error"] == "invalid_json"
    assert tc_ev["args"]["raw_length"] == len(truncated)
    assert len(tc_ev["args"]["raw_hash"]) == 64
    # Žádný raw_preview → "hi" string NESMÍ leaknout do eventu
    assert "raw_preview" not in tc_ev["args"]

    # tool_result: ok=False, payload obsahuje _parse_error
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False
    payload = json.loads(tr["content"])
    assert payload["_parse_error"] == "invalid_json"
    assert payload["raw_hash"] == tc_ev["args"]["raw_hash"]
    assert "malformed" in payload["error"]

    # History: assistant.tool_calls arguments MUSÍ být dict (sentinel object),
    # jinak by Ollama na další round vyhodila 400.
    asst = next(m for m in loop.messages if m["role"] == "assistant" and "tool_calls" in m)
    args = asst["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)  # Ollama chce object, ne JSON string
    assert args["_parse_error"] == "invalid_json"

    # Tool message taky validní JSON (per spec).
    tool_msg = next(m for m in loop.messages if m["role"] == "tool")
    tool_payload = json.loads(tool_msg["content"])
    assert tool_payload["ok"] is False

    # Turn úspěšně doběhl — final text z druhého round, žádný agent_error
    types = [e["type"] for e in events]
    assert types[-1] == "agent_done"
    text = "".join(e["delta"] for e in events if e["type"] == "text")
    assert "ok, malformed" in text


@pytest.mark.asyncio
async def test_loop_malformed_args_does_not_execute_tool(tmp_path: Path):
    """Security regression: tool s side effects NESMÍ proběhnout, když args
    jsou malformed. Bez short-circuit by classifier viděl prázdné `{}` a
    tool dostal defaults → destruktivní akce mimo audit.
    """
    from voice.agent import permissions as perm
    from voice.agent.permissions import Decision, PermissionResult, register_classifier

    executed_count = 0

    async def _counting_exec(args, ctx):
        nonlocal executed_count
        executed_count += 1
        return {"ran": True}

    counting = Tool(
        name="counting_destructive_unique",
        description="counter",
        parameters_schema={"type": "object", "properties": {}},
        execute=_counting_exec,
    )
    reg = ToolRegistry()
    reg.register(counting)

    @register_classifier("counting_destructive_unique")
    def _cls(args, workdir):
        # AUTO — kdyby short-circuit selhal, tool by se spustil bez ptaní.
        return PermissionResult(
            decision=Decision.AUTO, reason="t", summary="t", risk="low",
            requires_explicit=False,
        )

    try:
        scripts = [
            [_ollama_line(
                tool_calls=[{
                    "id": "tc_x",
                    "function": {
                        "name": "counting_destructive_unique",
                        "arguments": '{"path": "/tmp/x',  # truncated
                    },
                }],
                done=True,
            )],
            [_ollama_line(content="done"), _ollama_line(done=True)],
        ]
        turn_state = {"id": "t1", "canceled": False, "cancel_event": asyncio.Event()}
        loop = AgentLoop(
            model="test", messages=[{"role": "user", "content": "go"}],
            registry=reg, turn_state=turn_state, workdir=tmp_path,
        )
        with patch("voice.agent.loop.httpx.AsyncClient", return_value=_FakeClient(scripts)):
            events = [ev async for ev in loop.run()]

        # Tool nebyl spuštěn — short-circuit zafungoval PŘED classifier i execute
        assert executed_count == 0
        tr = next(e for e in events if e["type"] == "tool_result")
        assert tr["ok"] is False
        assert json.loads(tr["content"])["_parse_error"] == "invalid_json"
    finally:
        perm._CLASSIFIERS.pop("counting_destructive_unique", None)
