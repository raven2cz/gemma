"""PrintModeAdapter - spustí `claude -p` per ask() volání (ephemeral).

Refactor z `voice/agent/claude_bridge.py`. Žádná funkční změna proti původní
implementaci, jen package move + interface fit. Stávající testy musí projít
beze změny.

Behavior summary:
- Per-call subprocess: spawn `claude -p ...`, send prompt via stdin
- Stream-json output → progress events via callback + accumulated text
- Cancel: cancel_event race s killpg fallback
- Timeout: hard kill po `timeout_sec`
- Env scrub: allowlist filter (HOME, USER, PATH, LANG, XDG_*, ANTHROPIC_API_KEY)
- mode="consult": empty tmp cwd, --permission-mode plan --tools ""
- mode="edit":    workdir cwd, --permission-mode acceptEdits --tools R/W/B/...

Per-line StreamReader limit = 32 MiB (single event tolerance, no total cap).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..base import (
    AbstractClaudeAdapter,
    AdapterCapabilities,
    ProgressCallback,
    SessionState,
)
from ..config import AdapterConfig
from ..parsing.stream_json import parse_stream_event
from ..progress import ProgressEvent, SessionInfo
from ..result import ClaudeResult, Mode

log = logging.getLogger("claude_bridge.print_mode")


# Mode=edit tools allowlist. Subset Claude built-in tools.
_EDIT_MODE_TOOLS = "Read,Edit,Write,Bash,Glob,Grep"


_DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
    "PATH", "TMPDIR",
    "ANTHROPIC_API_KEY",
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
)


def _build_subprocess_env(
    cwd: str | None = None,
    allowlist: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Filtruj parent env na allowlist + LC_* prefix.

    `cwd` (pokud daný) → PWD=cwd, aby Claude CLI co případně používá PWD
    místo getcwd() neunikl mimo sandbox cwd. PWD NESMÍ být v allowlist
    aby parent's PWD neprosakl (Codex iter-9).
    """
    if allowlist is None:
        allowlist = _DEFAULT_ENV_ALLOWLIST
    allow = frozenset(allowlist)
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k == "PWD":
            continue  # explicitně setujeme níž
        if k in allow or k.startswith("LC_"):
            out[k] = v
    if cwd is not None:
        out["PWD"] = cwd
    return out


def _build_argv(
    *,
    claude_bin: str,
    mode: str,
    model: str,
    system: str | None,
    workdir: Path | None,
) -> list[str]:
    """Sestaví argv pro `claude -p` podle módu."""
    argv = [
        claude_bin,
        "-p",
        "--output-format", "stream-json",
        "--input-format", "text",
        "--verbose",                       # required pro stream-json
        "--include-partial-messages",      # incremental events
        "--no-session-persistence",
        "--model", model,
    ]
    if mode == "consult":
        argv += [
            "--permission-mode", "plan",
            "--tools", "",
        ]
    elif mode == "edit":
        if workdir is None:
            raise ValueError("mode=edit requires workdir")
        argv += [
            "--permission-mode", "acceptEdits",
            "--tools", _EDIT_MODE_TOOLS,
            "--add-dir", str(workdir),
        ]
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if system:
        argv += ["--append-system-prompt", system]
    return argv


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM → 2s wait → SIGKILL. Cílí celý process group."""
    if proc.returncode is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=2.0)
        return
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=1.0)
    except asyncio.TimeoutError:
        log.warning("claude pid=%d survived SIGKILL", proc.pid)


async def _safe_progress(
    cb: Callable[[dict], Awaitable[None]] | None,
    payload: dict,
) -> None:
    """Progress emit s tolerancí chyb (callback NESMÍ shodit bridge)."""
    if cb is None:
        return
    try:
        await cb(payload)
    except Exception:
        log.exception("progress callback failed: %s", payload.get("stage"))


async def _read_stream(
    proc: asyncio.subprocess.Process,
    progress_cb: Callable[[dict], Awaitable[None]] | None,
    state: dict,
) -> dict | None:
    """Čte NDJSON ze stdout řádek po řádku, dispatch events. Vrací finální
    `result` event obj (nebo None pokud stream skončil bez result)."""
    log.info("claude _read_stream START pid=%s", proc.pid)
    lines_read = 0
    payloads_emitted = 0
    while True:
        try:
            line = await proc.stdout.readline()
        except (asyncio.LimitOverrunError, ValueError) as e:
            log.warning("claude stream single-line limit exceeded (skip): %s", e)
            continue
        except Exception as e:
            log.warning("claude stdout read error: %s", e)
            break
        if not line:
            log.info(
                "claude _read_stream EOF pid=%s lines=%d emitted=%d",
                proc.pid, lines_read, payloads_emitted,
            )
            break
        lines_read += 1
        try:
            obj = json.loads(line.decode("utf-8", errors="replace").rstrip())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "result":
            log.info(
                "claude _read_stream RESULT pid=%s lines=%d emitted=%d",
                proc.pid, lines_read, payloads_emitted,
            )
            return obj
        payload = parse_stream_event(obj, state)
        if payload is not None:
            payloads_emitted += 1
            await _safe_progress(progress_cb, payload)
    return None


async def _drain_stderr(proc: asyncio.subprocess.Process, cap_bytes: int) -> bytes:
    """Async stderr drain - bez něj plný pipe zamrzne proces."""
    buf = bytearray()
    while True:
        try:
            chunk = await proc.stderr.read(8192)
        except Exception:
            break
        if not chunk:
            break
        if len(buf) < cap_bytes:
            buf.extend(chunk[:cap_bytes - len(buf)])
    return bytes(buf)


async def ask_claude_oneshot(
    *,
    prompt: str,
    system: str | None = None,
    model: str,
    mode: str,                                         # "consult" | "edit"
    workdir: Path | None,                              # required for mode=edit
    timeout_sec: float,
    output_cap_bytes: int = 0,                         # deprecated, ignored
    claude_bin: str = "claude",
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[dict], Awaitable[None]] | None = None,
    env_allowlist: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Oneshot subprocess call na Claude Code CLI s stream-json parserem.

    Drop-in kompatibilní s původním `voice/agent/claude_bridge.ask_claude_oneshot`
    - vrací dict shape co voice/agent/tools/claude.py očekává.
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "error": "empty prompt", "mode": mode}
    if cancel_event is not None and not isinstance(cancel_event, asyncio.Event):
        # threading.Event.wait() is sync-blocking; asyncio.create_task on it
        # freezes the entire event loop (no subprocess I/O, no progress).
        raise TypeError(
            f"cancel_event must be asyncio.Event, got "
            f"{type(cancel_event).__module__}.{type(cancel_event).__name__}"
        )
    try:
        argv = _build_argv(
            claude_bin=claude_bin, mode=mode, model=model,
            system=system, workdir=workdir,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e), "mode": mode}

    allowlist = env_allowlist or (
        "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
        "PATH", "TMPDIR",
        "ANTHROPIC_API_KEY",
        "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
    )

    # cwd: edit = workdir, consult = empty temp
    with tempfile.TemporaryDirectory(prefix="claude_bridge_") as tmp:
        cwd = str(workdir) if mode == "edit" else tmp
        env = _build_subprocess_env(cwd=cwd, allowlist=allowlist)
        log.info("claude spawn mode=%s model=%s cwd=%s", mode, model, cwd)

        try:
            stream_limit = 32 * 1024 * 1024
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd, env=env,
                start_new_session=True,
                limit=stream_limit,
            )
        except FileNotFoundError:
            return {
                "ok": False, "mode": mode,
                "error": f"claude CLI nenalezen (binary={claude_bin!r}). "
                         "Nainstaluj Claude Code nebo nastav CLAUDE_CLI_BIN.",
            }
        except Exception as e:
            return {"ok": False, "mode": mode,
                    "error": f"subprocess spawn: {type(e).__name__}: {e}"}

        start = time.monotonic()
        state: dict = {"assistant_text": "", "tool_uses": []}
        stdout_task = asyncio.create_task(
            _read_stream(proc, progress_callback, state)
        )
        stderr_task = asyncio.create_task(_drain_stderr(proc, 64 * 1024))

        async def _write_stdin() -> None:
            try:
                proc.stdin.write(prompt.encode("utf-8"))
                await proc.stdin.drain()
                proc.stdin.close()
                await proc.stdin.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log.warning("stdin write failed: %s", e)
        stdin_task = asyncio.create_task(_write_stdin())
        wait_task = asyncio.create_task(proc.wait())
        waiters: list[asyncio.Task] = [wait_task]
        cancel_task = None
        if cancel_event is not None:
            cancel_task = asyncio.create_task(cancel_event.wait())
            waiters.append(cancel_task)

        try:
            done, _pending = await asyncio.wait(
                waiters, timeout=timeout_sec, return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            await _kill_process_group(proc)
            for t in (stdout_task, stderr_task, stdin_task, *waiters):
                if not t.done():
                    t.cancel()
            raise

        was_canceled = cancel_task is not None and cancel_task in done
        timed_out = wait_task not in done and not was_canceled
        if timed_out or was_canceled:
            await _kill_process_group(proc)
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        if not stdin_task.done():
            stdin_task.cancel()

        try:
            result_obj = await asyncio.wait_for(stdout_task, timeout=3.0)
        except asyncio.TimeoutError:
            stdout_task.cancel()
            result_obj = None
        except Exception as e:
            log.warning("stdout_task raised: %s", e)
            result_obj = None
        try:
            stderr_bytes = await asyncio.wait_for(stderr_task, timeout=2.0)
        except asyncio.TimeoutError:
            stderr_task.cancel()
            stderr_bytes = b""

        duration_ms = int((time.monotonic() - start) * 1000)
        exit_code = proc.returncode

        if was_canceled:
            return {
                "ok": False, "mode": mode, "error": "canceled",
                "duration_ms": duration_ms, "exit_code": exit_code,
            }
        if timed_out:
            return {
                "ok": False, "mode": mode,
                "error": (
                    f"Claude nestihl odpovědět do {timeout_sec:.0f}s. "
                    f"Komplexní úkoly (design dokument, multi-file refaktor) "
                    f"mohou vyžadovat víc času - env AGENT_CLAUDE_TIMEOUT_SEC."
                ),
                "timeout": True,
                "duration_ms": duration_ms, "exit_code": exit_code,
            }
        if exit_code != 0:
            stderr_preview = stderr_bytes[:500].decode("utf-8", errors="replace")
            return {
                "ok": False, "mode": mode,
                "error": f"claude CLI exit code {exit_code}",
                "exit_code": exit_code, "stderr_preview": stderr_preview,
                "duration_ms": duration_ms,
            }

        if result_obj is None:
            stderr_preview = stderr_bytes[:500].decode("utf-8", errors="replace")
            return {
                "ok": False, "mode": mode,
                "error": "no result event (CLI protocol failure or premature EOF)",
                "exit_code": exit_code,
                "stderr_preview": stderr_preview,
                "duration_ms": duration_ms,
                "session_id": state.get("session_id"),
                "tool_uses": state.get("tool_uses", []),
            }

        is_error = bool(result_obj.get("is_error"))
        text = state.get("assistant_text", "")
        r = result_obj.get("result")
        if isinstance(r, str) and r:
            text = r

        subtype_raw = result_obj.get("subtype")
        subtype = subtype_raw if isinstance(subtype_raw, str) else ""
        if is_error or subtype.startswith("error_"):
            return {
                "ok": False, "mode": mode,
                "error": text or f"claude error ({subtype or 'unknown'})",
                "duration_ms": duration_ms, "exit_code": exit_code,
                "session_id": state.get("session_id"),
                "tool_uses": state.get("tool_uses", []),
            }

        result_model = result_obj.get("model")
        result_model_str = result_model if isinstance(result_model, str) else model
        return {
            "ok": True,
            "mode": mode,
            "text": text,
            "model": result_model_str,
            "session_id": state.get("session_id"),
            "total_cost_usd": result_obj.get("total_cost_usd"),
            "duration_ms": duration_ms,
            "tool_uses": state.get("tool_uses", []),
            "num_turns": result_obj.get("num_turns"),
        }


class PrintModeAdapter:
    """AbstractClaudeAdapter implementace co spustí `claude -p` per ask().

    Wraps existing `ask_claude_oneshot()` function. Session metody jsou
    no-op (print mode nemá persistent state).
    """

    name = "print"

    def __init__(self, config: AdapterConfig) -> None:
        self.config = config
        self.capabilities = AdapterCapabilities.print_mode()

    async def ask(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str,
        mode: Mode,
        workdir: Path | None,
        session_id: str | None = None,  # ignored - print is ephemeral
        timeout_sec: float = 600.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ClaudeResult:
        # Convert ProgressCallback (typed) to legacy dict-based callback
        # for ask_claude_oneshot. Když caller passne typed callback, wrapneme.
        # Pokud passne dict-based (legacy), předáme přímo.
        legacy_cb: Callable[[dict], Awaitable[None]] | None = None
        if progress_callback is not None:
            async def _dict_cb(payload: dict) -> None:
                # Detect: voláme typed callback (s ProgressEvent) nebo dict?
                # Pro Phase 0 backwards compat: caller dnes passuje dict-based
                # callback (voice/agent/tools/claude.py emitter), takže ho
                # předáme přímo. Phase 4+ ho přepneme na typed.
                await progress_callback(payload)  # type: ignore[arg-type]
            legacy_cb = _dict_cb

        result_dict = await ask_claude_oneshot(
            prompt=prompt,
            system=system,
            model=model,
            mode=mode,
            workdir=workdir,
            timeout_sec=timeout_sec,
            claude_bin=self.config.claude_bin,
            cancel_event=cancel_event,
            progress_callback=legacy_cb,
            env_allowlist=self.config.env_allowlist,
        )

        # Convert legacy dict → ClaudeResult dataclass
        return _dict_to_result(result_dict, mode=mode, adapter="print")

    async def list_sessions(self) -> list[SessionInfo]:
        return []  # print mode nemá persistent sessions

    async def get_session(self, session_id: str) -> SessionInfo | None:
        return None

    async def clear_session(self, session_id: str) -> bool:
        return False  # not supported

    async def kill_session(self, session_id: str) -> bool:
        return False  # not supported

    async def health_check(self, session_id: str) -> SessionState:
        return "READY"  # print je vždy "ready" (žádné session)

    async def close(self) -> None:
        return None  # noop


def _dict_to_result(d: dict, *, mode: Mode, adapter: str) -> ClaudeResult:
    """Convert legacy dict shape (z ask_claude_oneshot) na ClaudeResult."""
    tool_uses = d.get("tool_uses") or ()
    if isinstance(tool_uses, list):
        tool_uses = tuple(tool_uses)
    return ClaudeResult(
        ok=bool(d.get("ok")),
        mode=d.get("mode", mode),
        text=d.get("text", ""),
        model=d.get("model", ""),
        session_id=d.get("session_id"),
        total_cost_usd=d.get("total_cost_usd"),
        duration_ms=int(d.get("duration_ms", 0) or 0),
        tool_uses=tool_uses,
        exit_code=d.get("exit_code"),
        error=d.get("error"),
        stderr_preview=d.get("stderr_preview"),
        timeout=bool(d.get("timeout", False)),
        canceled=d.get("error") == "canceled",
        adapter=adapter,
    )
