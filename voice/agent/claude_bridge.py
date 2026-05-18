"""Claude Code CLI bridge - subprocess wrapper s stream-json parserem.

Spawnuje `claude -p --output-format stream-json --include-partial-messages`
a parsuje NDJSON events. Voláno z `ask_claude` toolu jako oneshot per call
(persistent warm session je out of scope, viz plans/agent_mode.md Fáze 6).

Auth: Claude CLI si řeší sám (OAuth/keychain/API key dle setupu uživatele).
Bridge nenastavuje ANTHROPIC_API_KEY explicitně, jen ho propouští z env
přes allowlist (pokud user ho má). Pokud `claude` funguje v terminálu,
funguje i tady.

Dva módy:

`mode="consult"` (default, pure expert review):
  --tools "" --permission-mode plan --no-session-persistence
  cwd = empty temp dir, žádný --add-dir
  Bezpečnost: Claude nemá žádné tools, nic na disku nesahá. Sandbox = nic.

`mode="edit"` (FULL SHELL DELEGACE v cwd=workdir):
  --tools "Read,Edit,Write,Bash,Glob,Grep" --permission-mode acceptEdits
  --add-dir <WORKDIR>, cwd = workdir
  Bezpečnost: Claude má Bash + Edit/Write v --add-dir. To je full shell
  v cwd. Bash může `cd /` a operovat dále (Bash NENÍ omezený --add-dir,
  ten omezuje jen file tools). Classifier proto MUSÍ require_explicit=True
  (user řekne "ano povoluju"). Audit zachytí spawned ask_claude call, NE
  jednotlivé Claude tool calls - pokud user chce per-tool tracing, dostane
  ho přes progress events v UI.

Stream parser:
- `proc.stdout.readline()` (nativní NDJSON fragmentation handling)
- Per-line `json.loads`, dispatch dle `event["type"]`
- Progress callback dostane normalizovaný envelope `{stage, message, ...}`
- Buffer pro accumulated assistant text → final result
- Output cap: per-line, agregate counter, killpg pokud překročí

Cancel:
- `cancel_event` (asyncio.Event z turn_state) race proti readline + proc.wait
- Cleanup: SIGTERM process group → 2s → SIGKILL

Env scrub: HOME, USER, PATH, LANG, LC_*, XDG_*, ANTHROPIC_API_KEY (volitelný).
Drop AGENT_*, BRAVE_*, GH_*, *_TOKEN, *_SECRET, MY_API_KEY atd.
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

log = logging.getLogger("agent-claude-bridge")


_ENV_ALLOWLIST = frozenset({
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
    "PATH", "TMPDIR",
    # NE PWD - Codex iter-9: PWD z parent env by mohl Claude CLI použít místo
    # getcwd() a najít project/CLAUDE.md mimo náš cwd=tmpdir. Setujeme ho
    # explicitně níž na skutečné cwd.
    "ANTHROPIC_API_KEY",  # volitelné; OAuth/keychain je primární
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
})


# Mode=edit tools allowlist. Subset Claude built-in tools. NE `default`
# (vyloučí WebSearch/WebFetch - máme vlastní; TodoWrite - sub-agent nesmysl).
_EDIT_MODE_TOOLS = "Read,Edit,Write,Bash,Glob,Grep"

# Throttle pro repetitive progress events (thinking, partial text).
_PROGRESS_THROTTLE_SEC = 1.5


def _build_subprocess_env(cwd: str | None = None) -> dict[str, str]:
    """Filtruj parent env na allowlist. Claude CLI si auth řeší sám.
    `cwd` (pokud daný) → PWD=cwd, aby Claude CLI co případně používá PWD
    místo getcwd() neunikl mimo sandbox cwd."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _ENV_ALLOWLIST or k.startswith("LC_"):
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
            "--permission-mode", "plan",   # plan: žádné edits ani s tools
            "--tools", "",                 # žádné tools
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
        # System přes argv (CLI 2.1.x nemá --append-system-prompt-file).
        # Je v `ps`, ale system bývá krátký a obecnější než user prompt.
        argv += ["--append-system-prompt", system]
    return argv


def _short_tool_input_summary(name: str, inp: dict | None) -> str:
    """Zkrácený popis tool_use vstupu pro progress event."""
    if not isinstance(inp, dict):
        return name
    if name in ("Edit", "Write", "Read", "NotebookEdit"):
        p = inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
        if isinstance(p, str):
            return f"{name} {p}"
    if name == "Bash":
        cmd = inp.get("command")
        if isinstance(cmd, str):
            return f"Bash: {cmd[:80]}"
    if name in ("Glob", "Grep"):
        pat = inp.get("pattern")
        if isinstance(pat, str):
            return f"{name} {pat}"
    return name


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


def _parse_stream_event(
    obj: dict,
    state: dict,
) -> dict | None:
    """Mapuje jeden stream-json event na progress payload (nebo None).

    `state` mutable dict pro accumulating: `assistant_text`, `last_emit`,
    `tool_uses` (list dict)."""
    t = obj.get("type")
    if t == "system":
        st = obj.get("subtype")
        if st == "init":
            # Codex iter-7: scalar fields guard - CLI mohl pošle non-string
            # session_id/model. Slice/format na non-string crash.
            sid = obj.get("session_id")
            sid_str = sid if isinstance(sid, str) else ""
            model = obj.get("model")
            model_str = model if isinstance(model, str) else "?"
            state["session_id"] = sid_str
            return {
                "stage": "started",
                "message": f"Claude {model_str} session {sid_str[:8]}",
                "session_id": sid_str,
                "model": model_str,
            }
        return None  # status/heartbeat - skip

    if t == "stream_event":
        # Codex iter-6: nested non-dict guard. `event` MUSÍ být dict (jinak
        # .get() AttributeError shodí celý reader task).
        ev = obj.get("event")
        if not isinstance(ev, dict):
            return None
        et = ev.get("type")
        if et == "content_block_start":
            cb = ev.get("content_block")
            if not isinstance(cb, dict):
                return None
            cbt = cb.get("type")
            if cbt == "thinking":
                # Throttle: thinking emit max 1× per window
                now = time.monotonic()
                if now - state.get("last_thinking_emit", 0) < _PROGRESS_THROTTLE_SEC:
                    return None
                state["last_thinking_emit"] = now
                return {"stage": "thinking", "message": "přemýšlí…"}
            if cbt == "tool_use":
                # Codex iter-8: name scalar guard.
                name_raw = cb.get("name")
                name = name_raw if isinstance(name_raw, str) else "?"
                summary = _short_tool_input_summary(name, cb.get("input"))
                state.setdefault("tool_uses", []).append(name)
                return {
                    "stage": "tool_use",
                    "message": summary,
                    "tool_name": name,
                }
            if cbt == "text":
                # Nový text block - neemit tady, počkáme až bude něco akumulované
                state["text_block_started"] = True
                return None
        return None  # ostatní stream_event subtypes (deltas, stops) → skip

    if t == "assistant":
        # Accumulated assistant message; extrahuj text pro finální result.
        # Codex iter-6: message MUSÍ být dict, content MUSÍ být list.
        msg = obj.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        text_parts = [
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        if text_parts:
            state["assistant_text"] = "".join(text_parts)
        return None

    if t == "user":
        # User-role message obsahuje tool_result (Claude vidí výsledek tool).
        # Codex iter-6: nested non-dict guards.
        msg = obj.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                is_err = b.get("is_error") is True
                return {
                    "stage": "tool_result",
                    "message": "tool selhal" if is_err else "tool OK",
                    "ok": not is_err,
                }
        return None

    if t == "result":
        # Finální event - caller ho zpracuje mimo dispatcher (terminal).
        return None

    return None  # rate_limit_event, mcp_*, unknown → skip


async def _read_stream_capped(
    proc: asyncio.subprocess.Process,
    cap_bytes: int,
    progress_cb: Callable[[dict], Awaitable[None]] | None,
    state: dict,
    overflow_event: asyncio.Event,
) -> dict | None:
    """Čte NDJSON ze stdout řádek po řádku, dispatch events. Vrací finální
    `result` event obj (nebo None pokud stream skončil bez result).

    Při překročení capu nebo asyncio LimitOverrunError (line > stream_limit):
    set state["overflow"], signal overflow_event, return None. Caller pak
    race-detekuje overflow a killpg eskaluje.
    """
    def _signal_overflow():
        state["overflow"] = True
        overflow_event.set()

    total = 0
    while True:
        try:
            line = await proc.stdout.readline()
        except (asyncio.LimitOverrunError, ValueError) as e:
            # Codex iter-4/5 HIGH: StreamReader.readline() při overflow limitu
            # v Pythonu 3.11+ raises plain ValueError (ne LimitOverrunError jak
            # bych myslel). Treat as overflow (ne tichý break → "no result").
            log.warning("claude stream line exceeded asyncio limit: %s", e)
            _signal_overflow()
            try:
                await proc.stdout.read(-1)  # drain rest
            except Exception:
                pass
            break
        except Exception as e:
            log.warning("claude stdout read error: %s", e)
            break
        if not line:
            break  # EOF
        total += len(line)
        if total > cap_bytes:
            log.warning("claude stream exceeded %d bytes - overflow", cap_bytes)
            _signal_overflow()
            break
        try:
            obj = json.loads(line.decode("utf-8", errors="replace").rstrip())
        except (ValueError, UnicodeDecodeError):
            continue
        # Codex iter-5 MEDIUM: parser předpokládá dict. `[]` / `"x"` / `123`
        # by jinak hodili AttributeError v obj.get(). Skip non-dict lines.
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "result":
            return obj
        payload = _parse_stream_event(obj, state)
        if payload is not None:
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
    system: str | None,
    model: str,
    mode: str,                                         # "consult" | "edit"
    workdir: Path | None,                              # required for mode=edit
    timeout_sec: float,
    output_cap_bytes: int,
    claude_bin: str = "claude",
    cancel_event: asyncio.Event | None = None,
    progress_callback: Callable[[dict], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Oneshot subprocess call na Claude Code CLI s stream-json parserem.

    Vrací (drop-in kompatibilní napříč módy a s/bez progress_callback):
      {
        "ok": bool,
        "text": str,                    # finální assistant text
        "model": str,
        "mode": str,
        "session_id": str?,
        "total_cost_usd": float?,
        "duration_ms": int,
        "tool_uses": [str, ...],        # list jmen tools které Claude použil
        "error": str?,                  # pokud ok=False
        "exit_code": int?,
        "stderr_preview": str?,
      }

    Args validace probíhá v callsite (`tools/claude.py`). Bridge předpokládá
    sanitizovaný prompt/system/model/mode/workdir.
    """
    if not prompt or not prompt.strip():
        return {"ok": False, "error": "empty prompt", "mode": mode}
    try:
        argv = _build_argv(
            claude_bin=claude_bin, mode=mode, model=model,
            system=system, workdir=workdir,
        )
    except ValueError as e:
        return {"ok": False, "error": str(e), "mode": mode}

    # cwd: edit = workdir, consult = empty temp
    with tempfile.TemporaryDirectory(prefix="claude_bridge_") as tmp:
        cwd = str(workdir) if mode == "edit" else tmp
        env = _build_subprocess_env(cwd=cwd)  # PWD = cwd (Codex iter-9 fix)
        log.info("claude spawn mode=%s model=%s cwd=%s", mode, model, cwd)

        try:
            # Codex audit HIGH: default asyncio StreamReader limit ~64 KiB per
            # line. Validní finální `result` event může mít text > 64 KiB
            # → ValueError → parser zhroutí, vrátí "no result event". Nastavit
            # limit na output_cap (s rezervou) ať readline zvládne jakýkoli
            # legitimní line. Cap stejně chrání proti unbounded RAM.
            stream_limit = max(output_cap_bytes + 16 * 1024, 256 * 1024)
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

        # Codex audit HIGH: reader tasky MUSÍ start PŘED stdin write. Pokud
        # Claude před přečtením stdinu zaplní stdout/stderr pipe (např. vypíše
        # init+status events okamžitě), `proc.stdin.drain()` by visel navždy
        # protože nikdo nečte stdout. Timeout/cancel race ještě neběží.
        state: dict = {"assistant_text": "", "tool_uses": []}
        # overflow_event je signál ze _read_stream_capped → main coroutine,
        # aby okamžitě killpg eskalovala (jinak by čekala na timeout).
        overflow_event = asyncio.Event()
        stdout_task = asyncio.create_task(
            _read_stream_capped(proc, output_cap_bytes, progress_callback,
                                state, overflow_event)
        )
        stderr_task = asyncio.create_task(_drain_stderr(proc, 64 * 1024))

        # Stdin write taky jako task - nesmí blokovat main coroutine, aby
        # timeout/cancel race fungoval i v patologickém případě.
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
        # Codex iter-4 HIGH: overflow_event MUSÍ být v race waiters. Jinak
        # main čeká až do timeout_sec, i když reader už dávno detekoval cap
        # overflow a SIGTERM mohl být eskalován okamžitě.
        overflow_task = asyncio.create_task(overflow_event.wait())
        waiters: list[asyncio.Task] = [wait_task, overflow_task]
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
        overflow_triggered = overflow_task in done
        timed_out = wait_task not in done and not was_canceled and not overflow_triggered
        if timed_out or was_canceled or overflow_triggered:
            await _kill_process_group(proc)
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()
        if not overflow_task.done():
            overflow_task.cancel()
        # Stdin task pravděpodobně už doběhl (proc.wait() se vrátí po jeho
        # close); pokud ne, cancel ho.
        if not stdin_task.done():
            stdin_task.cancel()

        # Drain reader tasks (po killu by měly skončit rychle). Codex iter-7:
        # parser may raise exception on malformed data - catch a degrade na
        # `no result event` místo propagace do callera.
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
        if state.get("overflow"):
            mb = output_cap_bytes / (1024 * 1024)
            return {
                "ok": False, "mode": mode,
                "error": (
                    f"Claude vygeneroval příliš velkou odpověď (přes "
                    f"{mb:.1f} MB raw stream). Zkus rozdělit úkol na menší "
                    f"kroky - např. nejdřív průzkum projektu, pak postupně "
                    f"jednotlivé sekce dokumentu. Pokud máš víc paměti, jde "
                    f"zvednout env AGENT_CLAUDE_OUTPUT_CAP_BYTES."
                ),
                "overflow": True,
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

        # Extract final result. Codex audit HIGH: pokud chybí `result` event
        # (CLI crash, protokol drift, exec0 ale stream nedoběhl), MUSÍME vrátit
        # ok=False. Mlčet by maskovalo invalid JSON, drift CLI verze i partial
        # výsledky v edit módu (Claude něco napsal na disk, my říkáme "OK").
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
        # CLI ve `result` má i `result` field s finální text odpovědí
        r = result_obj.get("result")
        if isinstance(r, str) and r:
            text = r

        # Codex iter-7: subtype scalar guard (může být None/int z buggy CLI).
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

        # Codex iter-8: scalar guard pro result.model (CLI by mohl pošle non-string).
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
