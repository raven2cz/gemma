"""Claude Code CLI bridge - oneshot subprocess call pro `ask_claude` tool.

Spawnuje `claude -p` jako subprocess, posílá prompt přes stdin, čte JSON
výstup. **Není to REST API call.** Vzor je `avatar-engine/avatar_engine/
bridges/claude.py` (zjednodušeno, oneshot místo persistentního).

Bezpečnostní model:
- Prompt se NEPOSÍLÁ přes argv (leak přes `ps`). Stdin only.
- `--bare`: skip hooks, LSP, plugin sync, auto-memory, keychain reads,
  CLAUDE.md auto-discovery. Auth je striktně přes `ANTHROPIC_API_KEY` env.
- `--tools ""`: žádné built-in Claude Code tools (Read/Edit/Bash). Sub-agent
  jen odpoví textem, žádné FS/shell side-effects. Naše tooly se přes tohle
  obejít nedají.
- `--no-session-persistence`: žádný session state na disku.
- `--permission-mode plan`: Claude plánuje, needituje (defense in depth).
- `cwd` = prázdný temp dir → Claude nemá kde číst project config, CLAUDE.md.
- Env scrub: keep jen needed (HOME/USER/PATH/LANG) + ANTHROPIC_API_KEY,
  drop AGENT_*/BRAVE_*/GH_*/*_SECRET/*_TOKEN atd.
- `start_new_session=True` (process group) + `os.killpg` cleanup. Claude CLI
  spawnuje child procesy (MCP servery, hooks); single PID kill nestačí.
- Per-chunk output cap během streaming reading. Při překročení killpg
  okamžitě, ne až po `communicate()` (jinak by CLI sežrala RAM).
- Stderr drain async task - bez něj plný stderr pipe zamrzne proces.

Cancel:
- Bridge přijímá `cancel_event` (asyncio.Event z turn_state). Race proti
  process wait + read tasks.
- Cleanup vždy: SIGTERM proces group → wait 2s → SIGKILL.
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
from typing import Any

log = logging.getLogger("agent-claude-bridge")


# Env vars které Claude CLI legitimně potřebuje. ANTHROPIC_API_KEY přidáme
# explicitně. Vše ostatní z parent env je drop (žádné AGENT_*, GH_*, *_SECRET).
_ENV_ALLOWLIST = frozenset({
    "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TERM",
    "PATH", "PWD", "TMPDIR",
    # Claude CLI honoruje:
    "CLAUDE_CODE_SIMPLE",  # nastaví ho samo přes --bare, ale necháme passthrough
    "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
})


def _build_subprocess_env(api_key: str) -> dict[str, str]:
    """Filtruj parent env na allowlist + přidej ANTHROPIC_API_KEY."""
    out: dict[str, str] = {}
    for k, v in os.environ.items():
        if k in _ENV_ALLOWLIST or k.startswith("LC_"):
            out[k] = v
    out["ANTHROPIC_API_KEY"] = api_key
    # --bare už nastavuje CLAUDE_CODE_SIMPLE=1 sám, ale explicit doesn't hurt.
    out.setdefault("CLAUDE_CODE_SIMPLE", "1")
    return out


async def _read_stream_capped(
    stream: asyncio.StreamReader,
    cap_bytes: int,
    *,
    on_overflow: callable[[], None],
) -> bytes:
    """Async per-chunk reader. Při překročení capu zavolá `on_overflow`
    (typicky killpg) a vrátí buffer ořezaný na cap. Nikdy nečte víc než cap+1
    do paměti - neuložíme komprimovaný/decompression bombu."""
    buf = bytearray()
    overflowed = False
    while True:
        try:
            chunk = await stream.read(8192)
        except Exception:
            break
        if not chunk:
            break
        if overflowed:
            # Pokračujeme drainem (jinak by SIGTERM nestihl), ale zahodíme.
            continue
        buf.extend(chunk)
        if len(buf) > cap_bytes:
            overflowed = True
            try:
                on_overflow()
            except Exception:
                log.exception("on_overflow callback failed")
    if overflowed:
        return bytes(buf[:cap_bytes])
    return bytes(buf)


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGTERM → wait 2s → SIGKILL. Cílí celý process group (Claude spawnuje
    child procesy: MCP servery, hooks)."""
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
        log.warning("claude subprocess pid=%d survived SIGKILL (timeout 1s)", proc.pid)


async def ask_claude_oneshot(
    *,
    prompt: str,
    system: str | None,
    model: str,
    timeout_sec: float,
    output_cap_bytes: int,
    api_key: str,
    claude_bin: str = "claude",
    cancel_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    """Oneshot subprocess call na Claude Code CLI. Vrací:

    {
      "ok": bool,
      "text": str,           # extracted text response
      "stop_reason": str?,
      "duration_ms": int,
      "model": str,
      "error": str?,         # pokud ok=False
      "exit_code": int?,
      "stderr_preview": str? # první ~500 bytů stderr při errorech
    }

    Žádné FS side-effects, žádné network calls z naší strany (Claude CLI
    si dělá HTTPS na Anthropic API sám).
    """
    if not api_key:
        return {
            "ok": False,
            "error": (
                "ANTHROPIC_API_KEY není nastavený. Env var nebo "
                "~/.anthropic-api-key (chmod 0600)."
            ),
        }
    if not prompt or not prompt.strip():
        return {"ok": False, "error": "empty prompt"}

    # Build argv - prompt NIKDY do argv (ps leak). Stdin only.
    argv = [
        claude_bin,
        "-p",                            # --print, non-interactive
        "--bare",                        # skip hooks/LSP/plugins/keychain/CLAUDE.md
        "--no-session-persistence",      # žádný state na disku
        "--permission-mode", "plan",     # plan only, žádné edits (defense in depth)
        "--output-format", "json",       # single JSON result
        "--input-format", "text",        # stdin = plain text prompt
        "--model", model,
        "--tools", "",                   # disable all built-in tools (Read/Edit/Bash)
    ]
    if system:
        # System přes argv - ne ideal kvůli `ps`, ale --append-system-prompt-file
        # v CLI 2.1.x neexistuje. System je obvykle krátký a méně citlivý než user
        # prompt. Pokud user dá secrets do system, je to jeho rozhodnutí.
        argv += ["--append-system-prompt", system]

    env = _build_subprocess_env(api_key)

    # Empty temp dir as cwd - Claude nemá kde najít project config / CLAUDE.md.
    with tempfile.TemporaryDirectory(prefix="claude_bridge_") as tmp:
        cwd = Path(tmp)
        log.debug("claude subprocess: argv=%s cwd=%s", argv, cwd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=env,
                start_new_session=True,  # process group → killpg
            )
        except FileNotFoundError:
            return {
                "ok": False,
                "error": f"claude CLI nenalezen (binary={claude_bin!r}). "
                         "Nainstaluj Claude Code nebo nastav CLAUDE_CLI_BIN.",
            }
        except Exception as e:
            return {"ok": False, "error": f"subprocess spawn: {type(e).__name__}: {e}"}

        start = time.monotonic()
        overflow_triggered = {"v": False}

        def _on_overflow():
            overflow_triggered["v"] = True
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

        # Spustit reader tasks PŘED stdin write, ať se pipe neblokuje
        # zpětným tlakem stdoutu.
        stdout_task = asyncio.create_task(_read_stream_capped(
            proc.stdout, output_cap_bytes, on_overflow=_on_overflow,
        ))
        # Stderr cap nižší (debug logy + occasional warnings), ale dostatečný.
        stderr_task = asyncio.create_task(_read_stream_capped(
            proc.stderr, 64 * 1024, on_overflow=lambda: None,  # neabortovat na stderr
        ))

        # Stdin: zapsat prompt + EOF
        try:
            proc.stdin.write(prompt.encode("utf-8"))
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            # Subprocess umřel dřív než přečetl stdin - pokračujeme do readeru
            # který zjistí exit code.
            pass
        except Exception as e:
            await _kill_process_group(proc)
            return {"ok": False, "error": f"stdin write: {type(e).__name__}: {e}"}

        # Race: timeout vs cancel_event vs process exit
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
            for t in (stdout_task, stderr_task, *waiters):
                if not t.done():
                    t.cancel()
            raise

        timed_out = wait_task not in done and (cancel_task is None or cancel_task not in done)
        was_canceled = cancel_task is not None and cancel_task in done

        if timed_out or was_canceled or overflow_triggered["v"]:
            await _kill_process_group(proc)

        # Cleanup orphan tasks
        if cancel_task is not None and not cancel_task.done():
            cancel_task.cancel()

        # Drain reader tasks (po killu by měly skončit rychle)
        try:
            stdout_bytes = await asyncio.wait_for(stdout_task, timeout=2.0)
        except asyncio.TimeoutError:
            stdout_task.cancel()
            stdout_bytes = b""
        try:
            stderr_bytes = await asyncio.wait_for(stderr_task, timeout=2.0)
        except asyncio.TimeoutError:
            stderr_task.cancel()
            stderr_bytes = b""

        duration_ms = int((time.monotonic() - start) * 1000)
        exit_code = proc.returncode

        if was_canceled:
            return {
                "ok": False, "error": "canceled",
                "duration_ms": duration_ms, "exit_code": exit_code,
            }
        if timed_out:
            return {
                "ok": False,
                "error": f"timeout after {timeout_sec:.0f}s",
                "duration_ms": duration_ms, "exit_code": exit_code,
            }
        if overflow_triggered["v"]:
            return {
                "ok": False,
                "error": f"response exceeded {output_cap_bytes} bytes - proces zabit",
                "duration_ms": duration_ms, "exit_code": exit_code,
            }
        if exit_code != 0:
            stderr_preview = stderr_bytes[:500].decode("utf-8", errors="replace")
            return {
                "ok": False,
                "error": f"claude CLI exit code {exit_code}",
                "exit_code": exit_code,
                "stderr_preview": stderr_preview,
                "duration_ms": duration_ms,
            }

        # Parse JSON output
        try:
            payload = json.loads(stdout_bytes.decode("utf-8", errors="replace"))
        except (ValueError, UnicodeDecodeError) as e:
            return {
                "ok": False,
                "error": f"invalid JSON output: {type(e).__name__}: {e}",
                "duration_ms": duration_ms,
            }

        # Claude CLI JSON output shape:
        # { "type": "result", "subtype": "success"|"error_*",
        #   "result": "text response", "session_id": "...",
        #   "total_cost_usd": ..., "usage": {...}, "is_error": false }
        text = ""
        if isinstance(payload, dict):
            r = payload.get("result")
            if isinstance(r, str):
                text = r
            # Některé verze CLI dávají do "message.content"
            elif isinstance(payload.get("message"), dict):
                content = payload["message"].get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts = [
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    text = "".join(parts)

        is_error = isinstance(payload, dict) and payload.get("is_error") is True
        if is_error:
            return {
                "ok": False,
                "error": text or "claude CLI reported error",
                "duration_ms": duration_ms,
                "model": model,
            }

        return {
            "ok": True,
            "text": text,
            "model": model,
            "stop_reason": payload.get("stop_reason") if isinstance(payload, dict) else None,
            "session_id": payload.get("session_id") if isinstance(payload, dict) else None,
            "total_cost_usd": payload.get("total_cost_usd") if isinstance(payload, dict) else None,
            "duration_ms": duration_ms,
        }
