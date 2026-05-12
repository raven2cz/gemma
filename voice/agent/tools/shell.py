"""Bash shell tool — `run_bash(command, cwd?)`.

Bezpečnostní model (synchronizováno s classifierem `_cls_run_bash` v permissions.py):

- **AUTO** (no shell metas + root v allowlistu): argv mode, shell=False. Tool re-detekuje
  shell metas a re-parseuje argv (TOCTOU defense — netrustuje classifieru).
- **ASK / requires_explicit** (gate via approval flow): /bin/bash -c mode, shell=False
  ale skrz explicit bash wrapper.
- **Vždy** (oba módy):
    * cwd uvnitř WORKDIR (resolve_safe + is_inside_workdir + is_special_file re-check)
    * env scrub: BASH_ENV_ALLOWLIST + LC_* (drop AGENT_*, API keys, tokens, …)
    * fixní PATH (anti PATH-injection)
    * stdin = DEVNULL (anti interaktivní readline ze sudo/ssh)
    * start_new_session=True (= os.setsid, vlastní process group → killpg)
    * graceful: SIGTERM → wait 2s → SIGKILL na deadline / cancel / output cap
    * output cap: stdout 768 KiB, stderr 256 KiB (součet = BASH_OUTPUT_CAP_BYTES)
"""
from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import threading
import time
from functools import partial
from pathlib import Path

from voice.agent.config import (
    BASH_ENV_ALLOWLIST,
    BASH_OUTPUT_CAP_BYTES,
    BASH_PATH,
    BASH_TIMEOUT_SEC,
)
from voice.agent.permissions import _has_shell_metas
from voice.agent.tools._sandbox import (
    is_inside_workdir,
    is_special_file,
    resolve_safe,
)
from voice.agent.tools.base import ExecuteContext, Tool

# Per-stream caps. Sum = BASH_OUTPUT_CAP_BYTES. Stdout dominant (3/4).
_STDOUT_CAP = (BASH_OUTPUT_CAP_BYTES * 3) // 4
_STDERR_CAP = BASH_OUTPUT_CAP_BYTES - _STDOUT_CAP

# SIGTERM → SIGKILL grace window.
_KILL_GRACE_SEC = 2.0

# Polling interval pro proc.poll() / cancel_event / deadline check.
_POLL_INTERVAL_SEC = 0.1

# Read chunk velikost.
_READ_CHUNK = 65536


def _scrub_env() -> dict[str, str]:
    """Vrátí scrubovaný env: keep jen BASH_ENV_ALLOWLIST + LC_*, plus fixní PATH.

    Defense in depth proti leakování secretů (AGENT_*, *_API_KEY, *_TOKEN, AWS_*)
    do shell commandu, který je řízen LLM agentem.
    """
    src = os.environ
    out: dict[str, str] = {}
    for k, v in src.items():
        if k in BASH_ENV_ALLOWLIST or k.startswith("LC_"):
            out[k] = v
    out["PATH"] = BASH_PATH
    # PWD je obvykle z parent shellu — přepíšeme korektní hodnotou níže
    # (subprocess defaultně nedostává PWD; bash si ho sám nastaví podle cwd).
    out.pop("PWD", None)
    out.pop("OLDPWD", None)
    return out


def _reader_thread(stream, cap: int, out: dict, key: str) -> None:
    """Read up to `cap` bytes from binary stream. Result do `out[key]`.

    Pokud cap dosažen → thread exit (stream se uzavře až smrt procesu).
    Main thread monitoruje out[key][1] (truncated flag) a může proces zabít,
    aby nehnijili na blocked write do plné pipe.
    """
    data = bytearray()
    truncated = False
    try:
        while True:
            try:
                chunk = stream.read(_READ_CHUNK)
            except (OSError, ValueError):
                # ValueError: stream closed; OSError: pipe broken.
                break
            if not chunk:
                break
            remaining = cap - len(data)
            if len(chunk) > remaining:
                data.extend(chunk[:remaining])
                truncated = True
                break
            data.extend(chunk)
    finally:
        out[key] = (bytes(data), truncated)


def _kill_pg(pgid: int, sig: int) -> None:
    """killpg s tichým swallowem chyby (process už mrtvý / nelze)."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _run_subprocess_blocking(
    command: str,
    use_shell: bool,
    cwd: Path,
    env: dict[str, str],
    deadline: float,
    cancel_event: asyncio.Event | None,
) -> dict:
    """Blocking subprocess execution — pouští se v thread executoru.

    Vrací dict {stdout, stderr, exit_code, killed, truncated, duration_ms}.
    Nikdy neraisí — všechny chyby se promítají do exit_code=-1 + stderr.
    """
    start = time.monotonic()

    if use_shell:
        argv: list[str] = ["/bin/bash", "-c", command]
    else:
        try:
            argv = shlex.split(command)
        except ValueError as e:
            return {
                "stdout": "",
                "stderr": f"shlex parse error: {e}",
                "exit_code": -1,
                "killed": False,
                "truncated": False,
                "duration_ms": int((time.monotonic() - start) * 1000),
            }
        if not argv:
            return {
                "stdout": "",
                "stderr": "empty argv after shlex",
                "exit_code": -1,
                "killed": False,
                "truncated": False,
                "duration_ms": int((time.monotonic() - start) * 1000),
            }

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd),
            env=env,
            start_new_session=True,  # setsid → vlastní process group
            close_fds=True,
        )
    except (OSError, FileNotFoundError) as e:
        return {
            "stdout": "",
            "stderr": f"launch failed: {type(e).__name__}: {e}",
            "exit_code": -1,
            "killed": False,
            "truncated": False,
            "duration_ms": int((time.monotonic() - start) * 1000),
        }

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    out_buf: dict[str, tuple[bytes, bool]] = {}
    t_stdout = threading.Thread(
        target=_reader_thread,
        args=(proc.stdout, _STDOUT_CAP, out_buf, "stdout"),
        daemon=True,
    )
    t_stderr = threading.Thread(
        target=_reader_thread,
        args=(proc.stderr, _STDERR_CAP, out_buf, "stderr"),
        daemon=True,
    )
    t_stdout.start()
    t_stderr.start()

    killed = False
    try:
        while True:
            if proc.poll() is not None:
                break
            now = time.monotonic()
            if now > deadline:
                killed = True
                break
            if cancel_event is not None and cancel_event.is_set():
                killed = True
                break
            # Output budget exhausted? Některý stream je truncated → kill aby
            # proces neviselo na blocked write do plné pipe.
            so = out_buf.get("stdout")
            se = out_buf.get("stderr")
            if (so and so[1]) or (se and se[1]):
                killed = True
                break
            time.sleep(_POLL_INTERVAL_SEC)

        if killed:
            _kill_pg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=_KILL_GRACE_SEC)
            except subprocess.TimeoutExpired:
                pass
            # Unconditional SIGKILL na celou PG — i když parent skončil rychle,
            # background děti (`sleep 1000 &`) by jinak přežily jako orphans.
            _kill_pg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=_KILL_GRACE_SEC)
            except subprocess.TimeoutExpired:
                pass

        # Wait pro reader threads aby flushnuly. Streamy dostanou EOF až proc
        # zemře (= teď). Hard timeout 2s; pokud thread visí, daemon ho ukončí.
        t_stdout.join(timeout=2.0)
        t_stderr.join(timeout=2.0)

        # Final returncode collection (může být None pokud reaper neproběhl).
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            _kill_pg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

        stdout_bytes, stdout_trunc = out_buf.get("stdout", (b"", False))
        stderr_bytes, stderr_trunc = out_buf.get("stderr", (b"", False))
        truncated = stdout_trunc or stderr_trunc
        exit_code = proc.returncode if proc.returncode is not None else -1

        return {
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "exit_code": exit_code,
            "killed": killed,
            "truncated": truncated,
            "duration_ms": int((time.monotonic() - start) * 1000),
        }
    finally:
        # Belt & suspenders — pokud něco vybouchlo nad tím, ať nezůstane zombie.
        if proc.poll() is None:
            _kill_pg(pgid, signal.SIGKILL)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        # Streamy zavřít explicitně (subprocess.Popen jinak čeká na __del__).
        for s in (proc.stdout, proc.stderr):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass


async def _execute(args: dict, ctx: ExecuteContext) -> dict:
    """run_bash execute. Re-validuje command a cwd (defense in depth — classifier
    už ověřil, ale re-check zabraňuje TOCTOU mezi check-time a exec-time)."""
    command = str(args.get("command", "")).strip()
    cwd_arg = str(args.get("cwd", ""))

    if not command:
        return {"ok": False, "error": "empty command"}

    if cwd_arg:
        resolved_cwd, err = resolve_safe(cwd_arg, ctx.workdir)
        if resolved_cwd is None:
            return {"ok": False, "error": f"invalid cwd: {err}"}
        if is_special_file(resolved_cwd):
            return {"ok": False, "error": "cwd is a special file"}
        if not is_inside_workdir(resolved_cwd, ctx.workdir):
            return {"ok": False, "error": "cwd outside workdir"}
        if not resolved_cwd.is_dir():
            return {"ok": False, "error": "cwd is not a directory"}
    else:
        resolved_cwd = ctx.workdir

    use_shell = _has_shell_metas(command)
    deadline = time.monotonic() + BASH_TIMEOUT_SEC
    env = _scrub_env()

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        partial(
            _run_subprocess_blocking,
            command=command,
            use_shell=use_shell,
            cwd=resolved_cwd,
            env=env,
            deadline=deadline,
            cancel_event=ctx.cancel_event,
        ),
    )
    result["ok"] = True
    result["command"] = command
    result["cwd"] = str(resolved_cwd)
    return result


TOOL = Tool(
    name="run_bash",
    description=(
        "Run a bash shell command. Safe read-only commands (pwd, ls, git status, "
        "find without -delete/-exec, cat, wc, head, tail, rg) run immediately. "
        "Other commands require user approval. Destructive commands (rm, sudo, "
        "mkfs, chmod, systemctl, ...) require an explicit Czech approval phrase. "
        "stdout/stderr are capped at ~1 MiB total; timeout 30s. cwd is always "
        "inside the project working directory."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute. Pipes, redirects, $() require user approval.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Optional working directory inside the project (relative or absolute). "
                    "Defaults to the project root. Must resolve inside the project."
                ),
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    execute=_execute,
)
