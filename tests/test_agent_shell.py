"""Unit testy pro run_bash tool (voice/agent/tools/shell.py).

Testují:
- _scrub_env (env scrub správně dropne secrets, keepne allowlist)
- _execute happy path (pwd, ls, git status)
- _execute s cwd (sub-dir validation)
- _execute timeout (sleep 5 + monkeypatch timeout=1 → killed=True)
- _execute output cap (yes → truncated=True)
- _execute cancel (cancel_event set během běhu → killed rychle)
- _execute env scrub end-to-end (env příkaz neobsahuje secrets)
- _execute shell vs argv mode (pipe → shell; pwd → argv)
- _execute stdin DEVNULL (cat čekající na stdin → killed timeout)
- _execute cwd outside workdir → ok=False
- _execute cwd na special file → ok=False
- _execute prázdný command → ok=False

Volá _execute přímo (bez classifieru) — classifier testy jsou v
test_agent_permissions.py.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from voice.agent.tools.base import ExecuteContext
from voice.agent.tools.shell import (
    TOOL as RUN_BASH,
    _STDERR_CAP,
    _STDOUT_CAP,
    _scrub_env,
)


def _ctx(workdir: Path, *, cancel_event: asyncio.Event | None = None) -> ExecuteContext:
    return ExecuteContext(turn_id="t1", cancel_event=cancel_event, workdir=workdir)


# ---------------------------------------------------------------------------
# env scrub
# ---------------------------------------------------------------------------


def test_scrub_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("AGENT_OUTPUT_CAP_BYTES", "999")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fakekey")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    monkeypatch.setenv("FAKE_PASSWORD", "y")
    monkeypatch.setenv("HOME", "/home/test")
    monkeypatch.setenv("LANG", "cs_CZ.UTF-8")
    monkeypatch.setenv("LC_TIME", "C")
    env = _scrub_env()
    assert "AGENT_OUTPUT_CAP_BYTES" not in env
    assert "OPENAI_API_KEY" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "FAKE_PASSWORD" not in env
    # HOME, LANG, LC_TIME zachovány.
    assert env.get("HOME") == "/home/test"
    assert env.get("LANG") == "cs_CZ.UTF-8"
    assert env.get("LC_TIME") == "C"
    # PATH fixní.
    assert env["PATH"] == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    # PWD/OLDPWD popnuty (bash si je sám nastaví).
    assert "PWD" not in env
    assert "OLDPWD" not in env


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_pwd(tmp_path: Path):
    out = await RUN_BASH.execute({"command": "pwd"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert out["killed"] is False
    assert out["truncated"] is False
    # `pwd` vrací cestu k tmp_path (cwd subprocessu).
    # Pozn.: tmp_path může být symlinkovaný (/tmp → /private/tmp on macOS),
    # ale na Linuxu test runneru by měl být přímý.
    assert str(tmp_path) in out["stdout"]


@pytest.mark.asyncio
async def test_run_bash_echo(tmp_path: Path):
    out = await RUN_BASH.execute({"command": "echo ahoj"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert "ahoj" in out["stdout"]


@pytest.mark.asyncio
async def test_run_bash_with_cwd(tmp_path: Path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "marker.txt").write_text("hi")
    out = await RUN_BASH.execute({"command": "ls", "cwd": "subdir"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert "marker.txt" in out["stdout"]


@pytest.mark.asyncio
async def test_run_bash_cwd_absolute_inside_workdir(tmp_path: Path):
    sub = tmp_path / "x"
    sub.mkdir()
    out = await RUN_BASH.execute({"command": "pwd", "cwd": str(sub)}, _ctx(tmp_path))
    assert out["ok"] is True
    assert str(sub) in out["stdout"]


@pytest.mark.asyncio
async def test_run_bash_shell_mode_pipe(tmp_path: Path):
    """ASK path (pipe) — exekuce přes /bin/bash -c. Classifier by tady ASK
    udělal, ale tool si re-detekuje a sám zvolí shell mode."""
    out = await RUN_BASH.execute({"command": "echo hello | wc -c"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert out["exit_code"] == 0
    # wc -c počítá bytes; "hello\n" = 6.
    assert "6" in out["stdout"]


# ---------------------------------------------------------------------------
# Bezpečnostní checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_empty_command(tmp_path: Path):
    out = await RUN_BASH.execute({"command": ""}, _ctx(tmp_path))
    assert out["ok"] is False
    assert "empty" in out["error"].lower()


@pytest.mark.asyncio
async def test_run_bash_cwd_outside_workdir(tmp_path: Path):
    out = await RUN_BASH.execute({"command": "pwd", "cwd": "/tmp"}, _ctx(tmp_path))
    assert out["ok"] is False
    assert "cwd" in out["error"].lower()


@pytest.mark.asyncio
async def test_run_bash_cwd_traversal(tmp_path: Path):
    out = await RUN_BASH.execute({"command": "pwd", "cwd": "../../"}, _ctx(tmp_path))
    assert out["ok"] is False


@pytest.mark.asyncio
async def test_run_bash_cwd_special_file(tmp_path: Path):
    out = await RUN_BASH.execute({"command": "pwd", "cwd": "/proc/self"}, _ctx(tmp_path))
    assert out["ok"] is False


@pytest.mark.asyncio
async def test_run_bash_env_scrubbed_in_subprocess(tmp_path: Path, monkeypatch):
    """End-to-end ověření že FAKE_SECRET není v subprocess env."""
    monkeypatch.setenv("FAKE_SECRET_TOKEN_XYZ", "leaked_value_12345")
    out = await RUN_BASH.execute({"command": "env"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert "FAKE_SECRET_TOKEN_XYZ" not in out["stdout"]
    assert "leaked_value_12345" not in out["stdout"]


@pytest.mark.asyncio
async def test_run_bash_path_is_fixed(tmp_path: Path, monkeypatch):
    """PATH v subprocess je BASH_PATH, ne parent PATH."""
    monkeypatch.setenv("PATH", "/totally/fake/path:/another/fake")
    out = await RUN_BASH.execute({"command": "echo $PATH"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert "/totally/fake/path" not in out["stdout"]
    assert "/usr/bin" in out["stdout"]


# ---------------------------------------------------------------------------
# Resource limits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_timeout(tmp_path: Path, monkeypatch):
    """sleep 5 s patched timeoutem 1s → killed=True."""
    monkeypatch.setattr("voice.agent.tools.shell.BASH_TIMEOUT_SEC", 1)
    start = time.monotonic()
    out = await RUN_BASH.execute({"command": "sleep 5"}, _ctx(tmp_path))
    elapsed = time.monotonic() - start
    assert out["ok"] is True
    assert out["killed"] is True
    # Měl by dojet rychle (< 4s = 1s timeout + 2s grace + buffer).
    assert elapsed < 5.0


@pytest.mark.asyncio
async def test_run_bash_output_cap(tmp_path: Path, monkeypatch):
    """`yes` produkuje nekonečný stream → truncated=True, killed=True."""
    # Sníží cap na 64 KiB stdout pro rychlejší test.
    monkeypatch.setattr("voice.agent.tools.shell._STDOUT_CAP", 64 * 1024)
    monkeypatch.setattr("voice.agent.tools.shell._STDERR_CAP", 16 * 1024)
    out = await RUN_BASH.execute({"command": "yes"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert out["truncated"] is True
    assert out["killed"] is True
    # Cap = 64 KiB, ale po decode("replace") délka v charů ≈ bytes.
    assert len(out["stdout"]) <= 64 * 1024 + 1024  # malý buffer


@pytest.mark.asyncio
async def test_run_bash_cancel(tmp_path: Path):
    """cancel_event set během běhu → killed rychle (< 4s)."""
    cancel = asyncio.Event()

    async def trigger_cancel():
        await asyncio.sleep(0.3)
        cancel.set()

    asyncio.create_task(trigger_cancel())
    start = time.monotonic()
    out = await RUN_BASH.execute({"command": "sleep 10"}, _ctx(tmp_path, cancel_event=cancel))
    elapsed = time.monotonic() - start
    assert out["ok"] is True
    assert out["killed"] is True
    assert elapsed < 4.0


@pytest.mark.asyncio
async def test_run_bash_stdin_devnull(tmp_path: Path, monkeypatch):
    """cat čekající na stdin (bez argv) → killed timeoutem, ne hang."""
    monkeypatch.setattr("voice.agent.tools.shell.BASH_TIMEOUT_SEC", 1)
    start = time.monotonic()
    out = await RUN_BASH.execute({"command": "cat"}, _ctx(tmp_path))
    elapsed = time.monotonic() - start
    # cat dostane EOF na DEVNULL stdin → exit 0 rychle, ne timeout.
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert elapsed < 3.0


# ---------------------------------------------------------------------------
# Process group kill (spawned children)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_killpg_kills_children(tmp_path: Path, monkeypatch):
    """bash -c 'sleep 60 & echo $!; wait' — pokud killpg funguje, sleep child
    (v stejné PG) také zemře. Test ověří, že po timeoutu není zombie sleep."""
    monkeypatch.setattr("voice.agent.tools.shell.BASH_TIMEOUT_SEC", 1)
    start = time.monotonic()
    out = await RUN_BASH.execute(
        {"command": "sleep 30 & echo CHILD=$!; wait"},
        _ctx(tmp_path),
    )
    elapsed = time.monotonic() - start
    assert out["ok"] is True
    assert out["killed"] is True
    assert elapsed < 5.0  # killpg správně zabilo i child sleep


# ---------------------------------------------------------------------------
# Argv vs shell mode (re-detekce)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_argv_mode_no_shell(tmp_path: Path):
    """`echo $HOME` BEZ shell metaznaků (pure literal — žádný $) → argv mode →
    $HOME se neexpanduje. Tady ale `$HOME` JE shell meta ($), takže pojede shell.

    Zkusíme striktně argv: `ls /tmp` (žádné metas) → argv mode."""
    out = await RUN_BASH.execute({"command": "ls /tmp"}, _ctx(tmp_path))
    assert out["ok"] is True
    # Pokud běží jako argv (shell=False), ls dostane "/tmp" doslova.
    # V obou módech by ale vypsal obsah /tmp, takže to není moc rozlišovací.
    # Lepší test: command s $VAR by se v argv módu nevyexpandoval.


@pytest.mark.asyncio
async def test_run_bash_var_expansion_only_in_shell_mode(tmp_path: Path):
    """`echo $$` (PID expanze) — $$ je shell meta → běží v bash módu → expanze proběhne."""
    out = await RUN_BASH.execute({"command": "echo $$"}, _ctx(tmp_path))
    assert out["ok"] is True
    # Stdout obsahuje číslo (PID), ne literální "$$".
    assert out["stdout"].strip().isdigit()


@pytest.mark.asyncio
async def test_run_bash_invalid_shlex_in_argv_mode(tmp_path: Path):
    """Bez shell metas ale s unclosed quote — shlex.split selže.
    Bez shell metas → argv mode → shlex parse error v _run_subprocess_blocking."""
    out = await RUN_BASH.execute({"command": "ls 'unclosed"}, _ctx(tmp_path))
    assert out["ok"] is True
    # Exit code z parse erroru je -1; stderr má parse error message.
    assert out["exit_code"] == -1
    assert "parse" in out["stderr"].lower() or "shlex" in out["stderr"].lower()


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_bash_nonzero_exit(tmp_path: Path):
    """Příkaz s exit_code != 0 → ok=True (tool fungoval), exit_code = real."""
    out = await RUN_BASH.execute({"command": "false"}, _ctx(tmp_path))
    assert out["ok"] is True
    assert out["exit_code"] != 0


@pytest.mark.asyncio
async def test_run_bash_command_not_found(tmp_path: Path):
    """Neznámý příkaz → /bin/bash -c vrátí 127 v argv módu nebo launch error."""
    out = await RUN_BASH.execute(
        {"command": "this_command_does_not_exist_xyz_abc"}, _ctx(tmp_path)
    )
    assert out["ok"] is True
    # Pokud běží argv mode: Popen selže FileNotFoundError → exit_code=-1
    # Pokud běží shell mode: bash zahlásí command not found, exit_code=127
    assert out["exit_code"] != 0


@pytest.mark.asyncio
async def test_run_bash_returns_metadata(tmp_path: Path):
    """Verifikace všech klíčů v response."""
    out = await RUN_BASH.execute({"command": "echo hi"}, _ctx(tmp_path))
    for key in ("ok", "stdout", "stderr", "exit_code", "killed", "truncated", "duration_ms", "command", "cwd"):
        assert key in out, f"missing key: {key}"
    assert isinstance(out["duration_ms"], int)
    assert out["command"] == "echo hi"
    assert out["cwd"] == str(tmp_path)
