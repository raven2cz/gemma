"""Sandbox primitives (voice/agent/tools/_sandbox.py).

Test priority = bezpečnost: symlink escape, traversal, special files.
Cokoli mimo workdir se *musí* dostat ven s clear error nebo False decision.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from voice.agent.tools._sandbox import (
    READ_ALLOWLIST_RESOLVED,
    is_inside_workdir,
    is_read_allowed,
    is_special_file,
    resolve_safe,
)


# ---------------------------------------------------------------------------
# resolve_safe
# ---------------------------------------------------------------------------


def test_resolve_safe_relative_path_inside(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    resolved, err = resolve_safe("x.txt", tmp_path)
    assert err is None
    assert resolved == f.resolve()


def test_resolve_safe_absolute_path(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    resolved, err = resolve_safe(str(f), tmp_path)
    assert err is None
    assert resolved == f.resolve()


def test_resolve_safe_non_existent_is_ok(tmp_path: Path):
    """Non-existent file → není error (write_file ho teprve vytvoří).
    Resolve strict=False vrátí očekávanou cestu, special check ji projde."""
    resolved, err = resolve_safe("new_file.txt", tmp_path)
    assert err is None
    assert resolved == (tmp_path / "new_file.txt").resolve()


def test_resolve_safe_empty_path_rejected(tmp_path: Path):
    resolved, err = resolve_safe("", tmp_path)
    assert resolved is None
    assert err is not None


def test_resolve_safe_non_string_rejected(tmp_path: Path):
    resolved, err = resolve_safe(None, tmp_path)  # type: ignore[arg-type]
    assert resolved is None


def test_resolve_safe_nul_byte_rejected(tmp_path: Path):
    resolved, err = resolve_safe("x\x00.txt", tmp_path)
    assert resolved is None
    assert "NUL" in err


def test_resolve_safe_traversal_normalizes_outside(tmp_path: Path):
    """`../foo` z workdir vede mimo workdir — resolve_safe to nezahazuje
    (je classifierova práce řešit containment), jen normalizuje."""
    resolved, err = resolve_safe("../escape.txt", tmp_path)
    assert err is None
    assert resolved == (tmp_path.parent / "escape.txt").resolve()
    # ale následně classifier (is_read_allowed) řekne False.


def test_resolve_safe_symlink_inside_to_outside(tmp_path: Path):
    """Symlink uvnitř workdir → soubor mimo workdir. Po resolve cíl je
    mimo workdir; classifier to chytí v is_inside_workdir."""
    target = tmp_path.parent / "outside.txt"
    target.write_text("secret")
    try:
        link = tmp_path / "link"
        os.symlink(target, link)
        resolved, err = resolve_safe("link", tmp_path)
        assert err is None
        assert resolved == target.resolve()
        # Containment check (následný krok): False.
        assert not is_inside_workdir(resolved, tmp_path)
    finally:
        target.unlink(missing_ok=True)


def test_resolve_safe_symlink_loop_rejected(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    os.symlink(b, a)
    os.symlink(a, b)
    resolved, err = resolve_safe("a", tmp_path)
    assert resolved is None
    assert err is not None


def test_resolve_safe_special_dev_null_rejected(tmp_path: Path):
    """`/dev/null` je character device → reject přes is_special_file."""
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null on this platform")
    resolved, err = resolve_safe("/dev/null", tmp_path)
    assert resolved is None
    assert "special" in err.lower() or "/dev" in err


def test_resolve_safe_proc_self_environ_rejected(tmp_path: Path):
    """`/proc/self/environ` má leak env vars (potential creds). Reject."""
    if not Path("/proc/self").exists():
        pytest.skip("no /proc on this platform")
    resolved, err = resolve_safe("/proc/self/environ", tmp_path)
    assert resolved is None
    assert err is not None


def test_resolve_safe_proc_pid_rejected(tmp_path: Path):
    if not Path("/proc/1").exists():
        pytest.skip("no /proc on this platform")
    resolved, err = resolve_safe("/proc/1/cmdline", tmp_path)
    assert resolved is None


def test_resolve_safe_allowlist_path_passes(tmp_path: Path):
    """`/proc/cpuinfo` (v allowlistu) projde — je v allowlist exception
    pro proc-pid reject."""
    if not Path("/proc/cpuinfo").exists():
        pytest.skip("no /proc/cpuinfo")
    resolved, err = resolve_safe("/proc/cpuinfo", tmp_path)
    assert err is None
    assert resolved == Path("/proc/cpuinfo").resolve()


def test_resolve_safe_fifo_rejected(tmp_path: Path):
    fifo = tmp_path / "myfifo"
    try:
        os.mkfifo(fifo)
    except (NotImplementedError, OSError):
        pytest.skip("mkfifo not supported")
    resolved, err = resolve_safe("myfifo", tmp_path)
    assert resolved is None
    assert "special" in err.lower()


# ---------------------------------------------------------------------------
# is_inside_workdir
# ---------------------------------------------------------------------------


def test_is_inside_workdir_basic(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert is_inside_workdir(f, tmp_path)


def test_is_inside_workdir_outside(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.txt"
    assert not is_inside_workdir(outside, tmp_path)


def test_is_inside_workdir_string_prefix_safe(tmp_path: Path):
    """Bezpečnostní bug: `is_relative_to` musí být safe proti `/foo/bar2`
    string-prefix match s `/foo/bar`. Vytvoříme sibling adresář se stejným
    prefix a ověříme, že není považován za inside."""
    sibling = tmp_path.parent / (tmp_path.name + "_evil")
    sibling.mkdir()
    try:
        evil_file = sibling / "x.txt"
        evil_file.write_text("nope")
        assert not is_inside_workdir(evil_file, tmp_path)
    finally:
        evil_file.unlink(missing_ok=True)
        sibling.rmdir()


# ---------------------------------------------------------------------------
# is_special_file
# ---------------------------------------------------------------------------


def test_is_special_file_dev_null():
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null")
    assert is_special_file(Path("/dev/null"))


def test_is_special_file_regular(tmp_path: Path):
    f = tmp_path / "regular.txt"
    f.write_text("ok")
    assert not is_special_file(f)


def test_is_special_file_non_existent(tmp_path: Path):
    """Neexistující soubor: lstat selže → False. Special check není pre-existence
    requirement; resolve_safe handluje cestu i tak."""
    assert not is_special_file(tmp_path / "ghost.txt")


def test_is_special_file_fifo(tmp_path: Path):
    fifo = tmp_path / "p"
    try:
        os.mkfifo(fifo)
    except (NotImplementedError, OSError):
        pytest.skip("mkfifo not supported")
    assert is_special_file(fifo)


# ---------------------------------------------------------------------------
# is_read_allowed
# ---------------------------------------------------------------------------


def test_is_read_allowed_inside(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    ok, reason = is_read_allowed(f.resolve(), tmp_path)
    assert ok
    assert "inside" in reason


def test_is_read_allowed_outside_no_allowlist(tmp_path: Path):
    outside = (tmp_path.parent / "elsewhere.txt").resolve()
    ok, reason = is_read_allowed(outside, tmp_path)
    assert not ok
    assert "outside" in reason or "allowlist" in reason


def test_is_read_allowed_allowlist_hit(tmp_path: Path):
    """`/etc/os-release` je v allowlistu (i kdyby nebyl resolved, fallback
    cesta je tam)."""
    if not Path("/etc/os-release").exists():
        pytest.skip("no /etc/os-release")
    ok, reason = is_read_allowed(Path("/etc/os-release").resolve(), tmp_path)
    assert ok
    assert "allowlist" in reason


def test_read_allowlist_resolved_contains_known():
    """Sanity: allowlist po resolve obsahuje typický `/etc/os-release` (na
    Linuxu existuje, na jiných platformách asi ne — proto skip)."""
    if not Path("/etc/os-release").exists():
        pytest.skip("no /etc/os-release")
    assert Path("/etc/os-release").resolve() in READ_ALLOWLIST_RESOLVED
