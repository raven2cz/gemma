"""Test permission resolver (voice/agent/permissions.py)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from voice.agent.permissions import Decision, decide


def test_echo_auto(tmp_path: Path):
    r = decide("echo", {"text": "hello"}, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"
    assert "hello" in r.summary


def test_unknown_tool_denied(tmp_path: Path):
    r = decide("no_such_tool", {}, tmp_path)
    assert r.decision == Decision.DENY
    assert "no_such_tool" in r.summary or "no_such_tool" in r.reason


def test_echo_summary_truncated(tmp_path: Path):
    long_text = "x" * 200
    r = decide("echo", {"text": long_text}, tmp_path)
    assert r.decision == Decision.AUTO
    # summary nemá obsahovat plnou délku
    assert len(r.summary) < 200


# ---------------------------------------------------------------------------
# Phase 2: FS classifiery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool", ["read_file", "list_files", "glob", "grep"])
def test_read_tools_auto_inside(tmp_path: Path, tool: str):
    args = {"path": str(tmp_path)} if tool != "glob" else {"pattern": "*", "path": str(tmp_path)}
    if tool == "grep":
        args = {"pattern": "x", "path": str(tmp_path)}
    r = decide(tool, args, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"


@pytest.mark.parametrize("tool", ["read_file", "list_files", "glob", "grep"])
def test_read_tools_ask_outside(tmp_path: Path, tool: str):
    outside = tmp_path.parent / "elsewhere"
    args = {"path": str(outside)}
    if tool == "glob":
        args = {"pattern": "*", "path": str(outside)}
    elif tool == "grep":
        args = {"pattern": "x", "path": str(outside)}
    r = decide(tool, args, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "medium"


def test_read_file_allowlist_auto(tmp_path: Path):
    if not Path("/etc/os-release").exists():
        pytest.skip("no /etc/os-release")
    r = decide("read_file", {"path": "/etc/os-release"}, tmp_path)
    assert r.decision == Decision.AUTO


def test_read_file_special_denied(tmp_path: Path):
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null")
    r = decide("read_file", {"path": "/dev/null"}, tmp_path)
    assert r.decision == Decision.DENY
    assert r.risk == "high"


def test_read_file_proc_self_denied(tmp_path: Path):
    if not Path("/proc/self").exists():
        pytest.skip("no /proc")
    r = decide("read_file", {"path": "/proc/self/environ"}, tmp_path)
    assert r.decision == Decision.DENY


def test_read_file_symlink_outside_asks(tmp_path: Path):
    """Symlink uvnitř workdir → soubor mimo workdir.
    Classifier po resolve uvidí outside → ASK (ne AUTO)."""
    outside = tmp_path.parent / "outside_classifier.txt"
    outside.write_text("data")
    try:
        link = tmp_path / "link"
        os.symlink(outside, link)
        r = decide("read_file", {"path": "link"}, tmp_path)
        assert r.decision == Decision.ASK
    finally:
        outside.unlink(missing_ok=True)


def test_write_file_auto_inside(tmp_path: Path):
    r = decide("write_file", {"path": "x.txt", "content": "hi"}, tmp_path)
    assert r.decision == Decision.AUTO
    assert r.risk == "low"
    assert r.requires_explicit is False


def test_write_file_destructive_outside(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.txt"
    r = decide("write_file", {"path": str(outside), "content": "x"}, tmp_path)
    assert r.decision == Decision.ASK
    assert r.risk == "destructive"
    assert r.requires_explicit is True


def test_edit_file_auto_inside(tmp_path: Path):
    r = decide(
        "edit_file",
        {"path": "x.py", "old_string": "a", "new_string": "b"},
        tmp_path,
    )
    assert r.decision == Decision.AUTO


def test_edit_file_destructive_outside(tmp_path: Path):
    outside = tmp_path.parent / "elsewhere.py"
    r = decide(
        "edit_file",
        {"path": str(outside), "old_string": "a", "new_string": "b"},
        tmp_path,
    )
    assert r.decision == Decision.ASK
    assert r.requires_explicit is True
    assert r.risk == "destructive"


def test_write_file_special_path_denied(tmp_path: Path):
    if not Path("/dev/null").exists():
        pytest.skip("no /dev/null")
    r = decide(
        "write_file", {"path": "/dev/null", "content": "x"}, tmp_path
    )
    assert r.decision == Decision.DENY


def test_classifier_invalid_path_denied(tmp_path: Path):
    """Empty path → DENY (resolve_safe odmítne)."""
    r = decide("read_file", {"path": ""}, tmp_path)
    assert r.decision == Decision.DENY


def test_read_file_summary_uses_short_path(tmp_path: Path):
    """Velmi dlouhá cesta v summary musí být zkrácená (≤ 80 znaků)."""
    deep = tmp_path
    for i in range(20):
        deep = deep / f"verylongdirname_{i:02d}"
    deep.mkdir(parents=True, exist_ok=True)
    f = deep / "x.txt"
    f.write_text("hi")
    r = decide("read_file", {"path": str(f)}, tmp_path)
    assert r.decision == Decision.AUTO
    # Summary "read: <path>" — část za "read: " je truncated
    after_prefix = r.summary.split(":", 1)[1]
    assert len(after_prefix.strip()) <= 82  # 80 + leading "…"
