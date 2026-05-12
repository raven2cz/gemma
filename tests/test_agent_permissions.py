"""Test permission resolver (voice/agent/permissions.py)."""
from __future__ import annotations

from pathlib import Path

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
