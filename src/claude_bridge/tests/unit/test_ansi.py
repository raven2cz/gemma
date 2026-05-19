"""Unit testy pro parsing/ansi.py."""
from __future__ import annotations

import pytest

from claude_bridge.parsing.ansi import (
    contains_black_circle,
    contains_spinner_frame,
    is_black_circle_char,
    is_reduced_motion_dot,
    strip_ansi,
)


# ──────────────── strip_ansi ────────────────

@pytest.mark.parametrize("text,expected", [
    ("Hello", "Hello"),
    ("Hello\x1b[31m World\x1b[0m", "Hello World"),
    ("\x1b[1mBold\x1b[0m", "Bold"),
    ("\x1b[?25hCursor\x1b[?25l", "Cursor"),
    ("\x1b]0;Title\x07Body", "Body"),  # OSC sequence
    ("Plain text without ANSI", "Plain text without ANSI"),
    ("", ""),
])
def test_strip_ansi(text, expected):
    assert strip_ansi(text) == expected


def test_strip_ansi_preserves_newlines():
    """\\n a \\t MUSÍ projít, jen ESC sekvence se odstraňují."""
    assert strip_ansi("line1\nline2\ttabbed") == "line1\nline2\ttabbed"


# ──────────────── BLACK_CIRCLE (tool_use marker) ────────────────

def test_black_circle_macos():
    """macOS BLACK_CIRCLE = ⏺ (U+23FA)."""
    assert is_black_circle_char("⏺")


def test_black_circle_linux():
    """Linux/Windows BLACK_CIRCLE = ● (U+25CF)."""
    assert is_black_circle_char("●")


def test_not_black_circle():
    for c in ("A", "1", " ", "○", "▶"):
        assert not is_black_circle_char(c)


def test_contains_black_circle():
    assert contains_black_circle("● Read /file.py")
    assert contains_black_circle("⏺ Bash(ls -la)")
    assert not contains_black_circle("plain text")


# ──────────────── Spinner frames ────────────────

@pytest.mark.parametrize("c", ["·", "✢", "*", "✶", "✻", "✽", "✳"])
def test_spinner_frame_chars(c):
    assert contains_spinner_frame(c)


def test_spinner_frame_in_text():
    assert contains_spinner_frame("text with · in middle")
    assert contains_spinner_frame("✽")


def test_no_spinner_in_plain_text():
    assert not contains_spinner_frame("plain ASCII text")
    assert not contains_spinner_frame("")


# ──────────────── Reduced motion dot ────────────────

def test_reduced_motion_dot():
    """Reduced motion fallback = ● (BLACK_CIRCLE on Linux). Ambiguous!"""
    assert is_reduced_motion_dot("●")
    assert not is_reduced_motion_dot("⏺")
    assert not is_reduced_motion_dot("·")
