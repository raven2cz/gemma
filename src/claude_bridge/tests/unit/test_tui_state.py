"""Unit testy pro parsing/tui_state.py.

Tyto testy simulují reálný Claude TUI screen content. Glyph references:
- Tool_use marker: ● (Linux) / ⏺ (macOS)
- Prompt indicator: ❯ (Unix, figures.pointer)
- Spinner frames: · ✢ * ✶ ✻ ✽
"""
from __future__ import annotations

import pytest

from claude_bridge.parsing.tui_state import TuiState


# ──────────────── Tool_use detection ────────────────

def test_tool_use_linux_marker():
    """Linux BLACK_CIRCLE ● + Read tool."""
    state = TuiState(cols=80, rows=24)
    state.feed("● Read /path/to/file.py\n")
    tools = state.poll_tool_uses()
    assert len(tools) == 1
    assert tools[0].tool_name == "Read"


def test_tool_use_macos_marker():
    """macOS BLACK_CIRCLE ⏺ + Bash tool."""
    state = TuiState(cols=120, rows=24)
    state.feed("⏺ Bash(ls -la)\n")
    tools = state.poll_tool_uses()
    assert len(tools) == 1
    assert tools[0].tool_name == "Bash"
    assert tools[0].args_preview == "ls -la"


def test_tool_use_multiple_in_sequence():
    """Multi tool_use events stacked."""
    state = TuiState(cols=120, rows=30)
    state.feed(
        "● Glob *.py\n"
        "  ⎿  Found 12 files\n"
        "● Read /a.py\n"
        "  ⎿  Read 50 lines\n"
        "● Write /b.py\n"
    )
    tools = state.poll_tool_uses()
    assert {t.tool_name for t in tools} == {"Glob", "Read", "Write"}


def test_tool_use_deduplication():
    """Tool_use už reportované se nevrací znovu při dalším polling."""
    state = TuiState(cols=80, rows=24)
    state.feed("● Read /file.py\n")
    first = state.poll_tool_uses()
    assert len(first) == 1
    # Druhé polling bez nového obsahu - stejný state
    second = state.poll_tool_uses()
    assert second == []


def test_tool_use_no_match_in_plain_text():
    """● musí být následované capital-letter tool name."""
    state = TuiState(cols=80, rows=24)
    state.feed("Regular text with ● bullet point, not tool\n")
    tools = state.poll_tool_uses()
    assert len(tools) == 0


# ──────────────── Ready prompt detection ────────────────

def test_is_ready_with_pointer_glyph_at_bottom():
    """❯ (figures.pointer) na bottom = Claude waiting."""
    state = TuiState(cols=80, rows=10)
    # Fill screen so ❯ is in last 5 rows (rows 5-9)
    state.feed(
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        "Line 6\nLine 7\nLine 8\nLine 9\n❯ "
    )
    assert state.is_ready() is True


def test_is_ready_no_pointer():
    state = TuiState(cols=80, rows=24)
    state.feed("Just running text, no prompt yet\n")
    assert state.is_ready() is False


def test_is_ready_windows_fallback():
    """`>` fallback pokud Windows / dumb terminal."""
    state = TuiState(cols=80, rows=10)
    state.feed("\n" * 8 + "> ")
    assert state.is_ready() is True


# ──────────────── Reset ────────────────

def test_reset_clears_state():
    state = TuiState(cols=80, rows=24)
    state.feed("● Read /file.py\n")
    state.poll_tool_uses()
    state.reset()
    # Po reset: znova feed stejný content → nový tool_use detection
    state.feed("● Read /file.py\n")
    tools = state.poll_tool_uses()
    assert len(tools) == 1


# ──────────────── Idle detection ────────────────

def test_check_idle_increments_when_unchanged():
    state = TuiState(cols=80, rows=24)
    state.feed("Hello\n")
    state.check_idle()  # first call sets baseline
    n2 = state.check_idle()  # no change
    n3 = state.check_idle()  # still no change
    assert n2 >= 1
    assert n3 >= n2


def test_check_idle_resets_on_change():
    state = TuiState(cols=80, rows=24)
    state.feed("Hello\n")
    state.check_idle()
    state.check_idle()
    state.feed("More text\n")
    n = state.check_idle()
    assert n == 0  # reset when screen changed


# ──────────────── Screen access ────────────────

def test_screen_lines_returns_text():
    state = TuiState(cols=20, rows=3)
    state.feed("hello\nworld\n")
    lines = state.screen_lines
    assert len(lines) == 3
    assert "hello" in lines[0]
    assert "world" in lines[1]


def test_cursor_pos():
    state = TuiState(cols=80, rows=24)
    state.feed("abc")
    x, y = state.cursor_pos
    assert x == 3
    assert y == 0
