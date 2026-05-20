"""Regression: bridge MUSI odmitnout threading.Event jako cancel_event.

Bug history (2026-05-19):
Server passed threading.Event() to PrintModeAdapter.ask. Bridge did
`asyncio.create_task(cancel_event.wait())`. For threading.Event, .wait()
is a sync-blocking call - froze entire asyncio loop. Claude subprocess ran
to completion but NO progress events, NO _read_stream execution, no
heartbeat. UI sat for 88+ seconds with zero activity.
"""
from __future__ import annotations

import asyncio
import threading

import pytest

from claude_bridge.adapters.print_mode import ask_claude_oneshot


@pytest.mark.asyncio
async def test_threading_event_rejected() -> None:
    with pytest.raises(TypeError, match="cancel_event must be asyncio.Event"):
        await ask_claude_oneshot(
            prompt="x",
            model="claude-haiku-4-5",
            mode="consult",
            workdir=None,
            timeout_sec=1.0,
            cancel_event=threading.Event(),
        )


@pytest.mark.asyncio
async def test_asyncio_event_accepted() -> None:
    ev = asyncio.Event()
    ev.set()
    result = await ask_claude_oneshot(
        prompt="x",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=1.0,
        cancel_event=ev,
        claude_bin="/nonexistent-claude-bin",
    )
    assert isinstance(result, dict)
    assert result.get("ok") is False


@pytest.mark.asyncio
async def test_none_accepted() -> None:
    result = await ask_claude_oneshot(
        prompt="x",
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=1.0,
        cancel_event=None,
        claude_bin="/nonexistent-claude-bin",
    )
    assert isinstance(result, dict)
    assert result.get("ok") is False
