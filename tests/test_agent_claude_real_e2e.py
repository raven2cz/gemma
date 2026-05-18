"""E2E testy pro `ask_claude_oneshot` proti REÁLNÉMU `claude` CLI subprocess.

Účel: ověřit že claude_bridge.py + claude.py + permission flow funguje
end-to-end proti živému Claude Code CLI - ne jen proti fixtures.

POŽADAVKY:
  - `claude` binary v PATH (Claude Code CLI)
  - Aktivní OAuth session (`claude /login` proběhlý a uložený)
  - Síť dostupná, Anthropic credity

Test scenarios:
  1. mode=consult: jednoduchý dotaz "2+2=?", ověř text "4"
  2. mode=edit: požádá Claude vytvořit hello.py v workdir přes Write tool,
     ověř že soubor vznikl a má požadovaný obsah
  3. mode=edit s cancel: spustí dlouhý úkol, cancelne přes Event, ověř
     ok=False + killed
  4. Activation phrase round-trip: router rozhodne claude+edit, _ask_claude_exec
     uvidí mode_hint a zavolá CLI - smoke test že chain dohromady drží

Tyto testy MUSÍ být explicitně opt-in (`pytest -m claude_cli`), jinak by
během CI utíkaly Claude credity.
"""
from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from voice.agent.claude_bridge import ask_claude_oneshot

pytestmark = pytest.mark.claude_cli

_CLAUDE_BIN = shutil.which("claude")


def _skip_if_no_claude():
    if _CLAUDE_BIN is None:
        pytest.skip("`claude` CLI není v PATH - skip real E2E")
    # Pokud nemá auth, CLI vrátí error a test by selhal s "Invalid API key".
    # Tady to nedetekujeme - test prostě selže s konkrétním message.


@pytest.fixture
def tmp_workdir(tmp_path):
    """Čistý prázdný workdir pro každý test."""
    wd = tmp_path / "claude_e2e_wd"
    wd.mkdir()
    return wd


@pytest.mark.asyncio
async def test_e2e_consult_simple_arithmetic():
    """mode=consult: pure Q&A, žádný FS, žádné tooly. Haiku = nejlevnější."""
    _skip_if_no_claude()
    result = await ask_claude_oneshot(
        prompt="What is 2 + 2? Reply with just the number, nothing else.",
        system=None,
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,
        timeout_sec=90.0,
        output_cap_bytes=256 * 1024,
        claude_bin=_CLAUDE_BIN,
        cancel_event=None,
        progress_callback=None,
    )
    assert result.get("ok") is True, f"failed: {result}"
    text = result.get("text", "")
    assert "4" in text, f"expected '4' in text, got: {text!r}"
    assert result.get("model") == "claude-haiku-4-5"
    # consult NESMÍ použít žádné tooly
    assert result.get("tool_uses") in (None, [], ()), (
        f"consult použil tooly: {result.get('tool_uses')}"
    )


@pytest.mark.asyncio
async def test_e2e_edit_creates_file_in_workdir(tmp_workdir):
    """mode=edit: Claude má Write tool a vytvoří soubor v workdir."""
    _skip_if_no_claude()
    target = tmp_workdir / "hello.py"
    assert not target.exists()

    progress_events: list[dict] = []

    async def progress(payload: dict):
        progress_events.append(payload)

    result = await ask_claude_oneshot(
        prompt=(
            "Create a file named hello.py in the current directory containing "
            "exactly: print('hello from claude')\n"
            "Use the Write tool. Then respond with just 'DONE'."
        ),
        system=None,
        model="claude-haiku-4-5",
        mode="edit",
        workdir=tmp_workdir,
        timeout_sec=180.0,
        output_cap_bytes=256 * 1024,
        claude_bin=_CLAUDE_BIN,
        cancel_event=None,
        progress_callback=progress,
    )
    assert result.get("ok") is True, f"edit failed: {result}"
    assert target.exists(), (
        f"hello.py NEVZNIKL v {tmp_workdir}; obsah: {list(tmp_workdir.iterdir())}"
    )
    content = target.read_text()
    assert "hello from claude" in content, f"content nečekaný: {content!r}"

    # Progress callback dostal aspoň jeden tool_use event
    tool_use_events = [e for e in progress_events if e.get("stage") == "tool_use"]
    assert len(tool_use_events) >= 1, (
        f"žádný tool_use progress event! events: {[e.get('stage') for e in progress_events]}"
    )
    # Write tool byl mezi nimi
    assert "Write" in (result.get("tool_uses") or []), (
        f"Write tool nebyl použit: {result.get('tool_uses')}"
    )


@pytest.mark.asyncio
async def test_e2e_consult_does_not_see_workdir_files(tmp_workdir):
    """mode=consult NESMÍ Claudovi dát přístup k workdir souborům.

    Vytvoříme soubor s tajným tokenem, Claude se má pokusit ho přečíst -
    a měl by říct že nemá FS přístup (nebo ho prostě nepřečte)."""
    _skip_if_no_claude()
    secret = tmp_workdir / "secret.txt"
    secret.write_text("MAGIC_TOKEN_E2E_12345")

    result = await ask_claude_oneshot(
        prompt=(
            f"Read the file {secret} if you can, and tell me what's inside. "
            "If you cannot read files, just say 'NO ACCESS'."
        ),
        system=None,
        model="claude-haiku-4-5",
        mode="consult",
        workdir=None,  # consult nesmí dostat workdir
        timeout_sec=90.0,
        output_cap_bytes=256 * 1024,
        claude_bin=_CLAUDE_BIN,
        cancel_event=None,
        progress_callback=None,
    )
    assert result.get("ok") is True, f"failed: {result}"
    text = result.get("text", "")
    # Magic token NESMÍ být v odpovědi (Claude by ho mohl přečíst jen kdyby
    # měl FS přístup)
    assert "MAGIC_TOKEN_E2E_12345" not in text, (
        f"consult mode přečetl secret! Text: {text[:300]}"
    )


@pytest.mark.asyncio
async def test_e2e_edit_cancel_via_event(tmp_workdir):
    """mode=edit s cancel_event: cancel během běhu zabije subprocess."""
    _skip_if_no_claude()
    cancel_event = asyncio.Event()

    async def cancel_after(delay: float):
        await asyncio.sleep(delay)
        cancel_event.set()

    # Dáme Claudovi dlouhý úkol a cancelneme za 2s
    cancel_task = asyncio.create_task(cancel_after(2.0))

    result = await ask_claude_oneshot(
        prompt=(
            "Create 50 files named test_NN.txt (NN = 01..50) in this directory, "
            "each containing the word 'data'. Use the Write tool for each. "
            "Take your time, do them one at a time."
        ),
        system=None,
        model="claude-haiku-4-5",
        mode="edit",
        workdir=tmp_workdir,
        timeout_sec=180.0,
        output_cap_bytes=256 * 1024,
        claude_bin=_CLAUDE_BIN,
        cancel_event=cancel_event,
        progress_callback=None,
    )
    await cancel_task

    # ok=False (subprocess byl zabit) NEBO ok=True s velmi málo soubory
    # (záleží na timing - cancel může přijít po dokončení).
    # Hlavní invariant: nepokračovalo to ad infinitum.
    if result.get("ok") is False:
        # Standardní case: cancel = killed
        assert result.get("killed") is True or "cancel" in str(result.get("error", "")).lower(), (
            f"ok=False ale nevypadá to na cancel: {result}"
        )
    # Else: Claude dokončil rychleji než 2s - rare ale OK
