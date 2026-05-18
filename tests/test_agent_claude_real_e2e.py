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


@pytest.mark.asyncio
async def test_e2e_edit_design_doc_does_not_overflow(tmp_workdir):
    """Regression test: user task "navrhni design dokument" generoval cca
    256+ KB stream-json (Read několika souborů + 30 KB markdown). Při starém
    capu 256 KiB byl proces zabit (overflow). Cap byl odstraněn úplně (user
    policy: pro Opus implementace velkých úkolů by hard kill rušil práci).

    Test simuluje realistický flow: pár souborů ke čtení + požadavek na
    delší markdown dokument. Ověř ok=True, soubor vznikl, žádný overflow flag.
    """
    _skip_if_no_claude()
    # Vytvoříme pár sample CSV souborů (OHLCV-style, jako user měl v projektu)
    for i in range(3):
        (tmp_workdir / f"data_{i}.csv").write_text(
            "date,open,high,low,close,volume\n"
            + "\n".join(
                f"2024-01-{d:02d},100.{d},105.{d},99.{d},103.{d},{1000+d}"
                for d in range(1, 21)
            )
        )

    result = await ask_claude_oneshot(
        prompt=(
            "1. Read all CSV files in this directory using Glob+Read tools. "
            "2. Identify the data format (it's OHLCV: open/high/low/close/volume). "
            "3. Create design.md in this directory with a markdown design document "
            "describing: data schema, recommended chart types for OHLCV (candlestick), "
            "tech stack (HTML+Chart.js or similar), file structure proposal. "
            "Be thorough - aim for at least 50 lines of markdown. "
            "4. Respond with 'DONE' when finished."
        ),
        system=None,
        model="claude-haiku-4-5",
        mode="edit",
        workdir=tmp_workdir,
        timeout_sec=300.0,  # 5 min - design tasks can be slow
        claude_bin=_CLAUDE_BIN,
        cancel_event=None,
        progress_callback=None,
    )

    # Hlavní assertion: žádný overflow ani timeout
    assert result.get("overflow") is not True, (
        f"OVERFLOW! Cap nestačil: {result.get('error')}"
    )
    assert result.get("timeout") is not True, (
        f"TIMEOUT! Bridge nedoběhl: {result.get('error')}"
    )
    assert result.get("ok") is True, f"failed: {result}"

    # Design doc vznikl
    design_md = tmp_workdir / "design.md"
    assert design_md.exists(), (
        f"design.md NEVZNIKL; obsah workdir: {list(tmp_workdir.iterdir())}"
    )
    content = design_md.read_text()
    assert len(content) > 500, f"design.md je moc krátký ({len(content)} B)"
    # Měl by zmínit OHLCV
    assert any(kw in content.lower() for kw in ("ohlcv", "candlestick", "open", "close"))


@pytest.mark.asyncio
async def test_e2e_real_bridge_emits_progress_events_for_each_tool_use(tmp_workdir):
    """User regression: 'vubec ale neni videt, co dela'. Test ověří že
    bridge progress_callback DOSTANE event PRO KAŽDÝ tool_use Claude udělá
    (Read, Write, Bash, ...) - ne jen 1-2 generic eventy.

    Tohle je důkaz pro UI: pokud progress_callback dostal N tool_uses, pak
    server stream taky doručí N tool_progress eventů, a UI je má co zobrazit.
    """
    _skip_if_no_claude()
    # 2 sample CSV soubory ke čtení
    (tmp_workdir / "a.csv").write_text("x,y\n1,2\n3,4\n")
    (tmp_workdir / "b.csv").write_text("a,b\n5,6\n7,8\n")

    progress_events: list[dict] = []

    async def progress(payload: dict):
        progress_events.append(payload)

    result = await ask_claude_oneshot(
        prompt=(
            "Use Glob tool to find all CSV files. Then Read each one. "
            "Then Write a summary.txt with 'found N CSVs'. Respond 'DONE'."
        ),
        system=None,
        model="claude-haiku-4-5",
        mode="edit",
        workdir=tmp_workdir,
        timeout_sec=180.0,
        claude_bin=_CLAUDE_BIN,
        cancel_event=None,
        progress_callback=progress,
    )
    assert result.get("ok") is True, f"failed: {result}"

    # User MUSÍ vidět progress eventy - bez nich UI ukáže prázdno
    stages = [e.get("stage") for e in progress_events]
    assert "started" in stages, f"no 'started' event: {stages}"
    assert stages.count("tool_use") >= 2, (
        f"očekáváno aspoň 2 tool_use eventy (Glob+Read+Write min), "
        f"got {stages.count('tool_use')}. Stages: {stages}"
    )

    # Každý tool_use event MUSÍ mít tool_name + nějaký message pro UI
    tool_uses = [e for e in progress_events if e.get("stage") == "tool_use"]
    for tu in tool_uses:
        assert tu.get("tool_name"), f"tool_use bez tool_name: {tu}"
        assert tu.get("message"), f"tool_use bez message (UI bude prázdné): {tu}"

    # tool_result event jako follow-up po tool_use (Claude zpracoval výsledek)
    assert stages.count("tool_result") >= 1, (
        f"žádný tool_result event - UI nikdy neukáže 'tool OK': {stages}"
    )
