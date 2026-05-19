"""Comprehensive real-CLI testy pro TmuxAdapter parsing.

Cílem je pokrýt MNOHO scénářů aby parsing byl bullet-proof:
- Single/multi-line responses
- Tool flows: Read/Write/Edit/Bash/Glob/Grep
- Edge cases: long responses, wrapped output, code blocks, errors
- Session lifecycle: continuity, /clear, multiple turns
- Stress: 5-10 turnů v jedné session

Každý test stvíčí spawne fresh tmux+claude session a verifikuje konkrétní
behavior. Markery: claude_cli + tmux_real (skip bez tmux/claude).
"""
from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import pytest

from claude_bridge.adapters.tmux_mode import TmuxAdapter
from claude_bridge.config import AdapterConfig, BridgeMode
from claude_bridge.progress import ProgressEvent

pytestmark = [pytest.mark.claude_cli, pytest.mark.tmux_real]

_HAS_TMUX = shutil.which("tmux") is not None
_HAS_CLAUDE = shutil.which("claude") is not None
if not _HAS_TMUX:
    pytest.skip("tmux not in PATH", allow_module_level=True)
if not _HAS_CLAUDE:
    pytest.skip("claude not in PATH", allow_module_level=True)


@pytest.fixture
async def adapter(tmp_path):
    """Per-test fresh adapter s dedicated tmux socket + cleanup."""
    config = AdapterConfig(
        mode=BridgeMode.TMUX,
        tmux_session_prefix="cmp_claude_",
        metadata_dir=str(tmp_path),
        default_timeout_sec=180.0,
    )
    a = TmuxAdapter(config)
    yield a
    # Cleanup
    for sid in list(a._sessions.keys()):
        try:
            await a._kill_tmux_session(sid)
        except Exception:
            pass
    rc, stdout, _ = await a._tmux("list-sessions", "-F", "#{session_name}")
    if rc == 0:
        for line in stdout.decode().splitlines():
            if line.strip().startswith("cmp_claude_"):
                await a._kill_tmux_session(line.strip())
    await a._tmux("kill-server", timeout=5.0)


# ──────────────── Group 1: Simple Q&A varianty ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_qa_arithmetic(adapter):
    """Basic math."""
    r = await adapter.ask(
        prompt="What is 7 * 8? Reply with just the number.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=45.0,
    )
    assert r.ok, f"failed: {r}"
    assert "56" in r.text, f"text: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_qa_capital_city(adapter):
    """Geographic Q&A."""
    r = await adapter.ask(
        prompt="What is the capital of France? Reply with just the city name.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=45.0,
    )
    assert r.ok, f"failed: {r}"
    assert "paris" in r.text.lower(), f"text: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_qa_multiline_response(adapter, tmp_path):
    """Response s newlines (list output). Edit mode (acceptEdits) má volnost
    odpovědět přímo bez plan-mode clarification dialogu."""
    wd = tmp_path / "multi_test"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "List the first 5 prime numbers separated by newlines. "
            "Output format: just the numbers, one per line, nothing else."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    for n in ("2", "3", "5", "7", "11"):
        assert n in r.text, f"missing {n} in: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_qa_czech_response(adapter):
    """Response v češtině (Unicode)."""
    r = await adapter.ask(
        prompt="Řekni česky 'Ahoj světe' a nic víc.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=45.0,
    )
    assert r.ok, f"failed: {r}"
    assert "ahoj" in r.text.lower(), f"text: {r.text!r}"


# ──────────────── Group 2: Edit mode tool flows ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_write_single_file(adapter, tmp_path):
    """Write tool: vytvoří soubor s konkrétním obsahem."""
    wd = tmp_path / "write_test"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "Use the Write tool to create greet.txt containing exactly: hello tmux\n"
            "Then respond 'DONE'."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    target = wd / "greet.txt"
    assert target.exists(), f"file not created: {list(wd.iterdir())}"
    assert "hello tmux" in target.read_text()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_read_existing_file(adapter, tmp_path):
    """Read tool: přečte soubor, vrátí jeho obsah."""
    wd = tmp_path / "read_test"
    wd.mkdir()
    (wd / "data.txt").write_text("MAGIC_TOKEN_42")
    r = await adapter.ask(
        prompt=(
            "Use the Read tool to read data.txt. Then respond with the exact content."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    assert "MAGIC_TOKEN_42" in r.text, f"content not in response: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_edit_existing_file(adapter, tmp_path):
    """Edit tool: změní specific string v souboru."""
    wd = tmp_path / "edit_test"
    wd.mkdir()
    target = wd / "config.txt"
    target.write_text("version: 1.0\nname: original\n")
    r = await adapter.ask(
        prompt=(
            "Use the Edit tool to replace 'original' with 'modified' in config.txt. "
            "Respond 'DONE'."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    content = target.read_text()
    assert "modified" in content, f"edit failed: {content!r}"
    assert "original" not in content


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_tool_multi_write(adapter, tmp_path):
    """Multi Write: 3 soubory v jednom turnu."""
    wd = tmp_path / "multi_write"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "Use the Write tool to create three files in this dir:\n"
            "- a.txt with content: alpha\n"
            "- b.txt with content: beta\n"
            "- c.txt with content: gamma\n"
            "Then respond 'DONE'."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=90.0,
    )
    assert r.ok, f"failed: {r}"
    assert (wd / "a.txt").read_text().strip() == "alpha"
    assert (wd / "b.txt").read_text().strip() == "beta"
    assert (wd / "c.txt").read_text().strip() == "gamma"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_bash_command(adapter, tmp_path):
    """Bash tool: spustí příkaz, použije output."""
    wd = tmp_path / "bash_test"
    wd.mkdir()
    (wd / "file1.txt").touch()
    (wd / "file2.txt").touch()
    (wd / "file3.txt").touch()
    r = await adapter.ask(
        prompt=(
            "Use the Bash tool to run 'ls -1 *.txt | wc -l' in the current dir. "
            "Respond with the number from the output."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    assert "3" in r.text, f"no '3' in response: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_tool_glob(adapter, tmp_path):
    """Glob tool: hledání souborů podle pattern."""
    wd = tmp_path / "glob_test"
    wd.mkdir()
    (wd / "test_one.py").touch()
    (wd / "test_two.py").touch()
    (wd / "other.txt").touch()
    r = await adapter.ask(
        prompt=(
            "Use the Glob tool to find all *.py files in this directory. "
            "Then respond with the count of .py files found."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    assert "2" in r.text, f"no '2' in response: {r.text!r}"


# ──────────────── Group 3: Long responses (parsing stress) ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_long_response_markdown(adapter, tmp_path):
    """Velmi dlouhá odpověď (2+ KB markdown). Parsing nesmí truncate.

    Edit mode (acceptEdits) - Claude má volnost přímo napsat doc do souboru
    + odpověd s preview. Plan mode by zatahoval do clarification dialog."""
    wd = tmp_path / "doc_test"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "Write a thorough markdown design document for a Todo App REST API. "
            "Include: title heading, 5+ endpoints (GET/POST/PUT/DELETE), each "
            "with request/response JSON examples, authentication section with "
            "JWT example, error handling section with HTTP status codes. "
            "Use code blocks. Use Write tool to save it as design.md. "
            "Then output the full design doc content in response."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=120.0,
    )
    assert r.ok, f"failed: {r}"
    # Soubor by měl vzniknout
    design_md = wd / "design.md"
    if design_md.exists():
        content = design_md.read_text()
        assert len(content) > 1500, f"design.md moc krátký: {len(content)} chars"
    else:
        # Pokud Write nepustil, alespoň response by měla obsahovat doc
        assert len(r.text) > 1500, f"response moc krátká: {len(r.text)} chars"


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_response_with_code_blocks(adapter):
    """Response obsahuje code blocks (``` markers, indented lines).
    Parser nesmí code spadnout jako tool_use."""
    r = await adapter.ask(
        prompt=(
            "Show me a Python function that calculates factorial. "
            "Use a code block."
        ),
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=60.0,
    )
    assert r.ok, f"failed: {r}"
    # Reálný code block markers
    assert "def" in r.text.lower() or "factorial" in r.text.lower(), \
        f"no python code in response: {r.text!r}"


# ──────────────── Group 4: Session lifecycle ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_session_3_turns_continuity(adapter):
    """3 turny ve stejné session - 3. má kombinovat info z 1. + 2."""
    r1 = await adapter.ask(
        prompt="My name is Alice. Acknowledge with 'OK'.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=60.0,
    )
    assert r1.ok
    sid = r1.session_id

    r2 = await adapter.ask(
        prompt="My favorite hobby is painting. Acknowledge with 'OK'.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        session_id=sid, timeout_sec=60.0,
    )
    assert r2.ok
    assert r2.session_id == sid

    r3 = await adapter.ask(
        prompt="What is my name and my hobby? Answer in one sentence.",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        session_id=sid, timeout_sec=60.0,
    )
    assert r3.ok
    text_l = r3.text.lower()
    assert "alice" in text_l, f"name lost: {r3.text!r}"
    assert "paint" in text_l, f"hobby lost: {r3.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_session_5_turns_no_drift(adapter):
    """5 turnů v rychlém sledu - parser nesmí ztratit context (session_id
    stable napříč turny, completion footer detekovaný correctly).

    Pozn: Claude v plan mode (consult) preferuje code-related questions.
    Test ptá na různé Python concepts, ověř že session_id zůstává konstantní."""
    sid = None
    prompts = [
        "What does Python's `len()` function do? One sentence.",
        "What does Python's `range()` do? One sentence.",
        "What does Python's `print()` do? One sentence.",
        "What does Python's `len()` again? Brief reminder.",
        "What does Python's `str()` do? One sentence.",
    ]
    for i, p in enumerate(prompts):
        r = await adapter.ask(
            prompt=p,
            model="claude-haiku-4-5", mode="consult", workdir=None,
            session_id=sid, timeout_sec=45.0,
        )
        assert r.ok, f"turn {i} failed: {r}"
        if i == 0:
            sid = r.session_id
            assert sid is not None
        else:
            assert r.session_id == sid, \
                f"session_id changed: turn {i}, was {sid}, now {r.session_id}"
        # Each turn musí vrátit non-empty text response
        assert len(r.text) > 20, f"turn {i} text too short: {r.text!r}"


# ──────────────── Group 5: Progress events accuracy ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_progress_events_count(adapter, tmp_path):
    """Edit task s 3 file operations → ≥3 tool_use events."""
    wd = tmp_path / "progress_test"
    wd.mkdir()
    events: list[ProgressEvent] = []

    async def callback(ev: ProgressEvent) -> None:
        events.append(ev)

    r = await adapter.ask(
        prompt=(
            "Create three files: x.txt, y.txt, z.txt, each with content 'data'. "
            "Use Write tool for each. Respond 'DONE'."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=90.0,
        progress_callback=callback,
    )
    assert r.ok, f"failed: {r}"
    tool_use_events = [e for e in events if e.stage == "tool_use"]
    # Aspoň 3 tool_use (3 Write calls)
    assert len(tool_use_events) >= 3, \
        f"expected ≥3 tool_use events, got {len(tool_use_events)}: " \
        f"{[(e.tool_name, e.message) for e in tool_use_events]}"


# ──────────────── Group 6: Edge cases ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_response_single_word(adapter):
    """Velmi krátká response (1 slovo)."""
    r = await adapter.ask(
        prompt="Reply with exactly one word: yes",
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=45.0,
    )
    assert r.ok, f"failed: {r}"
    assert "yes" in r.text.lower(), f"text: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_special_chars_in_prompt(adapter):
    """Prompt obsahuje special chars (quotes, slashes, ampersands)."""
    r = await adapter.ask(
        prompt='What does "foo&bar" mean? Reply with one sentence.',
        model="claude-haiku-4-5", mode="consult", workdir=None,
        timeout_sec=45.0,
    )
    assert r.ok, f"failed: {r}"
    # Just check že odpověď je něco non-trivial
    assert len(r.text) > 10, f"text too short: {r.text!r}"


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_response_with_table(adapter, tmp_path):
    """Response obsahuje table (long lines, parser nesmí truncate na border).

    Edit mode (acceptEdits) - Claude má volnost vykreslit tabulku přímo,
    plan mode by ho zatáhl do meta-planning workflow.
    """
    wd = tmp_path / "table_test"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "Output a 3-row markdown comparison table for Python, Rust, and Go. "
            "Columns: Name | Type | Year. Just output the table, no commentary."
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=150.0,
    )
    assert r.ok, f"failed: {r}"
    # Parser MUSÍ zachytit Python+Rust+Go všechny tři (= long lines + table syntax)
    text_l = r.text.lower()
    assert "python" in text_l, f"missing Python: {r.text!r}"
    assert "rust" in text_l, f"missing Rust: {r.text!r}"
    assert "go" in text_l, f"missing Go: {r.text!r}"
    # Table separator char (markdown | OR Unicode box drawing)
    table_chars = "|│┌┬┐├┼┤└┴┘─"
    assert any(c in r.text for c in table_chars), \
        f"no table chars: {r.text!r}"


# ──────────────── Group 7: Error scenarios ────────────────

@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_read_nonexistent_file_recovers(adapter, tmp_path):
    """Read tool na neexistující soubor → Claude error recovery + odpoví.

    Edit mode má acceptEdits, ale error recovery overhead = ~30-60s.
    """
    wd = tmp_path / "err_test"
    wd.mkdir()
    r = await adapter.ask(
        prompt=(
            "Try to Read the file 'missing_xyz_123.txt' in this directory. "
            "It doesn't exist. After the Read fails, simply respond: "
            "FILE_NOT_FOUND"
        ),
        model="claude-haiku-4-5", mode="edit", workdir=wd, timeout_sec=150.0,
    )
    assert r.ok, f"failed: {r}"
    text_l = r.text.lower()
    assert "not_found" in text_l or "not found" in text_l \
        or "doesn't exist" in text_l or "no such" in text_l, \
        f"no error mention: {r.text!r}"
