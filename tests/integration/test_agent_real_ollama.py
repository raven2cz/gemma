"""Integrační testy agent mode proti REÁLNÉ Ollamě.

Proč existují: všechny ostatní agent e2e testy mockují Ollamu. Mock ale
nereplikuje chování Ollama chat-template parseru — a přesně tam žil
root-cause bug: `tool_call.function.arguments` poslané jako JSON STRING
způsobily HTTP 400 `Value looks like object, but can't find closing '}'
symbol` při druhém round-tripu (= po každém použití toolu). Mockované testy
to neodhalily; tyto testy ano.

Tyto testy se SKIPnou, pokud:
  - Ollama neběží na :11434, nebo
  - není dostupný žádný tool-capable model.

Spuštění explicitně:
    ./voice/.venv-tts/bin/python -m pytest tests/integration/ -v
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import httpx
import pytest

from voice.agent.loop import AgentLoop
from voice.agent.tools import default_registry

OLLAMA = "http://localhost:11434"

# Preferované tool-capable modely (v pořadí priority). Test použije první
# dostupný. gemma4* je cílový model projektu; qwen2.5 je spolehlivý fallback.
_TOOL_MODELS = (
    "gemma4-e4b-32k", "gemma4-26b-32k", "gemma4:e4b", "gemma4:26b",
    "qwen2.5:14b", "qwen2.5", "llama3.1", "llama3.2",
)


def _available_model() -> str | None:
    """Vrátí název prvního dostupného tool-capable modelu, nebo None."""
    try:
        r = httpx.get(f"{OLLAMA}/api/tags", timeout=2.0)
        r.raise_for_status()
    except Exception:
        return None
    installed = {m["name"] for m in r.json().get("models", [])}
    # Přesná shoda nebo prefix shoda (qwen2.5 → qwen2.5:14b)
    for pref in _TOOL_MODELS:
        if pref in installed:
            return pref
        for name in installed:
            if name.startswith(pref):
                return name
    # Fallback: jakýkoli nainstalovaný model (i kdyby tools nezvládal —
    # test pak selže informativně, ne falešně projde).
    return next(iter(installed), None)


_MODEL = _available_model()
_skip_reason = "Ollama nedostupná nebo žádný model nainstalován"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_MODEL is None, reason=_skip_reason),
]


def _new_turn_state() -> dict:
    return {
        "id": "integ",
        "canceled": False,
        "cancel_event": asyncio.Event(),
        "approvals": {},
    }


async def _run_agent(prompt: str, workdir: Path) -> tuple[list[dict], AgentLoop]:
    """Spustí jeden agent turn proti reálné Ollamě. Auto-approve resolver
    (ASK se nezablokuje). Vrací (events, loop)."""
    system = (
        "Jsi agent v terminálu. Máš tooly pro práci se soubory a shell. "
        "Když uživatel požádá o akci se soubory, POUŽIJ příslušný tool. "
        "Neptej se na potvrzení, rovnou volej tool."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    loop = AgentLoop(
        model=_MODEL,
        messages=messages,
        registry=default_registry("agent"),
        turn_state=_new_turn_state(),
        workdir=workdir,
    )

    async def _auto_approve(approval_id: str, event: dict) -> bool:
        return True

    loop.set_approval_resolver(_auto_approve)
    events = [ev async for ev in loop.run()]
    return events, loop


def _assert_no_ollama_400(events: list[dict]) -> None:
    """Root-cause regrese guard: žádný agent_error obsahující Ollama 400.

    Konkrétně hlídá `Value looks like object, but can't find closing '}'
    symbol` — to byl symptom str-arguments bugu.
    """
    errors = [e for e in events if e.get("type") == "agent_error"]
    for e in errors:
        msg = str(e.get("msg", ""))
        assert "400" not in msg, f"Ollama 400 (root-cause regrese!): {msg}"
        assert "closing '}'" not in msg, f"str-args regrese: {msg}"
    # Žádný agent_error vůbec — turn musí proběhnout čistě.
    assert not errors, f"agent_error v turnu: {errors}"


def _assert_history_args_are_dict(loop: AgentLoop) -> None:
    """Každý tool_call v history MUSÍ mít arguments jako dict (Ollama spec)."""
    for m in loop.messages:
        for tc in (m.get("tool_calls") or []):
            args = tc.get("function", {}).get("arguments")
            assert isinstance(args, dict), (
                f"history tool_call arguments není dict: {type(args).__name__} "
                f"= {args!r} — reálná Ollama by na to vrátila HTTP 400"
            )


@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_real_ollama_create_file_with_test():
    """User scénář #1: "vytvoř soubor s TEST". Plný turn proti reálné Ollamě.

    Ověřuje:
      - tool_call write_file proběhne,
      - soubor REÁLNĚ vznikne na disku s obsahem TEST,
      - round 2 (po tool_result) NEVYHODÍ Ollama 400 — turn dokončí,
      - history má arguments jako dict.
    """
    with tempfile.TemporaryDirectory(prefix="integ_agent_") as td:
        workdir = Path(td)
        events, loop = await _run_agent(
            "Vytvoř soubor test.txt s obsahem TEST", workdir,
        )

        _assert_no_ollama_400(events)
        _assert_history_args_are_dict(loop)

        types = [e["type"] for e in events]
        assert "tool_call" in types, f"žádný tool_call — model nevolal tool: {types}"
        assert "tool_result" in types
        assert types[-1] == "agent_done", f"turn nedoběhl čistě: {types}"

        # tool_call musí být write_file
        tc = next(e for e in events if e["type"] == "tool_call")
        assert tc["name"] == "write_file", f"očekáván write_file, dostal {tc['name']}"

        # tool_result OK
        tr = next(e for e in events if e["type"] == "tool_result")
        assert tr["ok"] is True, f"write_file selhal: {tr['content']}"

        # Soubor REÁLNĚ existuje s obsahem TEST
        target = workdir / "test.txt"
        assert target.exists(), f"soubor nevznikl v {workdir}"
        assert target.read_text().strip() == "TEST", (
            f"špatný obsah: {target.read_text()!r}"
        )


@pytest.mark.timeout(180)
@pytest.mark.asyncio
async def test_real_ollama_list_directory():
    """User scénář #2: výpis adresáře. Reálný LLM je nedeterministický —
    může vybrat run_bash / list_files / glob, může nejdřív zavolat s chybnými
    args a self-korigovat. HARD assert proto jen na regresní guardy (žádné
    Ollama 400, dict args, čisté dokončení); tool-specifika jsou soft.
    """
    with tempfile.TemporaryDirectory(prefix="integ_agent_") as td:
        workdir = Path(td)
        (workdir / "marker_file.txt").write_text("x")

        events, loop = await _run_agent(
            "Vypiš obsah aktuálního adresáře", workdir,
        )

        # HARD: regresní guardy (root-cause bug = Ollama 400 v round-tripu)
        _assert_no_ollama_400(events)
        _assert_history_args_are_dict(loop)
        types = [e["type"] for e in events]
        assert types[-1] == "agent_done", f"turn nedoběhl čistě: {types}"

        # SOFT: model OBVYKLE zavolá tool, ale není to garantované (může jen
        # textově odpovědět). Pokud tool zavolal, ověř že to je rozumný tool.
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        for tc in tool_calls:
            assert tc["name"] in ("run_bash", "list_files", "glob", "read_file"), (
                f"neočekávaný tool pro výpis adresáře: {tc['name']}"
            )


@pytest.mark.timeout(240)
@pytest.mark.asyncio
async def test_real_ollama_multi_step_write_then_read():
    """User scénář #3: víc tool kol za sebou (write → read → finální text).

    Toto je nejtvrdší regrese guard: KAŽDÉ tool kolo přidává do history
    `assistant.tool_calls`, takže round 3 posílá history se DVĚMA tool_calls.
    Pokud by arguments byly string, Ollama by 400 vyhodila už v round 2.
    """
    with tempfile.TemporaryDirectory(prefix="integ_agent_") as td:
        workdir = Path(td)
        events, loop = await _run_agent(
            "Vytvoř soubor data.txt s textem 'ahoj svete', pak ho přečti "
            "a řekni mi co v něm je",
            workdir,
        )

        _assert_no_ollama_400(events)
        _assert_history_args_are_dict(loop)

        types = [e["type"] for e in events]
        assert types[-1] == "agent_done", f"turn nedoběhl: {types}"
        # Alespoň jeden tool_call proběhl (model může write+read spojit nebo ne)
        tool_calls = [e for e in events if e["type"] == "tool_call"]
        assert tool_calls, f"žádný tool_call: {types}"
        # write_file musí být mezi nimi
        names = {e["name"] for e in tool_calls}
        assert "write_file" in names, f"write_file nevolán, jen: {names}"

        # Soubor reálně vznikl
        target = workdir / "data.txt"
        assert target.exists(), f"data.txt nevznikl v {workdir}"
        assert "ahoj svete" in target.read_text()
