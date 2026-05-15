"""Test JS regex-based mode-switch intent matcher.

Spustí node subprocess, který natáhne `voice/webapp/static/app.js`, vytáhne
regex literály `_RE_INTENT_AGENT` / `_RE_INTENT_CHAT` a aplikuje je na test
case input. Pokud node není dostupný, test se skipne (CI bez node = OK).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_JS = REPO_ROOT / "voice" / "webapp" / "static" / "app.js"


def _run_node_intent(text: str) -> str | None:
    """Vrátí "agent" | "chat" | None — výsledek matchování v reálné JS regex."""
    script = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');
// Vytáhni `const _RE_INTENT_AGENT = /.../i;` a `_RE_INTENT_CHAT`.
function extract(name) {
  const re = new RegExp("const\\s+" + name + "\\s*=\\s*(/[\\s\\S]*?/[gimsuy]*);", 'm');
  const m = src.match(re);
  if (!m) throw new Error("not found: " + name);
  // Evaluuj regex literál — safe, je to vlastní soubor, ne user input.
  return (new Function("return " + m[1] + ";"))();
}
const reAgent = extract('_RE_INTENT_AGENT');
const reChat = extract('_RE_INTENT_CHAT');
const text = process.argv[2];
if (reAgent.test(text)) process.stdout.write('agent');
else if (reChat.test(text)) process.stdout.write('chat');
else process.stdout.write('null');
"""
    node = shutil.which("node")
    if not node:
        pytest.skip("node není v PATH")
    out = subprocess.run(
        [node, "-e", script, str(APP_JS), text],
        capture_output=True, text=True, timeout=10, check=True,
    )
    result = out.stdout.strip()
    return None if result == "null" else result


# ─── Pozitivní matches → agent ─────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "agent mód",
    "agent mod",
    "agent mode",
    "agent",
    "agentní mód",
    "agentni rezim",
    "agentní režim",
    "Přepni do agent módu.",
    "prepni do agenta",
    "Přepni na agent mode",
    "aktivuj agenta",
    "aktivuj agent mód",
    "Spusť agent.",
    "zapni agent mód",
    "jdi do agent módu",
    "AGENT MÓD",
])
def test_agent_intent_positive(text):
    assert _run_node_intent(text) == "agent"


# ─── Pozitivní matches → chat ─────────────────────────────────────────


@pytest.mark.parametrize("text", [
    "chat mód",
    "chat mode",
    "chat",
    "Přepni do chatu.",
    "přepni na chat",
    "přepni do chat módu",
    "zpět do chatu",
    "zpet do chatu",
    "jdi do chatu",
    "normální mód",
    "normalni rezim",
    "CHAT",
])
def test_chat_intent_positive(text):
    assert _run_node_intent(text) == "chat"


# ─── Negativní (NESMÍ se vyhodnotit jako mode switch) ────────────────


@pytest.mark.parametrize("text", [
    "zeptej se agenta na počasí",          # full sentence, ne switch
    "co umí agent mode?",                  # otázka
    "udělej v agent módu git status",      # imperativ s payload
    "vytvoř soubor README.md",
    "ahoj",
    "kolik je hodin",
    "chat s tebou je fajn",                # "chat" jako substring v jiném významu
    "v agent módu jsi schopnější",         # výrok, ne příkaz
    "",
    "   ",
])
def test_intent_negative(text):
    assert _run_node_intent(text) is None
