"""Pre-flight heuristic router pro agent mode (fáze 7).

Účel: PŘED spuštěním agent loopu prozkoumat poslední user message a vrátit
doporučení, kterým modelem by se dotaz měl ideálně zpracovat (lokální Gemma
vs. `ask_claude` na Anthropic API).

DŮLEŽITÉ: tato fáze NEPŘEPÍNÁ klienta runtime — runtime stále jede přes Ollama
(Gemma). Decision se vrací jako metadata `router_decision` event do NDJSON
streamu — observability pro UI + foundation pro Phase 8 (audit log).

Heuristiky (priority order, first match wins):
  1. Explicit user directive ("@claude", "použij claude", …) → claude/high
  2. Smart-home / rychlé lokální úkony ("rozsviť", "zhasni", "git status") → local/high
  3. Code-review / komplexní reasoning ("review", "najdi bug", …) → claude/high
  4. Code block (triple-backtick) → claude/low
  5. Dlouhý prompt (> 1500 znaků) → claude/low
  6. Default → local/high

Žádné ML, žádné embeddings, žádné tokeny — jen string match. Pure function,
žádné side effects. Cílem je jednoduchá, deterministická, debug-friendly
classifikace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterDecision:
    """Výsledek pre-flight router classifikace."""
    target: str        # "local" | "claude"
    reason: str        # human-readable důvod
    confidence: str    # "low" | "high"


# Priorita 1: explicit user directive → claude/high.
# Uživatel přímo říká „použij Clauda" — žádné další heuristiky se neevaluují.
# Czech skloňování (claude/clauda/claudovi/…) přes `claud\w*`.
_RE_EXPLICIT_CLAUDE = re.compile(
    r"(?i)(?:^|\W)(?:@claude|\[claude\]|ask\s+claude|"
    r"použij\s+claud\w*|pouzij\s+claud\w*|"
    r"zeptej\s+se\s+claud\w*|"
    r"pomocí\s+claud\w*|pomoci\s+claud\w*)(?:\W|$)"
)

# Priorita 2: rychlé lokální úkony → local/high.
# Smart home + jednoduché "kolik je hodin" / "git status" / pingy.
# Pozn.: `ping` bez trailing \s — pattern `ping` v alternaci stačí, end-anchor
# v outer skupině zvládne i `ping`, i `ping 8.8.8.8` (outer (?:\W|$) matchuje
# mezeru i konec řetězce).
_RE_LOCAL_FASTPATH = re.compile(
    r"(?i)(?:^|\W)(?:"
    r"rozsviť|rozsvit|zhasni|zhasnout|stmav|"
    r"turn\s+on|turn\s+off|switch\s+(?:on|off)|"
    r"kolik\s+je\s+hodin|kolik\s+je|"
    r"jaké\s+je\s+počasí|jake\s+je\s+pocasi|what\s+(?:time|weather)|"
    r"git\s+status|git\s+log|git\s+diff|"
    r"ping"
    r")(?:\W|$)"
)

# Priorita 3: komplexní reasoning / code review → claude/high.
# Pozn.: Czech slova mají skloňování (kontrol-a/u/y, chyb-a/u/y, architekt-ura/u/ury),
# proto za root prefixem povolíme `\w*`. Czech filler tokens (mi/tento/tenhle/ten/
# tu/ho) jsou volitelně přípustné mezi sloveso ↔ object.
_CZ_FILLERS = r"(?:mi|ten|tento|tenhle|tu|tuto|tuhle|ho)\s+"
_RE_REVIEW_REASONING = re.compile(
    r"(?i)\b(?:"
    r"code\s+review|review\s+\w+|refactor\w*|"
    r"security\s+(?:audit|review)|"
    r"bezpečnostn\w*\s+(?:audit\w*|review|kontrol\w*)|"
    r"bezpecnostn\w*\s+(?:audit\w*|review|kontrol\w*)|"
    r"vysvětli\s+(?:" + _CZ_FILLERS + r")?kód\w*|"
    r"vysvetli\s+(?:" + _CZ_FILLERS + r")?kod\w*|"
    r"explain\s+(?:this|the|that|my)?\s*code\w*|"
    r"najdi\s+(?:" + _CZ_FILLERS + r")?(?:bug\w*|chyb\w*|zranitelnost\w*)|"
    r"find\s+(?:bug\w*|vuln\w*)|"
    r"architekt\w*|optimize\w*|optimalizuj\w*"
    r")\b"
)

# Priorita 4: code block marker. Pokud user paste-uje code přes ```…```,
# nejspíš jde o kódový úkol → preferuj Claude (low confidence — může to být
# i log dump).
_RE_CODE_FENCE = re.compile(r"```")

# Priorita 5: délka. Hardcoded threshold — žádný token-counter trik.
_LENGTH_THRESHOLD = 1500


def _last_user_text(messages: list[dict]) -> str:
    """Vrátí string content posledního user message nebo "" pokud žádný."""
    if not isinstance(messages, list):
        return ""
    for m in reversed(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        # Anthropic-style content blocks: list of {type, text}
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts)
        return ""
    return ""


def decide_route(messages: list[dict]) -> RouterDecision:
    """Pure routing heuristic — vrátí RouterDecision pro poslední user message.

    Žádné side effects, žádné DNS/IO, žádné LLM. Volat lze kdekoli (test,
    server, audit). Pokud `messages` nemá user roli, vrací default local/high.
    """
    text = _last_user_text(messages).strip()
    if not text:
        return RouterDecision(
            target="local",
            reason="no user text — default local",
            confidence="high",
        )

    if _RE_EXPLICIT_CLAUDE.search(text):
        return RouterDecision(
            target="claude",
            reason="explicit user directive (@claude / použij claude)",
            confidence="high",
        )

    if _RE_LOCAL_FASTPATH.search(text):
        return RouterDecision(
            target="local",
            reason="smart-home / quick local task",
            confidence="high",
        )

    if _RE_REVIEW_REASONING.search(text):
        return RouterDecision(
            target="claude",
            reason="code review / complex reasoning request",
            confidence="high",
        )

    if _RE_CODE_FENCE.search(text):
        return RouterDecision(
            target="claude",
            reason="code block present",
            confidence="low",
        )

    if len(text) > _LENGTH_THRESHOLD:
        return RouterDecision(
            target="claude",
            reason=f"long prompt (>{_LENGTH_THRESHOLD} chars)",
            confidence="low",
        )

    return RouterDecision(
        target="local",
        reason="default — no escalation signal",
        confidence="high",
    )
