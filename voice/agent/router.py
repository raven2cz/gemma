"""Pre-flight heuristic router pro agent mode (fáze 7).

Účel: PŘED spuštěním agent loopu prozkoumat poslední user message a vrátit
doporučení, kterým modelem by se dotaz měl ideálně zpracovat (lokální Gemma
vs. delegace na `ask_claude` přes Claude Code CLI subagent).

DŮLEŽITÉ: tato fáze NEPŘEPÍNÁ klienta runtime - runtime stále jede přes Ollama
(Gemma). Decision se vrací jako metadata `router_decision` event do NDJSON
streamu - observability pro UI + foundation pro Phase 8 (audit log).

Heuristiky (priority order, first match wins):
  1. Explicit user directive ("@claude", "použij claude", …) → claude/high
  2. Smart-home / rychlé lokální úkony ("rozsviť", "zhasni", "git status") → local/high
  3. Code-review / komplexní reasoning ("review", "najdi bug", …) → claude/high
  4. Code block (triple-backtick) → claude/low
  5. Dlouhý prompt (> 1500 znaků) → claude/low
  6. Default → local/high

Žádné ML, žádné embeddings, žádné tokeny - jen string match. Pure function,
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
    # Volitelný hint který model uživatel řekl explicitně ("opus" / "sonnet" /
    # "haiku"). Server propaguje do ExecuteContext.model_hint a tool ho
    # použije pokud user nezadal `model` arg.
    model_hint: str | None = None


# Priorita 1: explicit user directive → claude/high. Musí být sloveso /
# imperativ - "Mluvil jsem s Claudem včera" NESMÍ matchnout (žádné akční sloveso).
# Czech skloňování (claude/clauda/claudovi/…) přes `claud\w*`.
_VERBS_DELEGATE = (
    r"použij|pouzij|"           # použij Claude/Opus/Sonnet
    r"udělej|udelej|"           # udělej (to) přes Claude
    r"zeptej\s+se|"             # zeptej se Claude
    r"ask|"                     # ask Claude
    r"pomocí|pomoci|"           # pomocí Clauda
    r"přes|pres|"               # přes Claude / přes Opus
    r"deleguj|delegate"
)
_RE_EXPLICIT_CLAUDE = re.compile(
    r"(?ix)"
    r"(?:^|\W)"
    r"(?:"
    r"  @claude | \[claude\]"
    r"  |"
    # Verb-prefixed mention claude/opus/sonnet/haiku/sonet
    r"  (?:" + _VERBS_DELEGATE + r")"
    r"  (?:\s+to)?"             # "udělej to přes Claude"
    r"  \s+"
    r"  (?:claud\w* | opus\w* | sonnet\w* | sonet\w* | haik\w*)"
    r")"
    r"(?:\W|$)"
)

# Detekce konkrétního model hintu z user věty.
_RE_MODEL_OPUS = re.compile(r"(?i)\bopus\w*\b")
_RE_MODEL_SONNET = re.compile(r"(?i)\b(?:sonnet|sonet)\w*\b")
_RE_MODEL_HAIKU = re.compile(r"(?i)\bhaik\w*\b")


def _detect_model_hint(text: str) -> str | None:
    """Vrátí "opus" / "sonnet" / "haiku" pokud user explicitně zmínil model.
    Voláno jen pokud router už rozhodl target=claude (jinak by každá zmínka
    Claudea v běžné větě nastavila hint)."""
    if _RE_MODEL_OPUS.search(text):
        return "opus"
    if _RE_MODEL_SONNET.search(text):
        return "sonnet"
    if _RE_MODEL_HAIKU.search(text):
        return "haiku"
    return None

# Priorita 2: rychlé lokální úkony → local/high.
# Smart home + jednoduché "kolik je hodin" / "git status" / pingy.
# Pozn.: `ping` bez trailing \s - pattern `ping` v alternaci stačí, end-anchor
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
# nejspíš jde o kódový úkol → preferuj Claude (low confidence - může to být
# i log dump).
_RE_CODE_FENCE = re.compile(r"```")

# Priorita 5: délka. Hardcoded threshold - žádný token-counter trik.
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
    """Pure routing heuristic - vrátí RouterDecision pro poslední user message.

    Žádné side effects, žádné DNS/IO, žádné LLM. Volat lze kdekoli (test,
    server, audit). Pokud `messages` nemá user roli, vrací default local/high.
    """
    text = _last_user_text(messages).strip()
    if not text:
        return RouterDecision(
            target="local",
            reason="no user text - default local",
            confidence="high",
        )

    if _RE_EXPLICIT_CLAUDE.search(text):
        return RouterDecision(
            target="claude",
            reason="explicit user directive (@claude / použij claude/opus/sonnet)",
            confidence="high",
            model_hint=_detect_model_hint(text),
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
        reason="default - no escalation signal",
        confidence="high",
    )
