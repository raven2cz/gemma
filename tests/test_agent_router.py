"""Unit testy pro voice/agent/router.py (fáze 7 — pre-flight heuristic router).

Cílem je deterministický classifikátor — testujeme každou prioritní větev
+ edge cases (empty messages, no user role, Anthropic-style content blocks,
priority ordering).
"""
from __future__ import annotations

import pytest

from voice.agent.router import RouterDecision, decide_route


def _u(text: str) -> list[dict]:
    """Minimální messages list s jedním user messagem."""
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# Priority 1: explicit user directive → claude/high
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "@claude jaký je názor na tohle?",
    "Use [claude] for this please",
    "ask claude what he thinks",
    "Použij Claude pro tohle",
    "pouzij claude",
    "Použij Clauda na tohle",      # accusative skloňování (Gemini iter-1)
    "pouzij clauda",                # accusative bez diakritiky
    "Zeptej se Clauda na názor",
    "zeptej se claude",
    "Vyřeš to pomocí claude",
    "pomoci claude",
    "Vyřeš to pomocí clauda",       # accusative (Gemini iter-1)
    "pomoci clauda",                # accusative bez diakritiky
])
def test_priority1_explicit_directive(text):
    r = decide_route(_u(text))
    assert r.target == "claude"
    assert r.confidence == "high"
    assert "explicit" in r.reason.lower()


# ---------------------------------------------------------------------------
# Priority 2: smart home / quick local fast-path → local/high
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Rozsviť obývák",
    "rozsvit kuchyn",
    "Zhasni všechna světla",
    "Turn on the kitchen light",
    "turn off lampa",
    "switch on the floor lamp",
    "switch off all lights",
    "Kolik je hodin?",
    "kolik je",
    "Jaké je počasí v Praze?",
    "jake je pocasi",
    "what time is it",
    "what weather today",
    "git status v repu",
    "git log poslední commit",
    "git diff",
    "ping 8.8.8.8",
    "ping",
])
def test_priority2_local_fastpath(text):
    r = decide_route(_u(text))
    assert r.target == "local", f"got {r}"
    assert r.confidence == "high"


@pytest.mark.parametrize("text", [
    "ping 8.8.8.8",
    "ping google.com",
    "git status",
    "git log --oneline",
    "Rozsviť obývák",
    "Kolik je hodin?",
])
def test_priority2_uses_fastpath_reason_not_default(text):
    """Regression (Gemini iter-1): fast-path musí TRIGGEROVAT, ne padnout do default.
    Pokud spadne na default, target je sice local, ale reason ukazuje 'default…'."""
    r = decide_route(_u(text))
    assert r.target == "local"
    assert "smart-home" in r.reason or "quick local" in r.reason, f"reason={r.reason!r}"


# ---------------------------------------------------------------------------
# Priority 3: code review / complex reasoning → claude/high
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Please code review this PR",
    "Review this implementation",
    "Review the architecture",
    "Refactor the auth module",
    "Run a security audit on the codebase",
    "security review prosím",
    "Bezpečnostní audit této funkce",
    "bezpecnostni kontrola",
    "Vysvětli kód v tomto souboru",
    "vysvetli kod",
    "Vysvětli mi kód",              # filler "mi" (Gemini iter-1)
    "Vysvětli tento kód prosím",    # filler "tento"
    "vysvetli tenhle kod",          # filler "tenhle"
    "Explain this code please",
    "explain the code",
    "Najdi bug v tomhle",
    "najdi chybu tady",
    "Najdi mi bug v handleru",      # filler "mi" (Gemini iter-1)
    "Najdi mi chybu v auth flow",   # filler "mi"
    "Najdi zranitelnost",
    "find vuln in this",
    "find bug in handler",
    "Pomoz s architekturou systému",
    "Optimize this query",
    "Optimalizuj algoritmus",
])
def test_priority3_review_reasoning(text):
    r = decide_route(_u(text))
    assert r.target == "claude", f"got {r}"
    assert r.confidence == "high"


# ---------------------------------------------------------------------------
# Priority 4: code fence → claude/low
# ---------------------------------------------------------------------------


def test_priority4_code_fence():
    r = decide_route(_u("Opravím to:\n```python\nx = 1\n```\nnějaký komentář."))
    assert r.target == "claude"
    assert r.confidence == "low"
    assert "code block" in r.reason.lower()


def test_priority4_inline_backticks_no_fence_does_not_trigger():
    # Inline `code` (single backticks) by NEMĚL spustit code-fence routing.
    r = decide_route(_u("Use `os.path.join` for paths."))
    assert r.target == "local"


# ---------------------------------------------------------------------------
# Priority 5: long prompt → claude/low
# ---------------------------------------------------------------------------


def test_priority5_long_prompt():
    text = "x " * 800  # 1600 chars
    r = decide_route(_u(text))
    assert r.target == "claude"
    assert r.confidence == "low"
    assert "long" in r.reason.lower()


def test_priority5_threshold_just_below():
    text = "x" * 1500
    r = decide_route(_u(text))
    assert r.target == "local"  # 1500 ≤ threshold, NOT triggering


def test_priority5_threshold_just_above():
    text = "x" * 1501
    r = decide_route(_u(text))
    assert r.target == "claude"
    assert r.confidence == "low"


# ---------------------------------------------------------------------------
# Priority 6: default → local/high
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Ahoj jak se máš",
    "Hello there",
    "Co je dnes za den",
    "Napiš mi krátký pozdrav",
    "Jak se řekne 'dobrý den' anglicky?",
    "Naplánuj mi víkend",
])
def test_priority6_default_local(text):
    r = decide_route(_u(text))
    assert r.target == "local"
    assert r.confidence == "high"
    assert "default" in r.reason.lower()


# ---------------------------------------------------------------------------
# Priority ordering (first match wins)
# ---------------------------------------------------------------------------


def test_priority_explicit_beats_fastpath():
    # @claude má vyšší prioritu než "kolik je"
    r = decide_route(_u("@claude kolik je hodin?"))
    assert r.target == "claude"
    assert r.confidence == "high"


def test_priority_explicit_beats_review():
    r = decide_route(_u("@claude review this code"))
    assert r.target == "claude"
    # explicit, ne review → reason musí být explicit
    assert "explicit" in r.reason.lower()


def test_priority_fastpath_beats_review():
    # Pokud user řekne "zhasni" (rychlý lokál) i když má v kontextu slovo
    # "review" — fast-path zvítězí (priority 2 < priority 3).
    r = decide_route(_u("Zhasni světlo, dělám security review."))
    assert r.target == "local"


def test_priority_review_beats_code_fence():
    r = decide_route(_u("Code review:\n```py\nx=1\n```"))
    assert r.target == "claude"
    assert r.confidence == "high"  # review > code fence
    assert "review" in r.reason.lower()


def test_priority_review_beats_length():
    text = "Please review my code. " + ("a" * 2000)
    r = decide_route(_u(text))
    assert r.target == "claude"
    assert r.confidence == "high"


def test_priority_code_fence_beats_length():
    text = "Tady:\n```\n" + ("x\n" * 100) + "```\n" + ("a" * 2000)
    r = decide_route(_u(text))
    assert r.target == "claude"
    # code fence → low confidence (před length kontrolou)
    assert "code block" in r.reason.lower()


# ---------------------------------------------------------------------------
# Edge cases: empty / missing user messages
# ---------------------------------------------------------------------------


def test_empty_messages_list():
    r = decide_route([])
    assert r.target == "local"
    assert r.confidence == "high"
    assert "no user text" in r.reason.lower()


def test_no_user_message_only_system():
    r = decide_route([{"role": "system", "content": "you are a bot"}])
    assert r.target == "local"


def test_only_assistant_and_tool_no_user():
    r = decide_route([
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "{}", "tool_call_id": "x"},
    ])
    assert r.target == "local"


def test_whitespace_only_user_message():
    r = decide_route(_u("   \n\t  "))
    assert r.target == "local"
    assert "no user text" in r.reason.lower()


def test_non_string_content_ignored():
    r = decide_route([{"role": "user", "content": 123}])
    assert r.target == "local"


def test_non_dict_in_messages_skipped():
    r = decide_route(["bogus", {"role": "user", "content": "ahoj"}])
    assert r.target == "local"


def test_non_list_messages_handled():
    r = decide_route("not a list")  # type: ignore[arg-type]
    assert r.target == "local"


def test_uses_LAST_user_message():
    # Pokud je víc user messages, použije se POSLEDNÍ (= novější dotaz).
    msgs = [
        {"role": "user", "content": "@claude review please"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "kolik je hodin"},
    ]
    r = decide_route(msgs)
    assert r.target == "local"  # poslední je fast-path


# ---------------------------------------------------------------------------
# Anthropic-style content blocks (list of {type, text})
# ---------------------------------------------------------------------------


def test_content_blocks_concatenated():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Please "},
            {"type": "text", "text": "review my code"},
        ],
    }]
    r = decide_route(msgs)
    assert r.target == "claude"
    assert "review" in r.reason.lower()


def test_content_blocks_with_non_text_skipped():
    msgs = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {}},
            {"type": "text", "text": "rozsviť obývák"},
        ],
    }]
    r = decide_route(msgs)
    assert r.target == "local"


def test_content_blocks_empty_list():
    msgs = [{"role": "user", "content": []}]
    r = decide_route(msgs)
    assert r.target == "local"
    assert "no user text" in r.reason.lower()


# ---------------------------------------------------------------------------
# Dataclass sanity
# ---------------------------------------------------------------------------


def test_router_decision_is_frozen():
    r = RouterDecision(target="local", reason="x", confidence="high")
    with pytest.raises(Exception):
        r.target = "claude"  # type: ignore[misc]


def test_router_decision_fields_present():
    r = decide_route(_u("ahoj"))
    assert isinstance(r.target, str)
    assert isinstance(r.reason, str)
    assert isinstance(r.confidence, str)
    assert r.target in ("local", "claude")
    assert r.confidence in ("low", "high")
    # model_hint je volitelný (None pro lokální)
    assert r.model_hint is None or isinstance(r.model_hint, str)


# ─────────────── v2: aktivační fráze pro explicit Claude/Opus/Sonnet ───────────────


@pytest.mark.parametrize("text,expected_hint", [
    # user-explicit "použij Claude"
    ("Použij Claude na tento problém", None),
    ("pouzij Clauda", None),
    # opus
    ("Použij Opus pro toto code review", "opus"),
    ("použij opus", "opus"),
    # sonnet
    ("Použij Sonnet, je rychlejší", "sonnet"),
    ("pouzij Sonet", "sonnet"),
    # "udělej přes Claude/Opus/Sonnet"
    ("Udělej to přes Claude", None),
    ("Udelej to pres Opus", "opus"),
    ("Udělej přes Sonnet, ať to máme rychle", "sonnet"),
    # @claude / [claude]
    ("@claude review my code", None),
    ("[claude] expert opinion needed", None),
    # ask claude
    ("ask Claude what's the best approach", None),
])
def test_router_explicit_claude_directives(text, expected_hint):
    """Všechny tyto věty MUSÍ být routovány na claude/high."""
    r = decide_route(_u(text))
    assert r.target == "claude", f"text={text!r}: got target={r.target}, reason={r.reason}"
    assert r.confidence == "high"
    assert r.model_hint == expected_hint, f"text={text!r}: hint={r.model_hint}"


@pytest.mark.parametrize("text", [
    # Žádné akční sloveso → NEMĚLO by triggernout
    "Mluvil jsem o Claudovi včera",
    "Claude je dobrý model",
    "Slyšel jsi o Opusu?",
    "Sonnet je rychlejší než Opus",
])
def test_router_no_false_positive_on_passive_mention(text):
    """Verb-prefixed regex nesmí matchnout pasivní zmínky."""
    r = decide_route(_u(text))
    # Tyto věty by neměly skončit jako claude/high (explicit directive)
    if r.target == "claude":
        assert r.confidence == "low", (
            f"text={text!r}: got claude/high (false positive!)"
        )
