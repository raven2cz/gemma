"""Permission resolver pro tool calls.

`decide(tool_name, args, workdir)` vrátí `PermissionResult` s rozhodnutím
AUTO / ASK / DENY. Třída rozhodnutí závisí na nástroji a argumentech.

Default pro neznámý tool = DENY (bezpečnost > UX). Tooly registrují vlastní
classifier dekorátorem `@register_classifier(name)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


class Decision(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


Risk = str  # "low" | "medium" | "high" | "destructive"


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str
    summary: str           # user-facing CZ string ("echo: hello")
    risk: Risk = "low"
    requires_explicit: bool = False   # destruktivní → potřeba „ano povoluju"


Classifier = Callable[[dict, Path], PermissionResult]

_CLASSIFIERS: dict[str, Classifier] = {}


def register_classifier(name: str) -> Callable[[Classifier], Classifier]:
    def wrap(fn: Classifier) -> Classifier:
        _CLASSIFIERS[name] = fn
        return fn
    return wrap


def decide(tool_name: str, args: dict, workdir: Path) -> PermissionResult:
    fn = _CLASSIFIERS.get(tool_name)
    if fn is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=f"no classifier registered for {tool_name!r}",
            summary=f"Neznámý nástroj {tool_name!r}",
            risk="high",
        )
    return fn(args, workdir)


# ----------------------------------------------------------------------
# Built-in classifiers (Phase 1 = echo only; další fáze rozšíří).
# ----------------------------------------------------------------------


@register_classifier("echo")
def _echo(args: dict, workdir: Path) -> PermissionResult:
    text = str(args.get("text", ""))[:60]
    return PermissionResult(
        decision=Decision.AUTO,
        reason="echo has no side effects",
        summary=f'echo: "{text}"',
        risk="low",
    )
