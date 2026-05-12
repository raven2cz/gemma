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
    # Canonical resolved Path, na které classifier rozhodl (jen pro FS tooly).
    # Loop ji předá do `ExecuteContext.resolved_path`, aby tool nemusel re-resolvovat
    # a aby se eliminoval TOCTOU gap mezi check-time a exec-time.
    resolved_path: Path | None = None


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


# ----------------------------------------------------------------------
# Phase 2: File-system classifiery.
# Sandbox primitiva v `tools/_sandbox.py` — tady jen mapujeme decision.
# ----------------------------------------------------------------------


def _short(p: str | Path, n: int = 80) -> str:
    s = str(p)
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _read_style_decision(
    tool_name: str, args: dict, workdir: Path, *, summary_verb: str,
) -> PermissionResult:
    """Společná logika pro read-only tooly (read_file/list_files/glob/grep):
    sandbox resolve → DENY pokud special, AUTO uvnitř / allowlist, jinak ASK.
    """
    from voice.agent.tools._sandbox import resolve_safe, is_read_allowed

    path_str = args.get("path", "") if tool_name != "glob" else args.get("path", ".")
    if tool_name in ("glob", "grep") and not path_str:
        path_str = "."
    resolved, err = resolve_safe(path_str, workdir)
    if resolved is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=err or "invalid path",
            summary=f"{summary_verb} odmítnuto: {err}",
            risk="high",
        )
    allowed, reason = is_read_allowed(resolved, workdir)
    short = _short(resolved)
    if allowed:
        return PermissionResult(
            decision=Decision.AUTO,
            reason=reason,
            summary=f"{summary_verb}: {short}",
            risk="low",
            resolved_path=resolved,
        )
    return PermissionResult(
        decision=Decision.ASK,
        reason=reason,
        summary=f"{summary_verb} mimo workdir: {short}",
        risk="medium",
        resolved_path=resolved,
    )


@register_classifier("read_file")
def _cls_read_file(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("read_file", args, workdir, summary_verb="read")


@register_classifier("list_files")
def _cls_list_files(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("list_files", args, workdir, summary_verb="list")


@register_classifier("glob")
def _cls_glob(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("glob", args, workdir, summary_verb="glob")


@register_classifier("grep")
def _cls_grep(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("grep", args, workdir, summary_verb="grep")


def _write_style_decision(
    args: dict, workdir: Path, *, summary_verb: str,
) -> PermissionResult:
    """Společná logika pro write_file/edit_file: AUTO inside workdir,
    ASK + requires_explicit ("ano povoluju") outside (= destructive).
    """
    from voice.agent.tools._sandbox import resolve_safe, is_inside_workdir

    path_str = args.get("path", "")
    resolved, err = resolve_safe(path_str, workdir)
    if resolved is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=err or "invalid path",
            summary=f"{summary_verb} odmítnuto: {err}",
            risk="high",
        )
    short = _short(resolved)
    if is_inside_workdir(resolved, workdir):
        return PermissionResult(
            decision=Decision.AUTO,
            reason="inside workdir",
            summary=f"{summary_verb}: {short}",
            risk="low",
            resolved_path=resolved,
        )
    # Mimo workdir = destructive, vyžaduje frázi "ano povoluju".
    return PermissionResult(
        decision=Decision.ASK,
        reason="write/edit outside workdir is destructive",
        summary=f"{summary_verb} MIMO workdir: {short}",
        risk="destructive",
        requires_explicit=True,
        resolved_path=resolved,
    )


@register_classifier("write_file")
def _cls_write_file(args: dict, workdir: Path) -> PermissionResult:
    return _write_style_decision(args, workdir, summary_verb="write")


@register_classifier("edit_file")
def _cls_edit_file(args: dict, workdir: Path) -> PermissionResult:
    return _write_style_decision(args, workdir, summary_verb="edit")
