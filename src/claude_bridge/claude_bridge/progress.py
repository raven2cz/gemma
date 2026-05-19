"""ProgressEvent data model emit-ovaný adapterem během `ask()` běhu.

Caller passes async callback do ask(progress_callback=...); adapter ho volá
s ProgressEvent instancemi jak detekuje stage v Claude streamu (print mode:
NDJSON eventy; tmux mode: pyte screen state changes).

Stage hodnoty:
  - "started":     Claude session inicializovaný (session_id + model)
  - "thinking":    Claude generuje thinking blok / extended thinking
  - "tool_use":    Claude vyvolá tool (Read, Write, Bash, Glob, Grep, Edit...)
  - "tool_result": Outcome předchozího tool_use (ok / chyba)
  - "text":        Streamed assistant text delta (best-effort, ne všechny adaptery)
  - "cost":        Cost update (typically jen v print mode - tmux nemá total_cost_usd)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any

ProgressStage = Literal[
    "started",
    "thinking",
    "tool_use",
    "tool_result",
    "text",
    "cost",
]


@dataclass(frozen=True)
class ProgressEvent:
    """Single progress update z adapter.ask() během běhu."""
    stage: ProgressStage
    message: str = ""

    # tool_use specific
    tool_name: str | None = None
    input: dict[str, Any] | None = None

    # tool_result specific
    ok: bool | None = None

    # text streaming (rare - ne všechny adaptery)
    text: str = ""

    # started event
    session_id: str | None = None
    model: str | None = None

    # cost event (jen print mode)
    cost_usd: float | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Backwards-compat serializer (matches starý dict-based progress payload).
        Voice agent loop wrapping `tool_progress` events používá dict klíče,
        takže adapter dnes emituje dict; tato funkce konverze pro budoucí
        čisté ProgressEvent passing."""
        out: dict[str, Any] = {"stage": self.stage}
        if self.message:
            out["message"] = self.message
        if self.tool_name is not None:
            out["tool_name"] = self.tool_name
        if self.input is not None:
            out["input"] = self.input
        if self.ok is not None:
            out["ok"] = self.ok
        if self.text:
            out["text"] = self.text
        if self.session_id is not None:
            out["session_id"] = self.session_id
        if self.model is not None:
            out["model"] = self.model
        if self.cost_usd is not None:
            out["cost_usd"] = self.cost_usd
        if self.duration_ms is not None:
            out["duration_ms"] = self.duration_ms
        return out


@dataclass(frozen=True)
class SessionInfo:
    """Metadata o adapter session (jen relevant pro persistent adaptery
    jako tmux - print mode session info je vždy fresh per ask())."""
    session_id: str
    created_at: float            # unix epoch
    last_active: float
    workdir: str | None          # str místo Path (JSON-friendly)
    model: str
    permission_mode: str         # "consult" | "edit"
    destructive_approved: bool
    state: str                   # SessionState literal (READY/RUNNING/CANCELING/DEAD)
    turn_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
