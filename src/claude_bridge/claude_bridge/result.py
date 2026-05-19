"""ClaudeResult - výsledek `adapter.ask()` volání.

Shared napříč všemi adaptery, drop-in kompatibilní (subset toho co
voice/agent/claude_bridge.py dnes vrací jako dict).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any

Mode = Literal["consult", "edit"]


@dataclass(frozen=True)
class ClaudeResult:
    """Výsledek single `adapter.ask()` volání."""
    ok: bool
    mode: Mode = "consult"
    text: str = ""                       # finální assistant text
    model: str = ""                      # claude-opus-4-7 / sonnet / haiku / ...
    session_id: str | None = None        # adapter-specific (může být None pro print)
    total_cost_usd: float | None = None  # None pro tmux (interactive neukazuje)
    duration_ms: int = 0
    tool_uses: tuple[str, ...] = ()      # jména použitých Claude tools
    exit_code: int | None = None
    error: str | None = None             # pokud ok=False
    stderr_preview: str | None = None    # pokud subprocess error
    timeout: bool = False
    canceled: bool = False
    # Adapter identification (debugging / telemetry)
    adapter: str = ""                    # "print" | "tmux"

    def to_dict(self) -> dict[str, Any]:
        """Backwards-compat serializer matching starý dict shape z
        `ask_claude_oneshot()`. Voice tool `ask_claude` returnoval dict
        s těmito klíči - shim layer to mapuje 1:1."""
        out: dict[str, Any] = {
            "ok": self.ok,
            "mode": self.mode,
        }
        if self.text:
            out["text"] = self.text
        if self.model:
            out["model"] = self.model
        if self.session_id is not None:
            out["session_id"] = self.session_id
        if self.total_cost_usd is not None:
            out["total_cost_usd"] = self.total_cost_usd
        if self.duration_ms:
            out["duration_ms"] = self.duration_ms
        if self.tool_uses:
            out["tool_uses"] = list(self.tool_uses)
        if self.exit_code is not None:
            out["exit_code"] = self.exit_code
        if self.error is not None:
            out["error"] = self.error
        if self.stderr_preview is not None:
            out["stderr_preview"] = self.stderr_preview
        if self.timeout:
            out["timeout"] = True
        if self.canceled:
            out["canceled"] = True
        if self.adapter:
            out["adapter"] = self.adapter
        return out
