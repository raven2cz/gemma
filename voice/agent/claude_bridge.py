"""Compat shim - delegates do `claude_bridge` library v src/claude_bridge/.

Tato vrstva existuje kvůli backwards compatibility:
- Voice agent kód (voice/agent/tools/claude.py) volá `ask_claude_oneshot(...)`
  od fáze 6 (Claude bridge tool); migrace na adapter pattern je v
  fáze tmux adapter (feature/claude-tmux-adapter branch).
- Testy `tests/test_agent_claude*.py` importují privátní helpery
  (`_parse_stream_event`, `_build_subprocess_env`, `_read_stream`, atd.)
  - tyto re-exporty drží zelené dokud tests nepřesunem do
  `src/claude_bridge/tests/`.

Plán roadmap (feature/claude-tmux-adapter):
- Fáze 0 (=teď): tato shim vrstva, adapter pattern v claude_bridge lib
- Fáze 5: voice/agent/tools/claude.py přejde na `create_adapter(config).ask(...)`
- Fáze 7: tato shim file deleted, kompletní migrace

Nová impl je v `src/claude_bridge/claude_bridge/adapters/print_mode.py`.
"""
from __future__ import annotations

# Re-export public API. ask_claude_oneshot je primární vstupní bod.
from claude_bridge.adapters.print_mode import (
    ask_claude_oneshot,
    # Privátní helpery exportované pro tests:
    _build_argv,
    _build_subprocess_env,
    _kill_process_group,
    _read_stream,
    _drain_stderr,
    _safe_progress,
)
from claude_bridge.parsing.stream_json import (
    parse_stream_event as _parse_stream_event,
    short_tool_input_summary as _short_tool_input_summary,
)

__all__ = [
    "ask_claude_oneshot",
    "_parse_stream_event",
    "_short_tool_input_summary",
    "_build_argv",
    "_build_subprocess_env",
    "_kill_process_group",
    "_read_stream",
    "_drain_stderr",
    "_safe_progress",
]
