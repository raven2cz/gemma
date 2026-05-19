"""Stream parsing helpers shared between adapters.

- stream_json: NDJSON line parser pro `claude -p --output-format stream-json`
- ansi: ANSI escape stripping (used by tmux adapter)
- tui_state: pyte-based TUI state machine (used by tmux adapter)
"""
