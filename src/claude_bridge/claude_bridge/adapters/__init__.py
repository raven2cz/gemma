"""Concrete adapter implementations.

PrintModeAdapter: spustí `claude -p` per ask() volání.
TmuxAdapter:      drive `claude` v tmux pseudo-TTY (experimental).
"""
from __future__ import annotations

from .print_mode import PrintModeAdapter

__all__ = ["PrintModeAdapter"]
