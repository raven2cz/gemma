"""pyte-based TUI state machine pro Claude interactive UI.

Claude Code CLI v interactive režimu používá Ink (React for terminals).
Output je full terminal UI: header, scrollback, current input box (s `❯`
prompt indicator), spinner při thinking. Žádný stream-json - jen TUI rendering.

Glyph reference (z claude-code source):
- Prompt indicator: `❯` (U+276F, RIGHT-POINTING ANGLE BRACKET) z `figures.pointer`
  npm package. Windows fallback: `>`.
- Tool_use marker (BLACK_CIRCLE): `⏺` (macOS) nebo `●` (Linux/Windows)
- Spinner frames: `· ✢ * ✶ ✻ ✽` (Linux), `· ✢ ✳ ✶ ✻ ✽` (macOS)
- Reduced motion fallback: `●` (CAUTION: stejný jako BLACK_CIRCLE na Linux)

Usage:
    state = TuiState(cols=200, rows=50)
    state.feed(raw_bytes)              # každý tmux capture iter
    if state.is_ready():               # Claude čeká na input?
        ...
    if state.is_thinking():            # spinner viditelný?
        ...
    new_tool_uses = state.poll_tool_uses()  # nově detected tool_uses
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pyte

from .ansi import contains_spinner_frame


# Tool_use marker (figures.pointer mapping platform).
# Pattern 1 (loading): `● ToolName(args)` nebo `⏺ ToolName(args)` na začátku řádku.
# Pattern 2 (collapsed/done): `  ToolName N <stuff>` BEZ marker. Used pro collapsed
# Read/Write tool results: `  Read 1 file (ctrl+o to expand)`.
# Pattern 1 použito v assistant message rendering Cesty load_indicator. Pattern 2
# je collapsed result text bez load animation.
_TOOL_USE_RE = re.compile(r"[⏺●]\s+([A-Z][A-Za-z0-9_]+)(?:\(([^)]*)\))?")

# Known Claude builtin tool names - pro pattern 2 (collapsed) matching.
# Striktní allowlist aby nematchli plain text "Read this code first" apod.
_KNOWN_BUILTIN_TOOLS = frozenset({
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "NotebookEdit", "Task", "TodoWrite",
    "WebFetch", "WebSearch", "BashOutput", "KillShell",
})
_TOOL_USE_COLLAPSED_RE = re.compile(
    r"^\s+(" + "|".join(_KNOWN_BUILTIN_TOOLS) + r")\s+(.+?)(?:\s+\(ctrl\+o)?$"
)

# Prompt indicator (figures.pointer = U+276F na Unix, > na Windows).
# Match jako standalone marker - obvyklý layout: "❯ " na začátku vstupního
# řádku (může být v Box s borderem, takže před ním border char).
_PROMPT_INDICATOR_CHARS = frozenset("❯>")


@dataclass
class ToolUseObserved:
    """Tool_use detected v TUI output. Best-effort, ne tak rich jako
    stream-json (input args jen heuristic z parens text)."""
    tool_name: str
    args_preview: str = ""
    line_index: int = -1


@dataclass
class TuiState:
    """pyte screen + Claude-specific stav detection.

    `feed()` per tmux capture iter pošle nový raw output. Stav se akumuluje
    napříč voláními.
    """
    cols: int = 200
    rows: int = 50

    _screen: pyte.Screen = field(init=False)
    _stream: pyte.Stream = field(init=False)
    _seen_tool_uses: set[tuple[str, str]] = field(default_factory=set)
    _last_screen_hash: int = 0
    _idle_count: int = 0
    # Spinner frame tracking - musíme vidět ALESPOŇ 2 různé spinner znaky
    # po sobě v krátké době abychom byli jistí že je to spinner (single
    # frame char by mohl být normal punctuation `·` v textu).
    _recent_cursor_chars: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._screen = pyte.Screen(self.cols, self.rows)
        self._stream = pyte.Stream(self._screen)

    def feed(self, data: bytes | str) -> None:
        """Push nový raw output do pyte stream."""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = data
        self._stream.feed(text)

    def reset(self) -> None:
        """Wipe pyte state - např. po `/clear` nebo restart session."""
        self._screen.reset()
        self._seen_tool_uses.clear()
        self._last_screen_hash = 0
        self._idle_count = 0
        self._recent_cursor_chars.clear()

    @property
    def screen_lines(self) -> list[str]:
        """Aktuální screen content jako list řádků (clean text, žádný ANSI)."""
        return list(self._screen.display)

    @property
    def cursor_pos(self) -> tuple[int, int]:
        """(x, y) cursor pozice."""
        return (self._screen.cursor.x, self._screen.cursor.y)

    def _get_cursor_char(self) -> str:
        """Vrátí znak na current cursor pozici (= místo kde spinner anim
        kresluje frame). Empty string pokud cursor mimo grid."""
        try:
            row = self._screen.buffer[self._screen.cursor.y]
            char_obj = row[self._screen.cursor.x]
            return char_obj.data
        except (IndexError, KeyError, AttributeError):
            return ""

    def is_thinking(self) -> bool:
        """True pokud Claude TUI spinner běží na current cursor pozici.

        Strategy: cursor char je v spinner frames alphabet (·✢*✶✻✽✳).
        Backup: hledej spinner symbol kdekoli na last 3 řádcích.

        POZOR: jen jedna detekce snapshot může být false-positive (znak `·`
        v normal textu). Caller by měl invokovat opakovaně - poll_thinking_state()
        akumuluje recent cursor chars a vrátí True jen pokud viděli 2+ různé
        spinner chars (= aktivní rotace, ne static text).
        """
        cursor_char = self._get_cursor_char()
        if cursor_char and cursor_char in "·✢*✶✻✽✳":
            self._recent_cursor_chars.append(cursor_char)
            # Trim na last 5
            self._recent_cursor_chars = self._recent_cursor_chars[-5:]
            # Considered spinning pokud aspoň 2 different chars v recent buffer
            if len(set(self._recent_cursor_chars)) >= 2:
                return True
        else:
            # Reset spinner tracking pokud cursor nesvítí na frame
            if cursor_char and not contains_spinner_frame(cursor_char):
                self._recent_cursor_chars.clear()
        return False

    def is_ready(self) -> bool:
        """True pokud Claude čeká na user input.

        Detekce: hledáme `❯` (figures.pointer) v CELÉM screen content.
        Claude TUI vyplní typicky řádky 1-18 (header + scrollback + input
        box + footer), zbytek je prázdný whitespace. `❯` je na řádku
        s input boxem (uprostřed screenu).

        Pozor: `>` jako fallback pro Windows existuje, ale je AMBIGUOUS
        (může být v textu, kód, Git logs, atd.). Proto pro `>` vyžadujeme
        striktnější check: `>` na začátku řádku po stripped whitespace.
        """
        for line in self._screen.display:
            if "❯" in line:
                # ❯ je platform-specific unicode pointer = unambiguous Claude marker
                return True
            # Windows/dumb terminal fallback: `> ` jako jediný content řádku
            # (pyte fillne řádek trailing spaces, takže nejdřív obě strany strip)
            stripped = line.strip()
            if stripped == ">" or stripped.startswith("> "):
                # `> ` na začátku po obousměrném strip + krátký content = input prompt
                if len(stripped) < 5:
                    return True
        return False

    def poll_tool_uses(self) -> list[ToolUseObserved]:
        """Vrátí nově detected tool_uses od posledního volání (deduplicated).

        Detekce přes dva regex patterns:
        1. `[⏺●] ToolName(args)` - loading state (animated ToolUseLoader)
        2. Collapsed: `^\\s+ToolName N <details>` - po completion bez marker
        """
        new_uses: list[ToolUseObserved] = []
        for idx, line in enumerate(self._screen.display):
            # Pattern 1: explicit marker
            for m in _TOOL_USE_RE.finditer(line):
                name = m.group(1)
                args = (m.group(2) or "").strip()
                key = (name, args)
                if key in self._seen_tool_uses:
                    continue
                self._seen_tool_uses.add(key)
                new_uses.append(ToolUseObserved(
                    tool_name=name,
                    args_preview=args[:120],
                    line_index=idx,
                ))
            # Pattern 2: collapsed (only checked pokud line nemá marker)
            if "●" not in line and "⏺" not in line:
                m2 = _TOOL_USE_COLLAPSED_RE.match(line)
                if m2 is not None:
                    name = m2.group(1)
                    details = m2.group(2).strip()
                    key = (name, details)
                    if key in self._seen_tool_uses:
                        continue
                    self._seen_tool_uses.add(key)
                    new_uses.append(ToolUseObserved(
                        tool_name=name,
                        args_preview=details[:120],
                        line_index=idx,
                    ))
        return new_uses

    def check_idle(self) -> int:
        """Increment idle counter pokud screen unchanged od posledního checku."""
        h = hash((tuple(self._screen.display), self._screen.cursor.x, self._screen.cursor.y))
        if h == self._last_screen_hash:
            self._idle_count += 1
        else:
            self._idle_count = 0
            self._last_screen_hash = h
        return self._idle_count
