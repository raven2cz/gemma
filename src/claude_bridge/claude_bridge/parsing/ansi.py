"""ANSI escape sequence utilities + Claude TUI glyph constants.

Glyph konstanty z claude-code/src/constants/figures.ts a Spinner/utils.ts:
- BLACK_CIRCLE platform-dependent: ⏺ na macOS, ● na Linux/Windows
- Spinner frames also platform-dependent (NE Braille, jak jsem dříve myslel)

Pyte simuluje terminal (renderuje pixely), tato fce je pro post-process
extraction text + spinner/icon detection.
"""
from __future__ import annotations

import re

# CSI/OSC/DCS escape sequences. Covers ANSI/VT100/xterm-256.
_ANSI_RE = re.compile(
    r"""
    \x1B  # ESC
    (?:
        \[ [?>!]? [0-9;]* [a-zA-Z]   # CSI: ESC[ <params> <letter>
      | \] [0-9]+ ;? [^\x07\x1B]* (?: \x07 | \x1B\\ )?  # OSC: ESC] ... BEL/ST
      | [PX^_] [^\x07\x1B]* (?: \x07 | \x1B\\ )?        # DCS/PM/APC/SOS
      | [@-Z\\-_=> ]               # 2-char ESC (e.g. ESC=, ESC>, ESC c)
      | %@                         # designate G0 charset
    )
    """,
    re.VERBOSE,
)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text."""
    return _ANSI_RE.sub("", text)


# claude-code/src/constants/figures.ts:
#   BLACK_CIRCLE = process.platform === 'darwin' ? '⏺' : '●'
# Tool_use messages, queued items, attachment markers all use BLACK_CIRCLE.
_BLACK_CIRCLE_CHARS = frozenset(("⏺", "●"))

# claude-code/src/components/Spinner/utils.ts:
#   darwin:  ['·', '✢', '✳', '✶', '✻', '✽']
#   linux:   ['·', '✢', '*', '✶', '✻', '✽']
#   ghostty: ['·', '✢', '✳', '✶', '✻', '*']
# Sjednocená frame alphabet pro platform-independent detection.
_SPINNER_FRAMES = frozenset("·✢✳✶✻✽*")

# Reduced-motion fallback (Spinner/SpinnerGlyph.tsx). Pozor: ● je STEJNÝ
# znak jako BLACK_CIRCLE na Linux - takže reduced-motion spinner a tool_use
# marker jsou nerozlišitelné z čistého textu. Caller MUSÍ použít kontext
# (např. cursor position, řádek index, surrounding text).
_REDUCED_MOTION_DOT = "●"


def is_black_circle_char(c: str) -> bool:
    """True pokud znak je Claude tool_use marker (BLACK_CIRCLE)."""
    return c in _BLACK_CIRCLE_CHARS


def contains_black_circle(text: str) -> bool:
    """True pokud text obsahuje tool_use marker."""
    return any(c in _BLACK_CIRCLE_CHARS for c in text)


def contains_spinner_frame(text: str) -> bool:
    """True pokud text obsahuje znak z Claude spinner frames alphabet.

    POZOR: `·` (middle dot) je legitimate normal punctuation v output -
    nepoužívej tento check izolovaně. Combine s kontextem (např. malá
    box na cursor řádku co se animuje, ne plain text bullet)."""
    return any(c in _SPINNER_FRAMES for c in text)


def is_reduced_motion_dot(c: str) -> bool:
    """True pokud znak je reduced-motion spinner dot (= ● = BLACK_CIRCLE
    na Linux). Caller musí použít kontext k rozlišení."""
    return c == _REDUCED_MOTION_DOT
