"""Konfigurace agent módu. Načítá se jednou při importu — WORKDIR je tedy
fixní pro celý běh procesu (= jak byla webapp spuštěna)."""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_workdir() -> Path:
    override = os.environ.get("AGENT_WORKDIR")
    raw = Path(override) if override else Path.cwd()
    return raw.resolve()


# Sandbox: pracovní adresář. Read/write/bash uvnitř = AUTO; mimo = ASK
# (kromě explicit allowlistu níže).
WORKDIR: Path = _resolve_workdir()


# Cesty mimo WORKDIR, kde je read = AUTO (čistě veřejné systémové info).
# Žádné dotfiles, žádné keys, žádné /proc/self/environ.
READ_ALLOWLIST_PATHS: tuple[str, ...] = (
    "/etc/os-release",
    "/etc/lsb-release",
    "/etc/hostname",
    "/proc/cpuinfo",
    "/proc/meminfo",
    "/proc/version",
    "/proc/uptime",
)


# Bash root commandy, které mohou jet AUTO (s cwd=WORKDIR + výstup capped).
# Cokoli mimo tento set = ASK. Žádné shell metacharaktery v AUTO příkazu
# (>, >>, |, ;, &&, ||, $(), backticks) — ty taky vynucují ASK.
BASH_AUTO_COMMANDS: frozenset[str] = frozenset(
    ["pwd", "ls", "rg", "find", "cat", "git", "wc", "head", "tail"]
)


# Git subkomendy povolené v AUTO (jen read-only).
GIT_AUTO_SUBCOMMANDS: frozenset[str] = frozenset(
    ["status", "diff", "log", "show", "branch", "remote", "config", "blame", "ls-files"]
)


# Find argumenty, které okamžitě eskalují na ASK (mutace FS).
FIND_FORBIDDEN_FLAGS: frozenset[str] = frozenset(["-delete", "-exec", "-execdir", "-ok", "-okdir"])


# Limity
MAX_WALL_TIME_SEC: int = int(os.environ.get("AGENT_MAX_WALL_TIME_SEC", "600"))
TOOL_OUTPUT_CAP_BYTES: int = int(os.environ.get("AGENT_OUTPUT_CAP_BYTES", str(4 * 1024 * 1024)))
READ_SIZE_CAP_BYTES: int = 256 * 1024
BASH_TIMEOUT_SEC: int = 30
BASH_OUTPUT_CAP_BYTES: int = 1024 * 1024
# Limit kolikrát smí model v jednom turnu chtít další round (tool round-trip).
# Zabrání runaway smyčce / token spamu i bez vyhynutí wall-time.
MAX_MODEL_ROUNDS_PER_TURN: int = int(os.environ.get("AGENT_MAX_ROUNDS", "16"))
# Cap součtu tool callů přes všechny roundy.
MAX_TOOL_CALLS_PER_TURN: int = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "32"))


# Claude bridge (fáze 6)
CLAUDE_DEFAULT_MODEL: str = os.environ.get("AGENT_CLAUDE_MODEL", "claude-opus-4-7")
CLAUDE_UNRESTRICTED: bool = os.environ.get("AGENT_CLAUDE_UNRESTRICTED", "0") == "1"


# Brave Search (fáze 4)
BRAVE_SEARCH_API_KEY: str = os.environ.get("BRAVE_SEARCH_API_KEY", "")


# Voice approval — destruktivní akce vyžadují explicit „ano povoluju".
DESTRUCTIVE_APPROVAL_PHRASE: str = "ano povoluju"
APPROVE_PHRASES: tuple[str, ...] = ("ano", "jo", "ok", "okej", "okay", "povol", "povoluju", "jasně", "jasne", "fajn", "yes")
DENY_PHRASES: tuple[str, ...] = ("ne", "stop", "zruš", "zrus", "nepovoluju", "nechci", "no")
