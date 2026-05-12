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
# Pozn.: `git` odstraněn z AUTO — git operace (status/diff/log/show/blame)
# čtou .git/config a .gitattributes, které jsou uvnitř workdir = write AUTO.
# LLM by mohl napsat malicious .git/config s diff.external/core.fsmonitor/
# textconv → RCE při kterémkoli git invocation. Git proto vyžaduje approval.
BASH_AUTO_COMMANDS: frozenset[str] = frozenset(
    ["pwd", "ls", "rg", "find", "cat", "wc", "head", "tail"]
)


# Git subkomendy povolené v AUTO (jen read-only).
# Pozn.: AKTUÁLNĚ NEPOUŽÍVÁNO — `git` je úplně vyhozen z BASH_AUTO_COMMANDS
# kvůli RCE vektoru přes .git/config (workdir-controlled). Konstanta zůstává
# pro budoucí hardened-AUTO mode (git -c overrides + .git/ write block).
GIT_AUTO_SUBCOMMANDS: frozenset[str] = frozenset(
    ["status", "diff", "log", "show", "blame", "ls-files"]
)


# Find mutating flagy → DESTRUCTIVE (vyžaduje "ano povoluju").
# `-delete`/`-exec` přímo modifikují FS; `-fprint*`/`-fls` zapisují do souboru.
FIND_MUTATING_FLAGS: frozenset[str] = frozenset([
    "-delete", "-exec", "-execdir", "-ok", "-okdir",
    "-fprint", "-fprint0", "-fprintf", "-fls",
])

# Symlink-follow + read-escape flagy pro find → eskalace na ASK medium.
# `-L/-H/-follow` aktivují symlink follow; `-files0-from FILE` čte path list
# ze souboru a tudíž obejde positional path check.
FIND_SYMLINK_FOLLOW_FLAGS: frozenset[str] = frozenset([
    "-L", "-H", "-follow", "-files0-from",
])

# Plný AUTO-deny set pro find = mutating + symlink-follow.
FIND_FORBIDDEN_FLAGS: frozenset[str] = FIND_MUTATING_FLAGS | FIND_SYMLINK_FOLLOW_FLAGS


# Ripgrep argumenty, které okamžitě eskalují na ASK (RCE přes preprocessing /
# decompression hooks). `--pre <cmd>` spustí libovolný binární soubor.
# `-z` / `--search-zip` může spustit dekompresor.
# `-L` / `--follow` aktivuje symlink follow → AUTO traversal mimo workdir.
# `-f` / `--file` čte pattern ze souboru → read escape přes pattern-file path.
# `--ignore-file` taky čte soubor (gitignore-style) → read escape stejný vektor.
RG_FORBIDDEN_FLAGS: frozenset[str] = frozenset([
    "--pre", "--pre-glob", "-z", "--search-zip", "--hostname-bin",
    "-L", "--follow", "-f", "--file", "--ignore-file",
])


# `ls` flagy, které eskalují na ASK — `-L`/`--dereference` derefuje symlinky;
# v kombinaci s `-R` rekurzivně listne přes link ven z workdir.
LS_FORBIDDEN_FLAGS: frozenset[str] = frozenset([
    "-L", "--dereference", "-H", "--dereference-command-line",
    "--dereference-command-line-symlink-to-dir",
])


# `wc` flagy, které čtou paths ze souboru (file-list-driven read escape).
# `--files0-from=FILE` — wc čte NULL-separated paths z FILE a zpracovává je.
# Pokud FILE obsahuje cestu mimo workdir, wc to přečte bez AUTO path checku.
WC_FORBIDDEN_FLAGS: frozenset[str] = frozenset([
    "--files0-from",
])


# Git argumenty (na úrovni argv, ne subcommand), které eskalují na ASK.
# `--output` (git diff/log file write), `-c key=value` (per-call config override
# = stejný vektor jako `git config`), `--global` / `--file` (config setters).
GIT_FORBIDDEN_FLAGS: frozenset[str] = frozenset([
    "--output", "-o", "--global", "--system", "--local", "--file", "-f", "-c",
])


# Wrapper commandy, které musíme přeskočit při detekci effective root tokenu.
# Příklad: `env rm -rf /` → root=`env`, ale efektivní root je `rm` (destruktivní).
# Pozn.: shell interprety (bash/sh/python) tady NEJSOU — jejich `-c` arg
# nelze staticky parsovat, takže je necháváme jako vlastní root (= ASK path).
BASH_WRAPPER_COMMANDS: frozenset[str] = frozenset([
    "env", "xargs", "nohup", "time", "command", "builtin", "exec",
])


# Bash root tokeny, které jsou destruktivní → vyžadují explicit "ano povoluju".
# Detekováno jako první token kteréhokoli segmentu (split na ; | &).
# Pokrývá: file mutation, privilege escalation, system control, network/firewall.
BASH_DESTRUCTIVE_COMMANDS: frozenset[str] = frozenset([
    "rm", "rmdir", "shred", "srm", "unlink",
    "dd", "mkfs", "mkfs.ext4", "mkfs.xfs", "fdisk", "parted", "wipefs",
    "sudo", "su", "doas", "pkexec",
    "systemctl", "service", "init", "telinit",
    "reboot", "shutdown", "halt", "poweroff",
    "mount", "umount", "swapon", "swapoff",
    "chmod", "chown", "chgrp", "setcap", "setfacl",
    "kill", "pkill", "killall",
    "iptables", "ip6tables", "nft", "ufw", "firewall-cmd",
    "userdel", "useradd", "usermod", "groupdel", "groupadd", "passwd",
    "crontab", "at",
    "modprobe", "rmmod", "insmod",
])


# Env vars zachované při exekuci bash (vše ostatní scrubováno — credentials, API keys).
# LC_* prefix je akceptován navíc (locale env vars).
BASH_ENV_ALLOWLIST: frozenset[str] = frozenset([
    "HOME", "USER", "LOGNAME", "SHELL",
    "LANG", "TZ", "TERM",
    "PWD", "OLDPWD", "SHLVL", "_",
])


# Fixní PATH pro bash subprocess (defense proti PATH injection do ~/.local/bin).
BASH_PATH: str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


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
