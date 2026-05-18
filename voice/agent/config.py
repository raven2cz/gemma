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
# `ask_claude` tool deleguje na **Claude Code CLI subprocess** (`claude -p`),
# ne přímo na REST API. Vzor: avatar-engine/avatar_engine/bridges/claude.py.
# Implementace bridge je v voice/agent/claude_bridge.py.
# Klíč: env `ANTHROPIC_API_KEY` → fallback soubor `~/.anthropic-api-key` (0600).
# CLI passuje klíč svým HTTPS klientem.
CLAUDE_CLI_BIN: str = os.environ.get("CLAUDE_CLI_BIN", "claude")
CLAUDE_DEFAULT_MODEL: str = os.environ.get("AGENT_CLAUDE_MODEL", "claude-opus-4-7")
# 600s = 10 min. Complex tasks (design dokument, multi-file refaktor) lehce
# vyžadují 2-5 min reálného běhu Claude CLI (čtení souborů, thinking, generování).
# Dřívější 60s default odpaloval žádost o design doc dřív než stihla doběhnout.
CLAUDE_TIMEOUT_SEC: float = float(os.environ.get("AGENT_CLAUDE_TIMEOUT_SEC", "600"))
CLAUDE_MAX_TOKENS_DEFAULT: int = int(os.environ.get("AGENT_CLAUDE_MAX_TOKENS", "1024"))
CLAUDE_MAX_TOKENS_LIMIT: int = int(os.environ.get("AGENT_CLAUDE_MAX_TOKENS_LIMIT", "4096"))
CLAUDE_MAX_PROMPT_BYTES: int = int(os.environ.get("AGENT_CLAUDE_MAX_PROMPT_BYTES", str(64 * 1024)))
CLAUDE_MAX_SYSTEM_BYTES: int = int(os.environ.get("AGENT_CLAUDE_MAX_SYSTEM_BYTES", str(16 * 1024)))
# Deprecated: dříve hard cap na total stream bytes, nyní bridge žádný cap
# nepoužívá (user policy: pro Opus implementace by hard kill při X MB rušil
# reálnou práci - soubory už jsou zapsané, ale my bychom vrátili "fail").
# Bezpečnostní mechanismy: timeout (CLAUDE_TIMEOUT_SEC) + cancel_event +
# asyncio per-line StreamReader limit. Konstanta zůstává pro backward compat
# a downstream tool_output cap (v loop.py, 4 MB) ji nevyužívá - má vlastní.
CLAUDE_OUTPUT_CAP_BYTES: int = 0  # deprecated, ignored by bridge

ANTHROPIC_API_KEY_FILE: str = os.environ.get(
    "ANTHROPIC_API_KEY_FILE", str(Path.home() / ".anthropic-api-key"),
)


def _load_anthropic_key() -> str:
    """env first, pak soubor s 0600 ověřením. Tichý fallback na ''."""
    env = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env:
        return env
    p = Path(ANTHROPIC_API_KEY_FILE)
    try:
        st = p.stat()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return ""
    # World/group readable → odmítnout (secret nesmí být group/other readable).
    if st.st_mode & 0o077:
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


ANTHROPIC_API_KEY: str = _load_anthropic_key()


# Brave Search (fáze 4)
# Auto-load: env var → soubor `~/.brave-search-api` (path z env override).
# Soubor MUSÍ být 0600 (jen owner read/write), jinak fallback selže.
BRAVE_SEARCH_API_KEY_FILE: str = os.environ.get(
    "BRAVE_SEARCH_API_KEY_FILE", str(Path.home() / ".brave-search-api"),
)


def _load_brave_key() -> str:
    """env first, pak soubor s 0600 ověřením. Tichý fallback na ''."""
    env = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if env:
        return env
    p = Path(BRAVE_SEARCH_API_KEY_FILE)
    try:
        st = p.stat()
    except (FileNotFoundError, PermissionError, OSError):
        return ""
    # Pouze regular file; symlink follow akceptován (uživatel může mít vault).
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return ""
    # World/group readable → odmítnout (secret nesmí být group/other readable).
    if st.st_mode & 0o077:
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


BRAVE_SEARCH_API_KEY: str = _load_brave_key()


# Web tooly (fáze 4)
# fetch_url: HTTP GET sandbox.
FETCH_URL_TIMEOUT_SEC: float = float(os.environ.get("AGENT_FETCH_TIMEOUT_SEC", "15"))
FETCH_URL_MAX_BYTES: int = int(os.environ.get("AGENT_FETCH_MAX_BYTES", str(2 * 1024 * 1024)))
FETCH_URL_MAX_REDIRECTS: int = int(os.environ.get("AGENT_FETCH_MAX_REDIRECTS", "5"))
# Allowed URL schemes. file://, ftp://, gopher://, dict://, ldap://, jar://, etc.
# jsou explicit blocked (DENY). Pouze http(s).
FETCH_URL_ALLOWED_SCHEMES: frozenset[str] = frozenset(["http", "https"])
# Content-types, které vrátíme jako text. Cokoli mimo = binary, vrátíme jen
# metadata (status + content_type + size), žádný body.
FETCH_URL_TEXT_CONTENT_TYPES: tuple[str, ...] = (
    "text/", "application/json", "application/xml", "application/xhtml+xml",
    "application/javascript", "application/x-yaml", "application/yaml",
    "application/ld+json", "application/rss+xml", "application/atom+xml",
)
# User-Agent string (žádný real-browser fingerprint — explicit agent identity).
FETCH_URL_USER_AGENT: str = "gemma-agent/0.1 (+local)"


# web_search (Brave Search API)
WEB_SEARCH_DEFAULT_COUNT: int = 5
WEB_SEARCH_MAX_COUNT: int = 20
WEB_SEARCH_TIMEOUT_SEC: float = float(os.environ.get("AGENT_SEARCH_TIMEOUT_SEC", "10"))
BRAVE_SEARCH_ENDPOINT: str = "https://api.search.brave.com/res/v1/web/search"


# Philips Hue / OpenHue (fáze 5)
# `light_list` + `light_set` volají přímo Hue Bridge API v2 přes HTTPS.
# Bridge IP + application key jsou v `~/.openhue/config.yaml` (vytvořeno
# příkazem `openhue setup`). Loader níže to čte a vrátí ("", "") pokud chybí.
HUE_CONFIG_FILE: str = os.environ.get(
    "HUE_CONFIG_FILE", str(Path.home() / ".openhue" / "config.yaml"),
)
HUE_TIMEOUT_SEC: float = float(os.environ.get("AGENT_HUE_TIMEOUT_SEC", "8"))
HUE_OUTPUT_CAP_BYTES: int = int(os.environ.get("AGENT_HUE_OUTPUT_CAP_BYTES", str(512 * 1024)))


def _load_hue_config() -> tuple[str, str]:
    """Načte (bridge_ip, app_key) z ~/.openhue/config.yaml.
    Vrátí ("", "") pokud konfigurace chybí nebo má špatná oprávnění.

    Formát:
        bridge: 192.168.x.x
        key: <40-char application key>

    Bridge IP musí být literal IPv4/IPv6 a privátní (defense — `_validate_url`
    by jinak vyžadoval public IP; tady cíleně cílíme local network).
    """
    p = Path(HUE_CONFIG_FILE)
    try:
        st = p.stat()
    except (FileNotFoundError, PermissionError, OSError):
        return "", ""
    import stat as _stat
    if not _stat.S_ISREG(st.st_mode):
        return "", ""
    # Nedáváme nutně 0600, protože openhue CLI ho píše s default umask.
    # Ale jakákoli world-write je no-go (cizí proces by mohl změnit klíč).
    if st.st_mode & 0o002:
        return "", ""
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "", ""
    bridge = ""
    key = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("bridge:"):
            bridge = line.split(":", 1)[1].strip()
        elif line.startswith("key:"):
            key = line.split(":", 1)[1].strip()
    return bridge, key


HUE_BRIDGE_IP, HUE_APP_KEY = _load_hue_config()


# Audit log (fáze 8)
# Per-tool-call append-only JSONL log s permission decision + approval outcome
# + result. Soubor `~/.gemma/agent-audit/YYYY-MM-DD.jsonl`, 0o600. Disabled
# když `AGENT_AUDIT_DIR=off` (case-insensitive) nebo prázdná hodnota.
def _resolve_audit_dir() -> Path | None:
    raw = os.environ.get("AGENT_AUDIT_DIR")
    if raw is not None:
        if raw.strip().lower() in ("", "off", "0", "false", "no"):
            return None
        return Path(raw).expanduser()
    return Path.home() / ".gemma" / "agent-audit"


AUDIT_DIR: Path | None = _resolve_audit_dir()
# Per-string truncation cap pro logged args (chars, ne bytes — JSON je text).
AUDIT_FIELD_CAP: int = int(os.environ.get("AGENT_AUDIT_FIELD_CAP", "500"))
# Preview cap pro tool-specific redaction (prompt/system).
AUDIT_ARG_PREVIEW: int = int(os.environ.get("AGENT_AUDIT_ARG_PREVIEW", "200"))


# Voice approval — destruktivní akce vyžadují explicit „ano povoluju".
DESTRUCTIVE_APPROVAL_PHRASE: str = "ano povoluju"

# Auto-degradace po destruktivní akci (Fáze 9). Po každém ÚSPĚŠNÉM schválení
# destruktivní operace se po dobu N sekund přepne AUTO → ASK pro stejný workdir,
# i pro jinak allowlistové příkazy. Anti-momentum: jedna fráze „ano povoluju"
# nesmí otevřít neomezený coast pro další destruktivní akce.
AUTO_DEGRADE_AFTER_DESTRUCTIVE_SEC: int = int(
    os.environ.get("AGENT_AUTO_DEGRADE_SEC", "300")
)


# Audio filler delay (Fáze 9.3). Pokud běží tool déle než tato hodnota,
# loop emit `audio_filler` event a frontend přehraje krátké „moment, hledám…"
# aby user věděl, že systém pracuje. 0 = vypnuto.
AUDIO_FILLER_DELAY_SEC: float = float(
    os.environ.get("AGENT_AUDIO_FILLER_DELAY_SEC", "2.0")
)


# Dangerous mode (opt-in). Pokud zapnuto, všechny ASK rozhodnutí se přepnou
# na AUTO — kromě `requires_explicit=True` (destruktivní akce: rm/sudo/chmod/...).
# DENY (special files, syntax err) zůstává. Aktivuje se přes `--dangerous`
# flag launcheru nebo `AGENT_DANGEROUS=1`. UI ukáže červený badge.
DANGEROUS_MODE: bool = os.environ.get("AGENT_DANGEROUS", "0").strip() == "1"


APPROVE_PHRASES: tuple[str, ...] = ("ano", "jo", "ok", "okej", "okay", "povol", "povoluju", "jasně", "jasne", "fajn", "yes")
DENY_PHRASES: tuple[str, ...] = ("ne", "stop", "zruš", "zrus", "nepovoluju", "nechci", "no")
