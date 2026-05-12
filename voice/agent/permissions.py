"""Permission resolver pro tool calls.

`decide(tool_name, args, workdir)` vrátí `PermissionResult` s rozhodnutím
AUTO / ASK / DENY. Třída rozhodnutí závisí na nástroji a argumentech.

Default pro neznámý tool = DENY (bezpečnost > UX). Tooly registrují vlastní
classifier dekorátorem `@register_classifier(name)`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable

from voice.agent import config


class Decision(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


Risk = str  # "low" | "medium" | "high" | "destructive"


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str
    summary: str           # user-facing CZ string ("echo: hello")
    risk: Risk = "low"
    requires_explicit: bool = False   # destruktivní → potřeba „ano povoluju"
    # Canonical resolved Path, na které classifier rozhodl (jen pro FS tooly).
    # Loop ji předá do `ExecuteContext.resolved_path`, aby tool nemusel re-resolvovat
    # a aby se eliminoval TOCTOU gap mezi check-time a exec-time.
    resolved_path: Path | None = None


Classifier = Callable[[dict, Path], PermissionResult]

_CLASSIFIERS: dict[str, Classifier] = {}


def register_classifier(name: str) -> Callable[[Classifier], Classifier]:
    def wrap(fn: Classifier) -> Classifier:
        _CLASSIFIERS[name] = fn
        return fn
    return wrap


# ----------------------------------------------------------------------
# Auto-degradace po destruktivní akci (Fáze 9).
# Po úspěšném schválení destruktivního tool callu (requires_explicit=True
# + user napsal/řekl „ano povoluju") přepneme AUTO → ASK pro daný workdir
# na AUTO_DEGRADE_AFTER_DESTRUCTIVE_SEC sekund. Brání eskalaci kdy jediný
# souhlas otevře okno pro libovolné další tichí AUTO příkazy.
# ----------------------------------------------------------------------

# Klíč = absolutní cesta workdiru, hodnota = monotonic timestamp poslední
# destruktivní approval. Modul-level state přetrvá přes turn boundaries
# (stejný proces serveru = jedna session). Reset přes `clear_degrade_state()`
# pro testy + případný admin reset endpoint.
_DESTRUCTIVE_APPROVAL_TS: dict[str, float] = {}


def _workdir_key(workdir: Path) -> str:
    try:
        return str(Path(workdir).resolve())
    except (OSError, RuntimeError):
        return str(workdir)


def mark_destructive_approval(workdir: Path) -> None:
    """Zaznamenat ÚSPĚŠNÉ schválení destruktivní operace pro daný workdir.
    Volá se ze server.py po validaci `requires_explicit` fráze.
    """
    _DESTRUCTIVE_APPROVAL_TS[_workdir_key(workdir)] = time.monotonic()


def _degrade_remaining_sec(workdir: Path) -> float:
    """Vrátí kolik sekund zbývá v degradačním okně (0 = mimo okno)."""
    ts = _DESTRUCTIVE_APPROVAL_TS.get(_workdir_key(workdir))
    if ts is None:
        return 0.0
    elapsed = time.monotonic() - ts
    remaining = config.AUTO_DEGRADE_AFTER_DESTRUCTIVE_SEC - elapsed
    return max(0.0, remaining)


def clear_degrade_state() -> None:
    """Vymaže auto-degrade state. Test-only / admin reset."""
    _DESTRUCTIVE_APPROVAL_TS.clear()


def decide(tool_name: str, args: dict, workdir: Path) -> PermissionResult:
    fn = _CLASSIFIERS.get(tool_name)
    if fn is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=f"no classifier registered for {tool_name!r}",
            summary=f"Neznámý nástroj {tool_name!r}",
            risk="high",
        )
    result = fn(args, workdir)
    # Auto-degradace: pokud jsme uvnitř okna po destruktivním approve a
    # classifier vrátil AUTO, downgrade na ASK. ASK/DENY rozhodnutí
    # nemodifikujeme — ta už jsou „bezpečnější nebo stejná".
    if result.decision == Decision.AUTO:
        remaining = _degrade_remaining_sec(workdir)
        if remaining > 0:
            return replace(
                result,
                decision=Decision.ASK,
                reason=(
                    f"auto-degradace {int(remaining)}s po destruktivní akci "
                    f"(původně: {result.reason})"
                ),
                risk="medium" if result.risk == "low" else result.risk,
            )
    return result


# ----------------------------------------------------------------------
# Built-in classifiers (Phase 1 = echo only; další fáze rozšíří).
# ----------------------------------------------------------------------


@register_classifier("echo")
def _echo(args: dict, workdir: Path) -> PermissionResult:
    text = str(args.get("text", ""))[:60]
    return PermissionResult(
        decision=Decision.AUTO,
        reason="echo has no side effects",
        summary=f'echo: "{text}"',
        risk="low",
    )


# ----------------------------------------------------------------------
# Phase 2: File-system classifiery.
# Sandbox primitiva v `tools/_sandbox.py` — tady jen mapujeme decision.
# ----------------------------------------------------------------------


def _short(p: str | Path, n: int = 80) -> str:
    s = str(p)
    return s if len(s) <= n else "…" + s[-(n - 1):]


def _read_style_decision(
    tool_name: str, args: dict, workdir: Path, *, summary_verb: str,
) -> PermissionResult:
    """Společná logika pro read-only tooly (read_file/list_files/glob/grep):
    sandbox resolve → DENY pokud special, AUTO uvnitř / allowlist, jinak ASK.
    """
    from voice.agent.tools._sandbox import resolve_safe, is_read_allowed

    path_str = args.get("path", "") if tool_name != "glob" else args.get("path", ".")
    if tool_name in ("glob", "grep") and not path_str:
        path_str = "."
    resolved, err = resolve_safe(path_str, workdir)
    if resolved is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=err or "invalid path",
            summary=f"{summary_verb} odmítnuto: {err}",
            risk="high",
        )
    allowed, reason = is_read_allowed(resolved, workdir)
    short = _short(resolved)
    if allowed:
        return PermissionResult(
            decision=Decision.AUTO,
            reason=reason,
            summary=f"{summary_verb}: {short}",
            risk="low",
            resolved_path=resolved,
        )
    return PermissionResult(
        decision=Decision.ASK,
        reason=reason,
        summary=f"{summary_verb} mimo workdir: {short}",
        risk="medium",
        resolved_path=resolved,
    )


@register_classifier("read_file")
def _cls_read_file(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("read_file", args, workdir, summary_verb="read")


@register_classifier("list_files")
def _cls_list_files(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("list_files", args, workdir, summary_verb="list")


@register_classifier("glob")
def _cls_glob(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("glob", args, workdir, summary_verb="glob")


@register_classifier("grep")
def _cls_grep(args: dict, workdir: Path) -> PermissionResult:
    return _read_style_decision("grep", args, workdir, summary_verb="grep")


def _write_style_decision(
    args: dict, workdir: Path, *, summary_verb: str,
) -> PermissionResult:
    """Společná logika pro write_file/edit_file: AUTO inside workdir,
    ASK + requires_explicit ("ano povoluju") outside (= destructive).
    """
    from voice.agent.tools._sandbox import resolve_safe, is_inside_workdir

    path_str = args.get("path", "")
    resolved, err = resolve_safe(path_str, workdir)
    if resolved is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=err or "invalid path",
            summary=f"{summary_verb} odmítnuto: {err}",
            risk="high",
        )
    short = _short(resolved)
    if is_inside_workdir(resolved, workdir):
        return PermissionResult(
            decision=Decision.AUTO,
            reason="inside workdir",
            summary=f"{summary_verb}: {short}",
            risk="low",
            resolved_path=resolved,
        )
    # Mimo workdir = destructive, vyžaduje frázi "ano povoluju".
    return PermissionResult(
        decision=Decision.ASK,
        reason="write/edit outside workdir is destructive",
        summary=f"{summary_verb} MIMO workdir: {short}",
        risk="destructive",
        requires_explicit=True,
        resolved_path=resolved,
    )


@register_classifier("write_file")
def _cls_write_file(args: dict, workdir: Path) -> PermissionResult:
    return _write_style_decision(args, workdir, summary_verb="write")


@register_classifier("edit_file")
def _cls_edit_file(args: dict, workdir: Path) -> PermissionResult:
    return _write_style_decision(args, workdir, summary_verb="edit")


# ----------------------------------------------------------------------
# Phase 3: Bash shell classifier.
# ----------------------------------------------------------------------

import os
import re
import shlex

# Shell metaznaky které vynucují ASK + shell=True exekuci.
# Pozn.: ne všechny mají destruktivní efekt sami o sobě, ale jejich přítomnost
# znamená, že nestačí argv parser a musíme jet přes /bin/bash -c.
_SHELL_META_RE = re.compile(r"[|<>;&$`()\n\\]|>>|<<")

# Segment separator pro destructive token detection (split podle ; | &).
_SEGMENT_SEP_RE = re.compile(r"[;|&]+")

# File redirect detection — pokrývá:
#   `> file`, `>>file`, `> /dev/null`    (basic stdout redirect)
#   `&> file`, `&>> file`                 (composite stdout+stderr → file)
#   `>& /tmp/x`                           (rare bashism, stderr+stdout → file)
# Nematch na `>&2`, `2>&1`, `&>&2`        (fd-duplikace, ne file write).
# `(?!\s*&\d)` lookahead: po operátoru nesmí následovat ws + & + digit
# (= fd-dup pattern jako `>&2`, `&>&1`).
_FILE_REDIRECT_RE = re.compile(r"(?:&>>?|>>?)(?!\s*&\d)\s*\S")


def _has_shell_metas(command: str) -> bool:
    """True pokud command obsahuje shell metaznaky vyžadující /bin/bash -c."""
    return bool(_SHELL_META_RE.search(command))


def _has_file_redirect(command: str) -> bool:
    """True pokud command obsahuje file-write redirect (> file, >> file).

    Nereaguje na fd-duplikaci jako `2>&1` ani `>&2`. Pozn.: může false-positive
    pokud `>` je uvnitř quoted stringu (regex nečte quote state), což je OK
    — escaluje na requires_explicit, user jen potvrdí.
    """
    return bool(_FILE_REDIRECT_RE.search(command))


def _effective_root(tokens: list[str]) -> str:
    """Vrátí basename efektivního root commandu po skipu wrapperů.

    Wrapper = command, který předává argv následujícímu commandu (env, xargs,
    nohup, time, command, builtin, exec). Příklad: `/usr/bin/env rm -rf /` →
    skip `env` → effective root = `rm`.

    Pro `env` taky skipne `VAR=value` tokeny (např. `env LC_ALL=C rm foo`).
    """
    from voice.agent.config import BASH_WRAPPER_COMMANDS

    i = 0
    while i < len(tokens):
        # Basename pro absolutní cesty: `/bin/rm` → `rm`, `./rm` → `rm`.
        bn = os.path.basename(tokens[i]) or tokens[i]
        if bn not in BASH_WRAPPER_COMMANDS:
            return bn
        i += 1
        # `env` syntax: env [-i] [VAR=value …] COMMAND [args…]
        if bn == "env":
            while i < len(tokens):
                t = tokens[i]
                if "=" in t and not t.startswith("-"):
                    i += 1
                    continue
                if t in ("-i", "--ignore-environment", "-0", "--null"):
                    i += 1
                    continue
                if t.startswith("-u") or t == "--unset":
                    i += 1
                    # -u VAR or --unset VAR má další token argument
                    if i < len(tokens) and not tokens[i].startswith("-"):
                        i += 1
                    continue
                break
    return ""


def _has_shell_substitution(command: str) -> bool:
    """True pokud command obsahuje shell substitution: `$(...)`, `` `...` ``,
    process substitution `<(...)` / `>(...)`.

    Substituce spawnují subshell s libovolným commandem, který shlex netokenizuje
    jako čistý argv (`$(rm` se objeví jako jeden token, nematch BASH_DESTRUCTIVE).
    Bezpečnostní eskalace: substituce = potenciální RCE → requires_explicit.
    """
    return ("$(" in command) or ("`" in command) or ("<(" in command) or (">(" in command)


# Pre-compiled word-boundary regex pro destructive tokens. Build lazily v
# `_has_destructive_word`, aby BASH_DESTRUCTIVE_COMMANDS byl k dispozici.
_DESTRUCTIVE_WORD_RE: re.Pattern[str] | None = None


def _has_destructive_word(command: str) -> bool:
    """True pokud command (raw string) obsahuje destruktivní root jako word.

    Defense in depth proti tokenizaci-bypass:
    - `bash -c "rm -rf x"` — shlex tokenizuje na ["bash", "-c", "rm -rf x"];
      token "rm -rf x" je jeden string, basename match selže. Raw regex zachytí.
    - `python -c "os.unlink(...)"` — netriviální, ale aspoň rm/sudo v stringu chytí.
    - `env LC_ALL=C rm foo` (už chytí token scan, ale belt&suspenders).

    Trade-off: false-positive pokud destruktivní slovo je v textu argumentu
    (`cat rm.txt`, `echo "remove rm"`). User schválí — bezpečnost > UX.
    Underscore je word char, takže `README_rm_fix.md` nematch.
    """
    global _DESTRUCTIVE_WORD_RE
    if _DESTRUCTIVE_WORD_RE is None:
        from voice.agent.config import BASH_DESTRUCTIVE_COMMANDS
        pattern = r"\b(?:" + "|".join(re.escape(c) for c in BASH_DESTRUCTIVE_COMMANDS) + r")\b"
        _DESTRUCTIVE_WORD_RE = re.compile(pattern)
    return bool(_DESTRUCTIVE_WORD_RE.search(command))


def _has_destructive_token(command: str) -> bool:
    """True pokud command obsahuje destruktivní token v ANY pozici segmentu
    (po wrapper skipu i přes wrappers s flagy), `find` s mutating flagem,
    file-write redirect (`>`/`>>`/`&>`/`>&` na soubor), nebo shell substitution.

    Scan VŠECH tokenů (ne jen first) brání wrapper-flag bypassu:
    `xargs -I{} rm -rf /tmp/x` — wrapper má vlastní flagy, takže `_effective_root`
    by se zastavil na `-I{}` a minul `rm`. Per-token scan to zachytí.

    Trade-off: false-positive na harmless commandy obsahující destructive name
    jako operand (`echo rm`, `cat rm.txt`). User schválí — bezpečnost > UX.

    Bezpečnostní fallback: pokud shlex.split selže, vrací True (lepší false
    positive než leak).
    """
    from voice.agent.config import BASH_DESTRUCTIVE_COMMANDS, FIND_MUTATING_FLAGS

    # Shell substitution → potenciální RCE skrytý před shlex tokenizací.
    if _has_shell_substitution(command):
        return True

    # Raw destructive-word scan (interpreter -c bypass: `bash -c "rm -rf"`).
    if _has_destructive_word(command):
        return True

    # File redirect → write side effect → vyžaduje explicit phrase.
    if _has_file_redirect(command):
        return True

    for segment in _SEGMENT_SEP_RE.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return True
        if not tokens:
            continue
        # Defense in depth: scan VŠECHNY tokeny (ne jen first), basename-match
        # destructive set. Pokrývá: `env rm`, `xargs -I{} rm`, `nohup -- rm`,
        # `command rm`, `time rm`, atd. — vše kde destructive root není token[0].
        for tok in tokens:
            bn = os.path.basename(tok) or tok
            if bn in BASH_DESTRUCTIVE_COMMANDS:
                return True
        # find s mutating flagem (-delete / -exec / -fprint*) — root detection
        # samostatně, protože MUTATING_FLAGS jsou flagy ne tokeny.
        # Pozn.: jen MUTATING (ne FIND_FORBIDDEN_FLAGS), aby `-L`/`-H`/`-follow`
        # nepřeřadilo na destructive — to je read-escape, ne file mutation.
        root = _effective_root(tokens)
        if root == "find" and any(t in FIND_MUTATING_FLAGS for t in tokens[1:]):
            return True
    return False


def _is_path_outside_workdir(operand: str, workdir: Path) -> bool:
    """True pokud operand jako filesystem path resolvuje mimo workdir.

    Defense vůči:
    - absolutním path (`/etc/passwd`)
    - traversal (`../foo`)
    - symlinkům uvnitř workdir mířícím ven (LLM napíše `ln -s /etc/passwd link`
      → pak `cat link` přečte mimo sandbox)
    - flag-value path (`--files0-from=/etc/passwd`)

    False pokud:
    - operand neexistuje a nevypadá jako path (plain identifier/pattern)
    - operand resolvuje uvnitř workdir
    """
    from voice.agent.tools._sandbox import is_inside_workdir

    if not operand:
        return False
    # Plain identifier (žádný path separator, žádný `..`, neexistuje jako
    # file ani symlink): considered safe — typicky rg pattern, git branch,
    # find condition value.
    looks_like_path = ("/" in operand) or operand == ".." or operand.startswith("..")
    if not looks_like_path:
        candidate = workdir / operand
        try:
            if not candidate.exists() and not candidate.is_symlink():
                return False
        except OSError:
            return False
    # Resolve fully (follows symlinks). Abs path operand overrides workdir join.
    try:
        resolved = (workdir / operand).resolve(strict=False)
    except (OSError, ValueError, RuntimeError):
        return True  # cannot resolve → safer to reject AUTO
    return not is_inside_workdir(resolved, workdir)


def _argv_safe_for_auto(root: str, argv: list[str], workdir: Path) -> tuple[bool, str]:
    """Validuje argv pro AUTO běh:
    - per-command flag denylist (`rg --pre`, `rg -z`)
    - žádné path operandy mimo workdir (absolute, ..,  symlink ven)
    - žádné `--flag=path` kde path je mimo workdir
    Vrací (safe, reason). False = ASK medium.
    """
    from voice.agent.config import (
        RG_FORBIDDEN_FLAGS, FIND_FORBIDDEN_FLAGS, LS_FORBIDDEN_FLAGS,
        WC_FORBIDDEN_FLAGS,
    )

    # Per-command forbidden flags. (git je vyhozen z AUTO úplně.)
    forbidden: frozenset[str] = frozenset()
    if root == "rg":
        forbidden = RG_FORBIDDEN_FLAGS
    elif root == "find":
        forbidden = FIND_FORBIDDEN_FLAGS
    elif root == "ls":
        forbidden = LS_FORBIDDEN_FLAGS
    elif root == "wc":
        forbidden = WC_FORBIDDEN_FLAGS

    # Sběr "single-letter" forbidden short flagů (např. `-L`, `-H`, `-o`, `-c`).
    # Použito pro detekci combined cluster `ls -laL` (kde `L` je uvnitř, ne na konci).
    forbidden_short_chars: set[str] = {
        f[1] for f in forbidden if len(f) == 2 and f.startswith("-") and f[1].isalpha()
    }

    # Long-flag prefixes from forbidden set — GNU coreutils (wc, ls, cat, head,
    # tail) přijímají jakýkoli unambiguous prefix long option. Bez znalosti
    # plného option setu binárky nelze určit ambiguity, takže safer to reject
    # každý prefix ≥3 chars (`--x` nebo delší), který prefixuje forbidden flag.
    # Cena: false-positive pro rg/find (které GNU abbreviation nemají) — uživatel
    # musí napsat flag plně, jinak ASK. Acceptable trade-off.
    forbidden_long: list[str] = [
        f for f in forbidden if f.startswith("--") and len(f) > 2
    ]

    for tok in argv[1:]:
        # Plný match: `--pre`, `-c`.
        if tok in forbidden:
            return False, f"{root} flag {tok!r} forbidden in AUTO"
        # Inline value forma: `--pre=rm`, `--output=/path`, `-files0-from=list`.
        if "=" in tok and tok.startswith("-"):
            flag = tok.split("=", 1)[0]
            if flag in forbidden:
                return False, f"{root} flag {tok!r} forbidden in AUTO"
        # GNU long-option abbreviation bypass: `wc --files0-f=list`,
        # `ls --der --recursive`. Token (bez `=value`) je prefix nějakého
        # forbidden long flagu → reject.
        if tok.startswith("--") and len(tok) > 2:
            flag = tok.split("=", 1)[0] if "=" in tok else tok
            for fb in forbidden_long:
                if len(flag) < len(fb) and fb.startswith(flag):
                    return False, (
                        f"{root} long-flag {tok!r} is prefix of "
                        f"forbidden {fb!r} (GNU abbreviation)"
                    )
        # Short flag cluster: `-laL`, `-RL`, `-oFILE`, `-cKEY=VAL`, `-1LR`, `-0L`.
        # Scan VŠECHNY chars clusteru (continue na non-alpha — digit/special se
        # přeskočí ale loop pokračuje). Pokud kterýkoli alpha char je v
        # forbidden_short_chars → reject. Kryje:
        #   `ls -1LR` (`L` na pozici 2 za digitem `1`)
        #   `rg -0L pattern` (`L` za digitem `0`)
        #   `rg -fPATH` (`f` první char, attached value)
        # Trade-off: false-positive pokud cluster obsahuje value-letter shodný
        # s forbidden short (např. `-IFILE` v ls — neexistující kombinace).
        if len(tok) > 2 and tok.startswith("-") and not tok.startswith("--"):
            cluster = tok[1:]
            for ch in cluster:
                if not ch.isalpha():
                    continue  # digit / special — pokračuj scanovat
                if ch in forbidden_short_chars:
                    return False, f"{root} short flag {ch!r} in cluster {tok!r} forbidden"

    # Path operand check — pokrývá:
    #   - positional path operandy (cat /etc/passwd, ls ../foo)
    #   - symlinky uvnitř workdir mířící ven (cat secret_link)
    #   - flag-value paths (rg --iglob=../../.ssh/id_rsa, wc --files0-from=/etc/x)
    #   - end-of-options `--`: po něm jsou všechny tokeny positional, i ty
    #     začínající `-` (jinak by `cat -- -secret_link` projel skip-flag větví).
    after_double_dash = False
    for tok in argv[1:]:
        if tok == "--":
            after_double_dash = True
            continue
        if not after_double_dash and tok.startswith("-"):
            # Flag — extract value pokud `=` form, check path safety.
            if "=" in tok:
                value = tok.split("=", 1)[1]
                if value and _is_path_outside_workdir(value, workdir):
                    return False, f"flag value path {value!r} outside workdir"
            continue
        # Positional operand (incl. `-leading` po `--`).
        if _is_path_outside_workdir(tok, workdir):
            return False, f"path operand {tok!r} outside workdir"
    return True, ""


def _classify_bash_cwd(cwd_str: str, workdir: Path) -> tuple[Path | None, str | None]:
    """Validate cwd argument. Vrací (resolved_cwd, error). Error None = OK.
    Empty cwd → (workdir, None) (= default)."""
    from voice.agent.tools._sandbox import resolve_safe, is_inside_workdir, is_special_file

    if not cwd_str:
        return workdir, None
    resolved, err = resolve_safe(cwd_str, workdir)
    if resolved is None:
        return None, err or "invalid cwd"
    if is_special_file(resolved):
        return None, "cwd is a special file"
    if not is_inside_workdir(resolved, workdir):
        return None, "cwd outside workdir"
    return resolved, None


@register_classifier("run_bash")
def _cls_run_bash(args: dict, workdir: Path) -> PermissionResult:
    """Klasifikace bash commandu — viz plán Phase 3.

    AUTO = root v BASH_AUTO_COMMANDS, žádné shell metaznaky, validní subkomandy
           pro git/find. Spouští se shell=False s argv listem.
    ASK = vše ostatní (neznámý command, pipes/redirecty). Po approve běží
          /bin/bash -c.
    ASK + requires_explicit = destruktivní token v jakémkoli segmentu.
    DENY = prázdný command, invalid cwd, special file cwd, neresolved syntax.
    """
    from voice.agent.config import (
        BASH_AUTO_COMMANDS,
        FIND_MUTATING_FLAGS,
    )

    command = str(args.get("command", "")).strip()
    cwd_str = str(args.get("cwd", ""))

    if not command:
        return PermissionResult(
            decision=Decision.DENY,
            reason="empty command",
            summary="bash: prázdný command",
            risk="high",
        )

    # cwd validation
    resolved_cwd, cwd_err = _classify_bash_cwd(cwd_str, workdir)
    if resolved_cwd is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=cwd_err or "invalid cwd",
            summary=f"bash: cwd odmítnuto ({cwd_err})",
            risk="high",
        )

    short_cmd = command if len(command) <= 80 else command[:77] + "..."

    # Destructive root token check — applies for both shell-mode and argv-mode.
    if _has_destructive_token(command):
        return PermissionResult(
            decision=Decision.ASK,
            reason="destructive root token detected",
            summary=f'$ {short_cmd}  [vyžaduje "ano povoluju"]',
            risk="destructive",
            requires_explicit=True,
        )

    # Shell metaznaky → ASK (medium) — po approve běží /bin/bash -c.
    if _has_shell_metas(command):
        return PermissionResult(
            decision=Decision.ASK,
            reason="shell metacharacters require approval",
            summary=f"$ {short_cmd}  [shell features]",
            risk="medium",
        )

    # Čistý argv path — parse a aplikuj allowlist logiku.
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return PermissionResult(
            decision=Decision.DENY,
            reason=f"invalid shell syntax: {e}",
            summary="bash: invalid syntax",
            risk="high",
        )

    if not argv:
        return PermissionResult(
            decision=Decision.DENY,
            reason="empty argv after shlex",
            summary="bash: prázdný command",
            risk="high",
        )

    root = argv[0]

    # find s mutating flagy = destructive (find -delete, find -exec rm).
    # Symlink-follow flagy (-L/-H/-follow) řeší samostatně _argv_safe_for_auto
    # (jen ASK medium, ne destructive).
    if root == "find" and any(a in FIND_MUTATING_FLAGS for a in argv[1:]):
        return PermissionResult(
            decision=Decision.ASK,
            reason="find with mutating flag (-delete/-exec)",
            summary=f'$ {short_cmd}  [vyžaduje "ano povoluju"]',
            risk="destructive",
            requires_explicit=True,
        )

    # Root není v AUTO allowlistu → ASK medium.
    if root not in BASH_AUTO_COMMANDS:
        return PermissionResult(
            decision=Decision.ASK,
            reason=f"command {root!r} not in AUTO allowlist",
            summary=f"$ {short_cmd}",
            risk="medium",
        )

    # Per-command argv validation: forbidden flags + path operand check
    # (incl. symlink-aware containment).
    safe, reason = _argv_safe_for_auto(root, argv, workdir)
    if not safe:
        return PermissionResult(
            decision=Decision.ASK,
            reason=reason,
            summary=f"$ {short_cmd}",
            risk="medium",
        )

    return PermissionResult(
        decision=Decision.AUTO,
        reason=f"{root} in AUTO allowlist",
        summary=f"$ {short_cmd}",
        risk="low",
    )


# ----------------------------------------------------------------------
# Phase 4: Web tools (fetch_url, web_search)
# ----------------------------------------------------------------------


def _is_private_or_blocked_host(host: str) -> tuple[bool, str]:
    """SSRF defense: blokuj private/loopback/link-local/multicast/reserved IPs
    a special-case hostnames (localhost, *.localhost, …). Vrací (blocked, reason).

    Pro DNS hostnames bez literal IP vrátí (False, "") — runtime fetch musí
    resolvovat a re-checkovat (TOCTOU defense udělá custom backend
    v tools/web.py). Tady jen statická validace.

    IPv4-mapped IPv6 (`::ffff:127.0.0.1`) — v Pythonu < 3.12 `is_private` vrací
    False, takže explicit unwrap přes `.ipv4_mapped` před privacy testem.
    Stejně tak NAT64 (`64:ff9b::/96`) a 6to4/Teredo IPv6 mapped IPv4 — tahaj
    embedded IPv4, ten musí být public.
    """
    import ipaddress

    if not host:
        return True, "empty host"

    h = host.strip().lower()
    # Bracketed IPv6: `[::1]` → `::1`
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]

    # Special hostnames — blokuj bez DNS lookupu.
    blocked_names = {
        "localhost", "ip6-localhost", "ip6-loopback",
        "broadcasthost",
    }
    if h in blocked_names or h.endswith(".localhost") or h.endswith(".local"):
        return True, f"blocked hostname {host!r}"

    # Try IP literal — pokud parseuje, classifikuj.
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False, ""  # not an IP literal — DNS resolve later

    # IPv4-mapped IPv6 unwrap. Python 3.11 nemá auto is_private propagation.
    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            check_ip: ipaddress._BaseAddress = mapped
        else:
            check_ip = ip
    else:
        check_ip = ip

    if (
        check_ip.is_private
        or check_ip.is_loopback
        or check_ip.is_link_local
        or check_ip.is_multicast
        or check_ip.is_reserved
        or check_ip.is_unspecified
        or getattr(check_ip, "is_site_local", False)
    ):
        return True, f"blocked IP {ip}"

    # Block IPv6 translation/compatibility prefixes that might embed private IPv4s
    if isinstance(check_ip, ipaddress.IPv6Address):
        # NAT64 Well-Known Prefix (RFC 6052)
        if check_ip in ipaddress.IPv6Network("64:ff9b::/96"):
            return True, f"blocked NAT64 IP {ip}"
        # NAT64 Local-Use Prefix (RFC 8215)
        if check_ip in ipaddress.IPv6Network("64:ff9b:1::/48"):
            return True, f"blocked NAT64 IP {ip}"
        # IPv4-Translated (RFC 2765)
        if check_ip in ipaddress.IPv6Network("::ffff:0:0:0/96"):
            return True, f"blocked IPv4-translated IP {ip}"
        # IPv4-Compatible (deprecated)
        if check_ip in ipaddress.IPv6Network("::/96") and check_ip not in (ipaddress.IPv6Address("::1"), ipaddress.IPv6Address("::")):
            return True, f"blocked IPv4-compatible IP {ip}"

    # CGNAT (100.64.0.0/10) — RFC 6598 shared address space, často interní.
    # ipaddress.is_private nezahrnuje. Blokujeme defensively.
    try:
        if isinstance(check_ip, ipaddress.IPv4Address):
            if check_ip in ipaddress.IPv4Network("100.64.0.0/10"):
                return True, f"blocked CGNAT IP {ip}"
    except ValueError:
        pass

    return False, ""


def _validate_url(url: str) -> tuple[str, str, str]:
    """Parse + validate URL. Vrací (scheme, host, error). error=='' = OK.

    - musí být http/https
    - host nesmí být prázdný
    - host nesmí být private/loopback IP nebo localhost (SSRF defense)
    - userinfo (`user:pass@`) odmítnut (credentials exfiltration / phishing vektor)
    - port mimo standard (80/443/8080/443/8443) → OK, ale runtime to může reject
    """
    from urllib.parse import urlparse

    if not url or not isinstance(url, str):
        return "", "", "empty url"
    if len(url) > 4096:
        return "", "", "url too long"

    try:
        parsed = urlparse(url)
    except (ValueError, TypeError) as e:
        return "", "", f"invalid url: {e}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return scheme, "", f"unsupported scheme {scheme!r}"

    # userinfo block (user:pass@) — credentials v URL je bezpečnostní risk.
    # Pozn.: `parsed.username`/`password` může vyhodit ValueError pro URL kde
    # syntax je nesmyslná — interpretovat jako "invalid url" než "no userinfo".
    try:
        if parsed.username or parsed.password:
            return scheme, "", "url contains userinfo"
    except ValueError as e:
        return scheme, "", f"invalid userinfo: {e}"

    # `parsed.port` může vyhodit ValueError pokud port není integer
    # (`http://host:abc/`) nebo je mimo 0..65535. Bez catch by urllib bombardovala.
    try:
        port = parsed.port
    except ValueError as e:
        return scheme, "", f"invalid port: {e}"
    if port is not None and (port < 1 or port > 65535):
        return scheme, "", f"port {port} out of range"

    host = (parsed.hostname or "").strip()
    if not host:
        return scheme, "", "missing host"

    blocked, why = _is_private_or_blocked_host(host)
    if blocked:
        return scheme, host, why

    return scheme, host, ""


@register_classifier("fetch_url")
def _cls_fetch_url(args: dict, workdir: Path) -> PermissionResult:
    """fetch_url(url) — AUTO pro public http(s); DENY pro file/ftp/private IPs.

    Side-effect: žádný side effect na FS, jen HTTP GET. Risk = low pro veřejnou
    síť. ASK by byl over-paranoidní (LLM dělá research). SSRF guard je tvrdý
    DENY — žádný „ano povoluju" interní síti.
    """
    url = str(args.get("url", "")).strip()
    scheme, host, err = _validate_url(url)
    short = url[:80] + ("…" if len(url) > 80 else "")
    if err:
        return PermissionResult(
            decision=Decision.DENY,
            reason=err,
            summary=f"fetch_url odmítnuto: {err}",
            risk="high",
        )
    return PermissionResult(
        decision=Decision.AUTO,
        reason=f"fetch {scheme}://{host}",
        summary=f"fetch_url: {short}",
        risk="low",
    )


# ----------------------------------------------------------------------
# Phase 5: Philips Hue smart-home tools (light_list, light_set)
# ----------------------------------------------------------------------


@register_classifier("light_list")
def _cls_light_list(args: dict, workdir: Path) -> PermissionResult:
    """light_list — AUTO. Read-only GET na local Hue Bridge.

    Bridge IP a app key jsou load-time fixed v config.py (LLM nekontroluje target).
    Žádný FS side effect, žádná persistent change.
    """
    return PermissionResult(
        decision=Decision.AUTO,
        reason="read-only Hue lights list (local network, config-fixed bridge)",
        summary="light_list",
        risk="low",
    )


@register_classifier("light_set")
def _cls_light_set(args: dict, workdir: Path) -> PermissionResult:
    """light_set(name, on?, brightness?, color_name?) — AUTO low.

    Změna stavu světla je reverzibilní (user může jen vypnout/rozsvítit zpět).
    LLM kontroluje jen name (mapped na resource ID via Hue), bool on, brightness
    integer (validated 0..100), color_name (validated proti fixed paletě).
    Bridge URL je hard-coded z config — žádná URL injection.
    Žádný credentials log, žádný FS side effect.
    """
    name = str(args.get("name", "")).strip()
    if not name:
        return PermissionResult(
            decision=Decision.DENY,
            reason="empty name",
            summary="light_set: prázdné jméno",
            risk="high",
        )
    if len(name) > 80:
        return PermissionResult(
            decision=Decision.DENY,
            reason="name too long",
            summary="light_set: jméno příliš dlouhé",
            risk="high",
        )
    # Reject control chars / newlines (could confuse log readers or
    # log-injection downstream — defense in depth).
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in name):
        return PermissionResult(
            decision=Decision.DENY,
            reason="name contains control characters",
            summary="light_set: jméno obsahuje řídicí znaky",
            risk="high",
        )
    # Validate types early (defense in depth — exec re-validates).
    on_arg = args.get("on", None)
    if on_arg is not None and not isinstance(on_arg, bool):
        return PermissionResult(
            decision=Decision.DENY,
            reason="on must be boolean",
            summary="light_set: on musí být bool",
            risk="high",
        )
    br_arg = args.get("brightness", None)
    if br_arg is not None:
        # bool is subclass of int — `True/False` would coerce to 1.0/0.0 and
        # silently pass; reject explicitly so user must pick the right field.
        if isinstance(br_arg, bool):
            return PermissionResult(
                decision=Decision.DENY,
                reason="brightness must be number, not bool",
                summary="light_set: brightness není číslo",
                risk="high",
            )
        try:
            br = float(br_arg)
        except (TypeError, ValueError):
            return PermissionResult(
                decision=Decision.DENY,
                reason="brightness must be number",
                summary="light_set: brightness není číslo",
                risk="high",
            )
        # NaN/inf bypass: `nan < 0` and `nan > 100` both False → bypass.
        # math.isfinite excludes NaN, +inf, -inf.
        import math as _math
        if not _math.isfinite(br):
            return PermissionResult(
                decision=Decision.DENY,
                reason="brightness must be finite number",
                summary="light_set: brightness není konečné",
                risk="high",
            )
        if br < 0 or br > 100:
            return PermissionResult(
                decision=Decision.DENY,
                reason="brightness out of range 0..100",
                summary="light_set: brightness mimo 0..100",
                risk="high",
            )
    color_arg = args.get("color_name", None)
    if color_arg is not None:
        from voice.agent.tools.hue import COLOR_PALETTE
        cname = str(color_arg).strip().lower()
        if cname not in COLOR_PALETTE:
            return PermissionResult(
                decision=Decision.DENY,
                reason=f"unknown color_name {cname!r}",
                summary=f"light_set: neznámá barva {cname!r}",
                risk="high",
            )
    if on_arg is None and br_arg is None and color_arg is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason="must specify at least one of: on, brightness, color_name",
            summary="light_set: žádná změna",
            risk="high",
        )

    parts = []
    if on_arg is not None:
        parts.append(f"on={on_arg}")
    if br_arg is not None:
        parts.append(f"bri={br_arg}")
    if color_arg is not None:
        parts.append(f"color={color_arg}")
    short_name = name if len(name) <= 40 else name[:37] + "…"
    return PermissionResult(
        decision=Decision.AUTO,
        reason="local hue light control",
        summary=f"light_set {short_name}: {', '.join(parts)}",
        risk="low",
    )


@register_classifier("web_search")
def _cls_web_search(args: dict, workdir: Path) -> PermissionResult:
    """web_search(query, count?) — AUTO. Brave Search API call, no side effects."""
    from voice.agent.config import WEB_SEARCH_MAX_COUNT

    query = str(args.get("query", "")).strip()
    if not query:
        return PermissionResult(
            decision=Decision.DENY,
            reason="empty query",
            summary="web_search: prázdný dotaz",
            risk="high",
        )
    if len(query) > 400:
        return PermissionResult(
            decision=Decision.DENY,
            reason="query too long (>400 chars)",
            summary="web_search: dotaz příliš dlouhý",
            risk="high",
        )
    count_arg = args.get("count", None)
    if count_arg is not None:
        try:
            count = int(count_arg)
        except (TypeError, ValueError):
            return PermissionResult(
                decision=Decision.DENY,
                reason="count must be integer",
                summary="web_search: neplatný count",
                risk="high",
            )
        if count < 1 or count > WEB_SEARCH_MAX_COUNT:
            return PermissionResult(
                decision=Decision.DENY,
                reason=f"count out of range 1..{WEB_SEARCH_MAX_COUNT}",
                summary=f"web_search: count mimo 1..{WEB_SEARCH_MAX_COUNT}",
                risk="high",
            )
    short = query if len(query) <= 60 else query[:57] + "…"
    return PermissionResult(
        decision=Decision.AUTO,
        reason="brave web search",
        summary=f'web_search: "{short}"',
        risk="low",
    )


@register_classifier("ask_claude")
def _cls_ask_claude(args: dict, workdir: Path) -> PermissionResult:
    """ask_claude(prompt, system?, max_tokens?) — ASK medium.

    Volá Anthropic Messages API (paid, external network). Defaultně přes
    approval flow (ne AUTO), protože:
      1) každý volání = cost
      2) odesílá LLM-controlled obsah na třetí stranu
      3) odpověď není FS/shell side-effect, ale ovlivní následující agent reasoning
    """
    from voice.agent.config import (
        CLAUDE_MAX_PROMPT_BYTES,
        CLAUDE_MAX_SYSTEM_BYTES,
        CLAUDE_MAX_TOKENS_LIMIT,
    )

    prompt_arg = args.get("prompt", None)
    if not isinstance(prompt_arg, str):
        return PermissionResult(
            decision=Decision.DENY,
            reason="prompt must be string",
            summary="ask_claude: prompt není string",
            risk="high",
        )
    prompt = prompt_arg.strip()
    if not prompt:
        return PermissionResult(
            decision=Decision.DENY,
            reason="empty prompt",
            summary="ask_claude: prázdný prompt",
            risk="high",
        )
    # Byte size check (UTF-8) — defense proti gigantickému promptu = cost blow-up.
    try:
        prompt_bytes = len(prompt.encode("utf-8"))
    except UnicodeEncodeError:
        return PermissionResult(
            decision=Decision.DENY,
            reason="prompt is not valid utf-8",
            summary="ask_claude: prompt není UTF-8",
            risk="high",
        )
    if prompt_bytes > CLAUDE_MAX_PROMPT_BYTES:
        return PermissionResult(
            decision=Decision.DENY,
            reason=f"prompt too large ({prompt_bytes} > {CLAUDE_MAX_PROMPT_BYTES} bytes)",
            summary=f"ask_claude: prompt > {CLAUDE_MAX_PROMPT_BYTES // 1024} KiB",
            risk="high",
        )

    system_arg = args.get("system", None)
    if system_arg is not None:
        if not isinstance(system_arg, str):
            return PermissionResult(
                decision=Decision.DENY,
                reason="system must be string",
                summary="ask_claude: system není string",
                risk="high",
            )
        try:
            system_bytes = len(system_arg.encode("utf-8"))
        except UnicodeEncodeError:
            return PermissionResult(
                decision=Decision.DENY,
                reason="system is not valid utf-8",
                summary="ask_claude: system není UTF-8",
                risk="high",
            )
        if system_bytes > CLAUDE_MAX_SYSTEM_BYTES:
            return PermissionResult(
                decision=Decision.DENY,
                reason=f"system too large ({system_bytes} > {CLAUDE_MAX_SYSTEM_BYTES} bytes)",
                summary=f"ask_claude: system > {CLAUDE_MAX_SYSTEM_BYTES // 1024} KiB",
                risk="high",
            )

    max_tokens_arg = args.get("max_tokens", None)
    if max_tokens_arg is not None:
        # bool je int subclass — odmítnout explicit, jinak True/False projde jako 1/0.
        if isinstance(max_tokens_arg, bool):
            return PermissionResult(
                decision=Decision.DENY,
                reason="max_tokens must be int, not bool",
                summary="ask_claude: max_tokens není int",
                risk="high",
            )
        try:
            max_tokens = int(max_tokens_arg)
        except (TypeError, ValueError):
            return PermissionResult(
                decision=Decision.DENY,
                reason="max_tokens must be integer",
                summary="ask_claude: max_tokens není int",
                risk="high",
            )
        if max_tokens < 1 or max_tokens > CLAUDE_MAX_TOKENS_LIMIT:
            return PermissionResult(
                decision=Decision.DENY,
                reason=f"max_tokens out of range 1..{CLAUDE_MAX_TOKENS_LIMIT}",
                summary=f"ask_claude: max_tokens mimo 1..{CLAUDE_MAX_TOKENS_LIMIT}",
                risk="high",
            )

    short = prompt if len(prompt) <= 60 else prompt[:57] + "…"
    return PermissionResult(
        decision=Decision.ASK,
        reason="external paid LLM call (Anthropic API)",
        summary=f'ask_claude: "{short}"',
        risk="medium",
    )
