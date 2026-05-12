"""Permission resolver pro tool calls.

`decide(tool_name, args, workdir)` vrátí `PermissionResult` s rozhodnutím
AUTO / ASK / DENY. Třída rozhodnutí závisí na nástroji a argumentech.

Default pro neznámý tool = DENY (bezpečnost > UX). Tooly registrují vlastní
classifier dekorátorem `@register_classifier(name)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable


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


def decide(tool_name: str, args: dict, workdir: Path) -> PermissionResult:
    fn = _CLASSIFIERS.get(tool_name)
    if fn is None:
        return PermissionResult(
            decision=Decision.DENY,
            reason=f"no classifier registered for {tool_name!r}",
            summary=f"Neznámý nástroj {tool_name!r}",
            risk="high",
        )
    return fn(args, workdir)


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
