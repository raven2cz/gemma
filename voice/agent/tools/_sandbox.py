"""Path safety primitives pro FS tooly.

Resolve, sandbox containment a special-file detekce — testováno samostatně,
classifiery a tooly se na tyto helpery spoléhají bez vlastní duplicitní logiky.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

from voice.agent import config


def _resolve_allowlist() -> frozenset[Path]:
    out: set[Path] = set()
    for s in config.READ_ALLOWLIST_PATHS:
        try:
            # strict=False — soubor nemusí existovat (běh v sandboxu bez procfs).
            out.add(Path(s).resolve(strict=False))
        except (OSError, RuntimeError):
            continue
    return frozenset(out)


READ_ALLOWLIST_RESOLVED: frozenset[Path] = _resolve_allowlist()


# Pre-resolve workdir jednou — Path.resolve je drahá a workdir je fixní.
_WORKDIR_CACHE: dict[Path, Path] = {}


def _resolved_workdir(workdir: Path) -> Path:
    cached = _WORKDIR_CACHE.get(workdir)
    if cached is None:
        cached = workdir.resolve()
        _WORKDIR_CACHE[workdir] = cached
    return cached


def is_inside_workdir(path: Path, workdir: Path) -> bool:
    """True pokud `path` (už resolved) je uvnitř `workdir`.

    Používá `Path.is_relative_to` — bezpečné proti `/foo/bar2` vs `/foo/bar`
    string-prefix bugu (kde `/foo/bar` by jinak matchnul `/foo/bar2/baz`).
    """
    try:
        return path.resolve(strict=False).is_relative_to(_resolved_workdir(workdir))
    except (OSError, ValueError):
        return False


def is_special_file(path: Path) -> bool:
    """True pro block/char device, socket, FIFO. Také odmítá `/proc/<pid>`
    (kromě explicitního allowlistu) a všechny cesty v `/dev/`.

    Defense in depth — i kdyby classifier něco propustil, tool helper to
    chytí. Funguje i na neexistujících cestách (vrátí False; ale stat selže
    elsewhere).
    """
    try:
        st = path.lstat()
    except (OSError, ValueError):
        return False
    mode = st.st_mode
    if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        return True
    if stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode):
        return True
    # /dev/* = vždy special (allowlist je jen pro /proc a /etc)
    try:
        if path.is_relative_to(Path("/dev")):
            return True
    except (OSError, ValueError):
        pass
    return False


def _is_under_proc_pid(path: Path) -> bool:
    """True pokud path je uvnitř /proc/<pid>/ (kromě allowlistu).

    `/proc/cpuinfo`, `/proc/version` apod. jsou OK (jsou v allowlistu).
    Ale `/proc/self/environ`, `/proc/1/cmdline` ne — proces-specific data.
    """
    try:
        parts = path.parts
    except (OSError, ValueError):
        return False
    if len(parts) < 3 or parts[0] != "/" or parts[1] != "proc":
        return False
    pid = parts[2]
    # "self", "thread-self" nebo numeric pid = per-proces info
    return pid == "self" or pid == "thread-self" or pid.isdigit()


def resolve_safe(path_str: str, workdir: Path) -> tuple[Path | None, str | None]:
    """Resolves user-provided path string. Vrátí (resolved, error).

    - Prázdný/nestring path → error.
    - Relativní path → relativně k `workdir`.
    - Resolves symlinky (strict=False, soubor nemusí existovat).
    - Odmítá special files (block/char dev, socket, fifo).
    - Odmítá `/proc/<pid>/` (per-process info).
    - NEzkoumá containment — to dělá classifier (může chtít AUTO mimo workdir
      pro allowlist nebo ASK).
    """
    if not isinstance(path_str, str) or not path_str:
        return None, "empty or non-string path"

    # NUL byte = guaranteed reject. Path() by spadl s ValueError, ale chceme
    # konzistentní error message.
    if "\x00" in path_str:
        return None, "path contains NUL byte"

    raw = Path(path_str)
    base = raw if raw.is_absolute() else (workdir / raw)
    try:
        resolved = base.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        # RuntimeError = symlink loop. OSError = ENAMETOOLONG, ELOOP, …
        return None, f"resolve failed: {type(e).__name__}: {e}"

    # Po resolve check special files (real device/socket/fifo) i /proc/<pid>.
    if is_special_file(resolved):
        return None, f"special file rejected: {resolved}"
    if _is_under_proc_pid(resolved) and resolved not in READ_ALLOWLIST_RESOLVED:
        return None, f"/proc/<pid> rejected: {resolved}"

    return resolved, None


def is_read_allowed(path: Path, workdir: Path) -> tuple[bool, str]:
    """Vrací (allowed, reason). True → classifier použije AUTO; False → ASK.

    Allowed pokud: path je uvnitř workdir NEBO matchuje READ_ALLOWLIST.
    Pre-podmínka: `path` už prošel `resolve_safe` (= není special).
    """
    if is_inside_workdir(path, workdir):
        return True, "inside workdir"
    if path in READ_ALLOWLIST_RESOLVED:
        return True, "in read allowlist"
    return False, "outside workdir, not in allowlist"
