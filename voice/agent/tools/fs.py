"""File-system tooly: read_file, list_files, glob, grep, write_file, edit_file.

Sandbox bezpečnost je delegovaná na `_sandbox.resolve_safe` — všechny path
operace musí projít přes něj. Tool nikdy nesahá na cesty mimo workdir/allowlist
přímo — classifier rozhoduje AUTO/ASK před exekucí.
"""
from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable

from voice.agent import config
from voice.agent.tools._sandbox import is_inside_workdir, resolve_safe
from voice.agent.tools.base import ExecuteContext, Tool


# ---------------------------------------------------------------------------
# Tool output limity (zabraňují flooding LLM kontextu i přes byte cap loopu).
# ---------------------------------------------------------------------------
READ_DEFAULT_LIMIT_LINES: int = 2000
GLOB_MAX_MATCHES: int = 1000
GREP_MAX_MATCHES: int = 200
GREP_TIMEOUT_SEC: int = 10
LIST_MAX_ENTRIES: int = 1000
# ReDoS mitigation pro Python fallback grep: limity bez perfect timeoutu,
# ale s coopativním time-budgetem a velikostními capy.
GREP_PATTERN_MAX_LEN: int = 200
GREP_LINE_MAX_BYTES: int = 16 * 1024
GREP_PYTHON_MAX_FILE_BYTES: int = 1024 * 1024  # 1 MiB per file ve fallback
RG_MAX_FILESIZE: str = "10M"

# Adresáře, které grep fallback nepoleze (snižuje šum / nestabilní výstup).
GREP_IGNORE_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".pytest_cache",
    ".venv", "venv", ".venv-tts", "dist", "build",
})


def _err(message: str) -> dict:
    return {"ok": False, "error": message}


def _resolved_or_recheck(args: dict, ctx: ExecuteContext,
                          *, default_path: str = "") -> tuple[Path | None, str | None]:
    """Vrátí canonical path, na které stojí permission rozhodnutí.

    Preferuje `ctx.resolved_path` z classifieru (eliminuje TOCTOU gap mezi
    klasifikací a exekucí). Pokud chybí (přímé volání tool.execute v testech
    bez agent loopu), fallback na `resolve_safe` v tool — méně bezpečné, ale
    konzistentní s legacy chováním. Post-class TOCTOU defense: znovu ověříme,
    že cesta není special file (kdyby ji někdo swapnul mezi classifierem
    a exekucí).
    """
    from voice.agent.tools._sandbox import is_special_file

    if ctx.resolved_path is not None:
        if is_special_file(ctx.resolved_path):
            return None, f"target became special file after classification: {ctx.resolved_path}"
        return ctx.resolved_path, None
    # Fallback: re-resolve.
    raw = args.get("path", default_path)
    if isinstance(raw, str) and not raw and default_path:
        raw = default_path
    return resolve_safe(raw, ctx.workdir)


def _verify_stable_resolve(resolved: Path) -> str | None:
    """Best-effort TOCTOU mitigation pro write/edit.

    Po `resolve_safe` ověříme, že:
      1) resolved.resolve() == resolved (path se nepřeresolvoval — žádný nový
         symlink uvnitř cesty mezi check a write).
      2) lstat(resolved) není symlink (kdyby někdo mezi tím vyměnil regular file
         za symlink).
      3) lstat(parent) není symlink (parent dir swap pro outside-workdir write).

    Není to 100% — full safety vyžaduje `openat` s `O_NOFOLLOW` + `dir_fd`,
    což je out-of-scope pro Phase 2 (LLM nemá symlink-create tool ani shell).
    Vrátí None pokud OK, nebo error message.
    """
    try:
        re_resolved = resolved.resolve(strict=False)
    except OSError as e:
        return f"re-resolve failed: {type(e).__name__}: {e}"
    if re_resolved != resolved:
        return "path resolution changed since policy check (TOCTOU)"
    try:
        st = resolved.lstat()
        if stat.S_ISLNK(st.st_mode):
            return "target is a symlink (refusing to follow at write time)"
    except FileNotFoundError:
        pass  # write_file creates new files; OK
    except OSError as e:
        return f"lstat failed: {type(e).__name__}: {e}"
    parent = resolved.parent
    try:
        pst = parent.lstat()
        if stat.S_ISLNK(pst.st_mode):
            return "parent directory is a symlink (refusing to write through it)"
    except FileNotFoundError:
        # Parent může být vytvořen create_dirs flagem; caller to ošetří.
        pass
    except OSError as e:
        return f"parent lstat failed: {type(e).__name__}: {e}"
    return None


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


async def _read_file_exec(args: dict, ctx: ExecuteContext) -> dict:
    offset_raw = args.get("offset")
    limit_raw = args.get("limit")

    resolved, err = _resolved_or_recheck(args, ctx)
    if resolved is None:
        return _err(err or "invalid path")
    if not resolved.exists():
        return _err(f"file not found: {resolved}")
    if not resolved.is_file():
        return _err(f"not a regular file: {resolved}")

    try:
        size = resolved.stat().st_size
    except OSError as e:
        return _err(f"stat failed: {e}")

    cap = config.READ_SIZE_CAP_BYTES
    truncated_bytes = size > cap
    try:
        with resolved.open("rb") as fh:
            raw = fh.read(cap)
    except OSError as e:
        return _err(f"read failed: {e}")

    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    total_lines = len(lines)

    # 1-indexed offset, default 1
    try:
        offset = int(offset_raw) if offset_raw is not None else 1
    except (TypeError, ValueError):
        offset = 1
    if offset < 1:
        offset = 1

    try:
        limit = int(limit_raw) if limit_raw is not None else READ_DEFAULT_LIMIT_LINES
    except (TypeError, ValueError):
        limit = READ_DEFAULT_LIMIT_LINES
    if limit < 1:
        limit = READ_DEFAULT_LIMIT_LINES

    start_idx = offset - 1
    end_idx = min(start_idx + limit, total_lines)
    sliced = lines[start_idx:end_idx]
    truncated_lines = end_idx < total_lines or start_idx > 0

    # Formát "<width>\t<text>" — Claude Code Read styl, model snadno najde line N.
    width = max(6, len(str(end_idx if end_idx else 1)))
    formatted = "\n".join(f"{(start_idx + i + 1):>{width}}\t{line}" for i, line in enumerate(sliced))
    return {
        "ok": True,
        "content": formatted,
        "path": str(resolved),
        "total_lines": total_lines,
        "shown_range": [start_idx + 1, end_idx],
        "truncated": truncated_lines or truncated_bytes,
        "size_bytes": size,
    }


READ_FILE = Tool(
    name="read_file",
    description=(
        "Read a text file and return its contents with 1-indexed line numbers. "
        "Uses 256 KiB byte cap + 2000-line default limit. Use offset/limit for "
        "paged reads of large files."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or workdir-relative path."},
            "offset": {"type": "integer", "minimum": 1,
                       "description": "1-indexed start line (default 1)."},
            "limit": {"type": "integer", "minimum": 1,
                      "description": "Max lines to return (default 2000)."},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    execute=_read_file_exec,
)


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


async def _list_files_exec(args: dict, ctx: ExecuteContext) -> dict:
    resolved, err = _resolved_or_recheck(args, ctx, default_path=".")
    if resolved is None:
        return _err(err or "invalid path")
    if not resolved.exists():
        return _err(f"path not found: {resolved}")
    if not resolved.is_dir():
        return _err(f"not a directory: {resolved}")

    try:
        items = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return _err(f"listdir failed: {e}")

    entries: list[dict] = []
    truncated = False
    for it in items:
        if len(entries) >= LIST_MAX_ENTRIES:
            truncated = True
            break
        try:
            st = it.lstat()
        except OSError:
            continue
        if it.is_symlink():
            kind = "symlink"
        elif it.is_dir():
            kind = "dir"
        elif it.is_file():
            kind = "file"
        else:
            kind = "other"
        entries.append({"name": it.name, "type": kind, "size": int(st.st_size)})
    return {"ok": True, "path": str(resolved), "entries": entries, "truncated": truncated}


LIST_FILES = Tool(
    name="list_files",
    description="List entries of a directory (non-recursive). Returns name, type, size.",
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Directory path (absolute or workdir-relative). Default '.'."},
        },
        "additionalProperties": False,
    },
    execute=_list_files_exec,
)


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


async def _glob_exec(args: dict, ctx: ExecuteContext) -> dict:
    pattern = args.get("pattern", "")
    if not isinstance(pattern, str) or not pattern:
        return _err("empty pattern")
    # Traversal DoS: `../**/*` by donutil `Path.glob` skenovat mimo workdir
    # i přes filter na výstupu (I/O spike + permission errors). Reject early.
    if ".." in Path(pattern).parts:
        return _err("pattern must not contain '..'")
    if pattern.startswith("/"):
        return _err("pattern must be relative (no absolute paths)")

    base, err = _resolved_or_recheck(args, ctx, default_path=".")
    if base is None:
        return _err(err or "invalid path")
    if not base.exists() or not base.is_dir():
        return _err(f"base not a directory: {base}")

    matches: list[str] = []
    truncated = False
    base_resolved = base.resolve()
    workdir_resolved = ctx.workdir.resolve()
    base_inside = base_resolved.is_relative_to(workdir_resolved)
    try:
        for p in base.glob(pattern):
            if len(matches) >= GLOB_MAX_MATCHES:
                truncated = True
                break
            # Per-match safety — symlink ven z workdir nesmí proklouznout.
            try:
                p_resolved = p.resolve(strict=False)
            except OSError:
                continue
            # Match musí ležet uvnitř base (= ne escape přes symlink). Pro user-approved
            # outside-workdir base ponecháme absolutní cesty.
            if not p_resolved.is_relative_to(base_resolved):
                continue
            if base_inside:
                try:
                    matches.append(str(p_resolved.relative_to(workdir_resolved)))
                except ValueError:
                    continue
            else:
                matches.append(str(p_resolved))
    except (OSError, ValueError) as e:
        return _err(f"glob failed: {type(e).__name__}: {e}")

    return {"ok": True, "matches": matches, "count": len(matches), "truncated": truncated}


GLOB = Tool(
    name="glob",
    description=(
        "Match files by glob pattern (e.g. '**/*.py'). Only paths inside workdir "
        "are returned. Cap 1000 matches."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (pathlib syntax)."},
            "path": {"type": "string",
                     "description": "Base directory (default '.', workdir root)."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    execute=_glob_exec,
)


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def _grep_via_rg(pattern: str, base: Path, workdir: Path, glob_filter: str | None,
                 case_insensitive: bool) -> tuple[list[dict] | None, bool, str | None]:
    """Vrátí (matches, truncated, error). matches=None pokud rg chybí → fallback.

    Bezpečnostní flagy:
        --no-config  ignoruj RIPGREP_CONFIG_PATH (mohl by zapnout --follow/--pre)
        --no-follow  nepřechod přes symlinky (ani uvnitř matched souboru)
        --max-filesize 10M  per-file cap → ochrana před DoS na obří soubor
        --null  použij NUL separátor (file\\0line\\0text) → robustní vůči `:` v cestě

    Stdout je čten po chunckách a po globálním capu GREP_MAX_MATCHES se proces
    killuje — `-m` v rg je per-file, ne globální, takže by mohl alokovat hodně RAM.
    """
    rg = shutil.which("rg")
    if rg is None:
        return None, False, None
    cmd = [rg, "--no-config", "--no-follow", "--no-heading", "--with-filename",
           "--line-number", "--color=never", "--null",
           "--max-filesize", RG_MAX_FILESIZE,
           "-m", str(GREP_MAX_MATCHES + 1)]
    if case_insensitive:
        cmd.append("-i")
    if glob_filter:
        cmd.extend(["-g", glob_filter])
    cmd.extend(["--", pattern, str(base)])

    matches: list[dict] = []
    truncated = False
    workdir_resolved = workdir.resolve()
    base_resolved = base.resolve()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError as e:
        return [], False, f"rg launch failed: {type(e).__name__}: {e}"

    deadline = time.monotonic() + GREP_TIMEOUT_SEC
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if time.monotonic() > deadline:
                truncated = True
                break
            if len(matches) >= GREP_MAX_MATCHES:
                truncated = True
                break
            # rg --null formát: filename\0lineno:text  (NUL jen za jménem).
            stripped = line.rstrip("\n")
            file_sep = stripped.find("\x00")
            if file_sep < 0:
                continue
            file_str = stripped[:file_sep]
            rest = stripped[file_sep + 1:]
            colon = rest.find(":")
            if colon < 0:
                continue
            lineno_str = rest[:colon]
            text = rest[colon + 1:]
            try:
                lineno = int(lineno_str)
            except ValueError:
                continue
            # Per-match resolved safety — zahodit cokoli, co neleží uvnitř base
            # (rg by neměl, ale --no-follow + --no-config defense in depth).
            try:
                p_resolved = Path(file_str).resolve(strict=False)
            except OSError:
                continue
            if not p_resolved.is_relative_to(base_resolved):
                continue
            # Format friendly: relativně k workdir, jinak absolutně.
            try:
                file_out = str(p_resolved.relative_to(workdir_resolved))
            except ValueError:
                file_out = str(p_resolved)
            matches.append({"file": file_out, "line": lineno, "text": text})
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        # rg returncode: 0 found, 1 no match, 2 error.
        # Pokud jsme killnuli na cap, returncode != 0/1 — to není error.
        rc = proc.returncode if proc.returncode is not None else -1
        if not truncated and rc not in (0, 1, -9):
            try:
                stderr = (proc.stderr.read() if proc.stderr else "") or ""
            except OSError:
                stderr = ""
            if stderr.strip():
                return [], False, f"rg error (rc={rc}): {stderr.strip()[:400]}"
    return matches, truncated, None


def _grep_via_python(pattern: str, base: Path, workdir: Path, glob_filter: str | None,
                     case_insensitive: bool) -> tuple[list[dict], bool, str | None]:
    """Fallback bez rg: re + os.walk. Pomalé, ale 0 závislostí.

    ReDoS pozn.: Python `re` nemá per-match timeout. Mitigace defense in depth:
        - GREP_PATTERN_MAX_LEN cap na délku regexu (omezí stavový prostor)
        - GREP_LINE_MAX_BYTES cap na délku skenované line (catastrophic backtracking
          škáluje exponenciálně s délkou stringu; krátká line = bounded čas)
        - GREP_PYTHON_MAX_FILE_BYTES cap na velikost souboru
        - GREP_IGNORE_DIRS prune nečte .git/.venv/node_modules
        - GREP_TIMEOUT_SEC kooperativní wall-time check mezi soubory/řádky
    Pro robustní DoS protection je doporučeno mít `rg` v PATH (preferovaná cesta).
    Threat model: LLM agent, single-user dev box; signal-based hard kill se nepoužívá
    aby běh nezasahoval do main asyncio loopu.
    """
    if len(pattern) > GREP_PATTERN_MAX_LEN:
        return [], False, f"pattern too long ({len(pattern)} > {GREP_PATTERN_MAX_LEN}) for Python fallback"
    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return [], False, f"invalid regex: {e}"

    matches: list[dict] = []
    truncated = False
    base_resolved = base.resolve()
    workdir_resolved = workdir.resolve()
    deadline = time.monotonic() + GREP_TIMEOUT_SEC
    for root, dirs, files in os.walk(base):
        if time.monotonic() > deadline:
            truncated = True
            break
        # In-place prune ignored dirs.
        dirs[:] = [d for d in dirs if d not in GREP_IGNORE_DIRS]
        for fname in files:
            if time.monotonic() > deadline:
                truncated = True
                break
            if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                continue
            fpath = Path(root) / fname
            try:
                if fpath.is_symlink() or not fpath.is_file():
                    continue
                try:
                    if fpath.stat().st_size > GREP_PYTHON_MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue
                with fpath.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, start=1):
                        if time.monotonic() > deadline:
                            truncated = True
                            return matches, truncated, None
                        # ReDoS guard: skipni přerostlé řádky (catastrophic backtracking
                        # škáluje s délkou).
                        if len(line) > GREP_LINE_MAX_BYTES:
                            continue
                        if regex.search(line):
                            if len(matches) >= GREP_MAX_MATCHES:
                                truncated = True
                                return matches, truncated, None
                            try:
                                p_resolved = fpath.resolve()
                            except OSError:
                                continue
                            # Defense in depth — files mimo base zahodit.
                            if not p_resolved.is_relative_to(base_resolved):
                                continue
                            try:
                                file_out = str(p_resolved.relative_to(workdir_resolved))
                            except ValueError:
                                file_out = str(p_resolved)
                            matches.append({
                                "file": file_out,
                                "line": lineno,
                                "text": line.rstrip("\n"),
                            })
            except (OSError, UnicodeDecodeError):
                continue
    return matches, truncated, None


async def _grep_exec(args: dict, ctx: ExecuteContext) -> dict:
    pattern = args.get("pattern", "")
    glob_filter = args.get("glob")
    case_insensitive = bool(args.get("case_insensitive", False))
    if not isinstance(pattern, str) or not pattern:
        return _err("empty pattern")
    if glob_filter is not None and not isinstance(glob_filter, str):
        return _err("glob must be string")

    base, err = _resolved_or_recheck(args, ctx, default_path=".")
    if base is None:
        return _err(err or "invalid path")
    if not base.exists() or not base.is_dir():
        return _err(f"not a directory: {base}")

    # rg subprocess je blocking → offload.
    loop = asyncio.get_running_loop()
    rg_result = await loop.run_in_executor(
        None, _grep_via_rg, pattern, base, ctx.workdir, glob_filter, case_insensitive,
    )
    matches, truncated, err = rg_result
    if matches is None:
        # rg nedostupný → Python fallback.
        matches, truncated, err = await loop.run_in_executor(
            None, _grep_via_python, pattern, base, ctx.workdir, glob_filter, case_insensitive,
        )
    if err is not None:
        return _err(err)
    return {"ok": True, "matches": matches, "count": len(matches), "truncated": truncated}


GREP = Tool(
    name="grep",
    description=(
        "Search for a regex pattern in files under a directory. Uses ripgrep if "
        "available, Python re fallback otherwise. Cap 200 matches, 10s timeout."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern."},
            "path": {"type": "string", "description": "Base directory (default '.')."},
            "glob": {"type": "string",
                     "description": "Optional filename glob filter (e.g. '*.py')."},
            "case_insensitive": {"type": "boolean", "description": "Default false."},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    },
    execute=_grep_exec,
)


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, content: str) -> int:
    """Atomic write: temp file in same dir + os.replace. Vrátí bytes_written.

    FD ownership: mkstemp vrací raw fd; pokud `os.fdopen` selže, raw fd by
    se neuzavřel a leakl by do GC. try/finally zajistí close.
    """
    encoded = content.encode("utf-8")
    parent = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".write_", suffix=".tmp")
    fd_closed = False
    try:
        try:
            with os.fdopen(fd, "wb") as fh:
                fd_closed = True  # fdopen převzala ownership fd-čka
                fh.write(encoded)
        except OSError:
            if not fd_closed:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        os.replace(tmp_path, path)
        return len(encoded)
    except OSError:
        # Cleanup pokud replace selže (tmp file zůstal viset).
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


async def _write_file_exec(args: dict, ctx: ExecuteContext) -> dict:
    content = args.get("content", "")
    create_dirs = bool(args.get("create_dirs", False))
    if not isinstance(content, str):
        return _err("content must be string")

    resolved, err = _resolved_or_recheck(args, ctx)
    if resolved is None:
        return _err(err or "invalid path")
    if resolved.exists() and not resolved.is_file():
        return _err(f"target exists and is not a regular file: {resolved}")

    parent = resolved.parent
    if not parent.exists():
        if not create_dirs:
            return _err(f"parent directory does not exist: {parent} (use create_dirs=true)")
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return _err(f"mkdir failed: {e}")

    # TOCTOU mitigation — re-resolve + lstat těsně před zápisem (defense in depth).
    toctou_err = _verify_stable_resolve(resolved)
    if toctou_err is not None:
        return _err(toctou_err)

    created = not resolved.exists()
    try:
        n = _atomic_write(resolved, content)
    except OSError as e:
        return _err(f"write failed: {type(e).__name__}: {e}")
    return {"ok": True, "path": str(resolved), "bytes_written": n, "created": created}


WRITE_FILE = Tool(
    name="write_file",
    description=(
        "Atomically write text content to a file (replaces existing). Outside the "
        "workdir requires explicit destructive confirmation ('ano povoluju')."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "create_dirs": {"type": "boolean",
                            "description": "If true, mkdir -p the parent. Default false."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    execute=_write_file_exec,
)


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


async def _edit_file_exec(args: dict, ctx: ExecuteContext) -> dict:
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))
    if not isinstance(old_string, str) or not isinstance(new_string, str):
        return _err("old_string and new_string must be strings")
    if old_string == "":
        return _err("old_string is empty")
    if old_string == new_string:
        return _err("old_string equals new_string (no-op)")

    resolved, err = _resolved_or_recheck(args, ctx)
    if resolved is None:
        return _err(err or "invalid path")
    if not resolved.exists():
        return _err(f"file not found: {resolved}")
    if not resolved.is_file():
        return _err(f"not a regular file: {resolved}")

    try:
        size = resolved.stat().st_size
    except OSError as e:
        return _err(f"stat failed: {e}")
    if size > config.READ_SIZE_CAP_BYTES:
        return _err(
            f"file too large for edit ({size} bytes > {config.READ_SIZE_CAP_BYTES} cap)"
        )

    try:
        text = resolved.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as e:
        return _err(f"read failed: {type(e).__name__}: {e}")

    count = text.count(old_string)
    if count == 0:
        return _err("old_string not found in file")
    if count > 1 and not replace_all:
        return _err(
            f"old_string is not unique (found {count} matches). "
            f"Provide more context or set replace_all=true."
        )

    new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
    # TOCTOU mitigation — re-resolve + lstat těsně před zápisem.
    toctou_err = _verify_stable_resolve(resolved)
    if toctou_err is not None:
        return _err(toctou_err)
    try:
        n = _atomic_write(resolved, new_text)
    except OSError as e:
        return _err(f"write failed: {type(e).__name__}: {e}")

    return {
        "ok": True,
        "path": str(resolved),
        "replacements": count if replace_all else 1,
        "bytes_written": n,
    }


EDIT_FILE = Tool(
    name="edit_file",
    description=(
        "Replace an exact substring in a file. Fails if old_string is not unique "
        "(unless replace_all=true). Outside the workdir requires explicit destructive "
        "confirmation ('ano povoluju'). 256 KiB file size cap."
    ),
    parameters_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "description": "Default false."},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    },
    execute=_edit_file_exec,
)


ALL_TOOLS: tuple[Tool, ...] = (READ_FILE, LIST_FILES, GLOB, GREP, WRITE_FILE, EDIT_FILE)
