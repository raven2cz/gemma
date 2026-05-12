"""Append-only JSONL audit log per tool call (fáze 8).

Každý tool invocation se zapíše jako jeden JSON object na řádek do
`AUDIT_DIR/YYYY-MM-DD.jsonl`. Záznam obsahuje:

  ts              ISO 8601 UTC se sufixem "Z"
  turn_id         turn identifikátor (server přiděluje na začátku každého
                  user dotazu)
  tool_call_id    LLM-přidělené tc_… id (z assistant.tool_calls)
  tool            název toolu (run_bash, read_file, …)
  args            redacted/truncated argumenty
  permission      {decision, risk, reason, requires_explicit, summary}
  approval        "approved" | "denied" | None (None = AUTO nebo DENY,
                  approval nebyl uživateli předložen)
  ok              bool — toolexec uspěl
  error           str | None
  result_bytes    int — délka serialized tool výstupu (utf-8 bytes)
  duration_ms     int — od začátku `_execute_one` do hotového výsledku

Zápis je append-only, soubor je vytvořen s mode 0o600 (může obsahovat
prompt preview → owner-only read). Open je proveden s `O_NOFOLLOW`,
takže symlink na audit dir/file vyústí v OSError místo arbitrary file
write. Existující soubor se forced re-chmodne na 0o600 (defense proti
předpřipravenému 0o644 souboru).

Per-tool redactor truncuje dlouhé string fieldy a NIKDY nezapisuje
plné `content` (write_file/edit_file old_string/new_string)
nebo plný `body` (fetch_url) — to by zbytečně bobtnalo log a vystavovalo
PII/secrets. Generic pravidlo: jakýkoli string > AUDIT_FIELD_CAP charů
se ořeže s „…[truncated]". Key-name based secret scrubber chytá klíče
typu `api_key`, `token`, `password`, `authorization` (case-insensitive
substring) → hodnota se nahradí „<redacted>".

Defense in depth proti malicious args:
  - rekurzivní walker má hard depth limit (_REDACT_MAX_DEPTH)
  - list/dict iterables jsou capped (_REDACT_MAX_ITEMS); přebytek se
    nahradí placeholder `"<N more items truncated>"`
  - non-serializable hodnoty → fallback přes `default=str`

Pure I/O modul. Volat lze i mimo agent loop (testy)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from voice.agent import config

log = logging.getLogger("agent-audit")


# Limity proti zlomyslným nested args (DoS / RecursionError).
_REDACT_MAX_DEPTH = 6
_REDACT_MAX_ITEMS = 50


# Substring patterny v klíčích, které vždy redaktujeme bez ohledu na tool.
# Case-insensitive substring match. Defense pokud LLM předá secret pod
# libovolným klíčem (např. malicious tool args { "headers": { "Authorization": "..." }}).
_SECRET_KEY_HINTS: tuple[str, ...] = (
    "api_key", "apikey", "api-key",
    "app_key", "appkey", "app-key",
    "secret", "password", "passwd", "passphrase",
    "token", "bearer", "authorization", "jwt",
    "private_key", "privatekey",
    "session", "cookie",
    "credential", "credentials",
)

# Známé legitimate klíče, které by jinak triggernuly `token` substring match.
# `max_tokens` je standardní arg do ask_claude — není to secret.
_SAFE_KEY_NAMES: frozenset[str] = frozenset([
    "max_tokens", "n_tokens", "num_tokens",
    "tokens_used", "tokens_limit", "completion_tokens", "prompt_tokens",
])


def _is_secret_key(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    low = name.lower()
    if low in _SAFE_KEY_NAMES:
        return False
    return any(h in low for h in _SECRET_KEY_HINTS)


# Pattern-based scrubber pro secrety uvnitř string hodnot. Aplikuje se na
# args stringy (např. `run_bash.command` = `curl -H "Authorization: Bearer …"`)
# a na permission.summary/reason (mohou obsahovat raw command/URL prefix).
# Bez tohoto by secret v args.value nebo summary prošel přes pouhou truncation
# do audit logu. Phase 8 iter-5 fix (Codex).
#
# Pravidla jsou konzervativní (false-positive raději než leak):
#   1. `Authorization: ...` / `Bearer ...` HTTP header
#   2. `API_KEY=...`, `TOKEN=...`, `PASSWORD=...` env-var style
#   3. URL query parametry typu `?token=...`, `&api_key=...`
#   4. Long hex/base64-ish strings (32+ chars) co následují po klíčových
#      slovech (jwt eyJ…, sk-… apod.)
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # HTTP Authorization header (curl -H "Authorization: Bearer xxx")
    re.compile(r'(?i)\bAuthorization\s*[:=]\s*["\']?(?:Bearer|Basic|Token|JWT)?\s*[A-Za-z0-9._\-+/=]{8,}'),
    # Standalone Bearer / Basic auth
    re.compile(r'(?i)\b(?:Bearer|Basic)\s+[A-Za-z0-9._\-+/=]{16,}'),
    # ENV-var style: API_KEY=xxx, ANTHROPIC_API_KEY=sk-..., TOKEN=...
    re.compile(r'(?i)\b(?:[A-Z][A-Z0-9_]*_)?(?:API[_-]?KEY|ACCESS[_-]?TOKEN|SECRET[_-]?KEY|AUTH[_-]?TOKEN|TOKEN|PASSWORD|PASSWD|PASSPHRASE|SECRET)\s*[:=]\s*["\']?[A-Za-z0-9._\-+/=]{6,}'),
    # URL query params: ?token=…, &api_key=…
    re.compile(r'(?i)[?&](?:token|api[_-]?key|access[_-]?token|auth[_-]?token|key|password|secret|signature|sig|hmac)=[^&\s"\']{4,}'),
    # JWT (header.payload.sig) — eyJ… start signature
    re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}'),
    # Anthropic / OpenAI style API keys
    re.compile(r'\bsk-(?:ant-)?[A-Za-z0-9_\-]{20,}'),
    # AWS access key
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16,}\b'),
    # GitHub PAT
    re.compile(r'\bghp_[A-Za-z0-9]{30,}\b'),
)


def _scrub_secret_patterns(s: str) -> str:
    """Nahradí matched secret patterny v stringu placeholderem. Pure function."""
    if not isinstance(s, str) or len(s) < 8:
        return s
    out = s
    for pat in _SECRET_VALUE_PATTERNS:
        out = pat.sub("<redacted-secret>", out)
    return out


def _utcnow_iso() -> str:
    """ISO 8601 UTC se sufixem 'Z' (microsecond precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _today_filename(now_iso: str) -> str:
    return f"{now_iso[:10]}.jsonl"


def _redact_string(s: str, cap: int) -> str:
    if not isinstance(s, str):
        return s
    # Scrub secret patterns NEJDŘÍV — truncation by jinak mohla rozseknout
    # secret v polovině, kde by zbytek prošel jako "jen prefix" a první půlka
    # by se nestihla matchnout. Phase 8 iter-5 fix.
    scrubbed = _scrub_secret_patterns(s)
    if len(scrubbed) <= cap:
        return scrubbed
    return scrubbed[:cap] + "…[truncated]"


def _generic_truncate(value: Any, cap: int, depth: int = 0) -> Any:
    """Rekurzivně ořeže stringy + capne hloubku/velikost iterables.
    Bezpečné proti adversarial nested args."""
    if depth >= _REDACT_MAX_DEPTH:
        return "<depth-limit>"
    if isinstance(value, str):
        return _redact_string(value, cap)
    if isinstance(value, dict):
        items = list(value.items())
        out: dict[str, Any] = {}
        for i, (k, v) in enumerate(items):
            if i >= _REDACT_MAX_ITEMS:
                out["<truncated>"] = f"<{len(items) - _REDACT_MAX_ITEMS} more items>"
                break
            if _is_secret_key(k):
                out[k] = "<redacted>"
            else:
                out[k] = _generic_truncate(v, cap, depth + 1)
        return out
    if isinstance(value, list):
        out_l: list[Any] = []
        for i, v in enumerate(value):
            if i >= _REDACT_MAX_ITEMS:
                out_l.append(f"<{len(value) - _REDACT_MAX_ITEMS} more items>")
                break
            out_l.append(_generic_truncate(v, cap, depth + 1))
        return out_l
    return value


def _redact_content_field(value: Any) -> str:
    """Redaktuje content field bez ohledu na typ. Vrátí markerový string
    s typem + odhadem velikosti — žádný actual content. LLM může poslat
    invalid type (dict/list/int) místo str → bez tohoto by `_generic_truncate`
    nechal obsah projít až po truncation cap (500 chars), což může obsahovat
    secret. Phase 8 iter-3 fix (Codex)."""
    if isinstance(value, str):
        return f"<{len(value.encode('utf-8'))} bytes>"
    if value is None:
        return "<null>"
    if isinstance(value, (bool, int, float)):
        return f"<{type(value).__name__}>"
    try:
        size = len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        return f"<non-str {type(value).__name__} ~{size} bytes>"
    except (TypeError, ValueError):
        return f"<non-str {type(value).__name__}>"


def _redact_args(tool_name: str, args: Any) -> Any:
    """Per-tool redakce → fallback na generic truncation se secret-key scrubem.
    Vrátí copy bez mutace originálu."""
    cap = config.AUDIT_FIELD_CAP
    preview = config.AUDIT_ARG_PREVIEW

    if not isinstance(args, dict):
        return _generic_truncate(args, cap)

    if tool_name == "ask_claude":
        out: dict[str, Any] = {}
        for k, v in args.items():
            if _is_secret_key(k):
                out[k] = "<redacted>"
            elif k in ("prompt", "system") and isinstance(v, str):
                out[k] = _redact_string(v, preview)
            else:
                out[k] = _generic_truncate(v, cap)
        return out

    if tool_name in ("write_file", "edit_file"):
        out = {}
        for k, v in args.items():
            if _is_secret_key(k):
                out[k] = "<redacted>"
            elif k in ("content", "old_string", "new_string"):
                # write_file.content + edit_file.old_string/new_string mohou
                # obsahovat plný soubor / secret řádek (.env, API key). Audit
                # log MUSÍ být minimum-PII: nikdy nezapisujeme actual content.
                # Type-agnostic — pokud LLM pošle non-str (dict/list), stále
                # redaktujeme aby secret nemohl uniknout přes alternative path.
                out[k] = _redact_content_field(v)
            else:
                out[k] = _generic_truncate(v, cap)
        return out

    if tool_name == "fetch_url":
        out = {}
        for k, v in args.items():
            if _is_secret_key(k):
                out[k] = "<redacted>"
            elif k in ("body", "data"):
                out[k] = _redact_content_field(v)
            else:
                out[k] = _generic_truncate(v, cap)
        return out

    # Generic — secret-key scrub se aplikuje uvnitř _generic_truncate.
    return _generic_truncate(args, cap)


class AuditLog:
    """Append-only JSONL audit log. Async safe v rámci single procesu
    (asyncio.Lock + write v thread executoru). Cross-process safety
    spoléhá na POSIX O_APPEND atomicity pro malé write()y."""

    def __init__(self, audit_dir: Path | None) -> None:
        self.dir = audit_dir
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    def _ensure_dir_sync(self) -> None:
        """Sync verze pro volání z to_thread. Volá se uvnitř lock + thread.
        Bez cache — každý write znovu prochází symlink check, defense proti
        post-create swap (attacker `rm -rf dir && ln -s /etc dir`)."""
        if not self.dir:
            return
        d: Path = self.dir
        # Pokud existuje a je to symlink → odmítnout.
        if d.is_symlink():
            raise OSError(f"audit dir is a symlink: {d}")
        d.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _write_sync(self, line: str, basename: str) -> None:
        """Sync append do souboru. Atomic dir-fd handle (O_DIRECTORY+O_NOFOLLOW)
        kotvíme parent inode → soubor otvíráme relativně k tomuto fd, takže
        post-mkdir symlink swap parent dir nemůže přesměrovat write. O_NOFOLLOW
        na file open odmítne symlink na finální path. Existující soubor MUSÍ
        mít mode 0o600 a být vlastněný naším eUID — pokud ne (nebo fchmod selže)
        write se zamítne. Krátký write se loopuje do vyčerpání bufferu."""
        assert self.dir is not None  # caller zajišťuje enabled
        # Otevři parent dir s O_NOFOLLOW — symlink swap parent → fail.
        dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        dir_fd = os.open(self.dir, dir_flags)
        try:
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
            fd = os.open(basename, file_flags, 0o600, dir_fd=dir_fd)
            try:
                st = os.fstat(fd)
                # Owner check — odmítnout pre-vytvořený soubor jiného uživatele.
                # (Bez root by attacker nemohl vytvořit soubor s naším eUID,
                # ale defense in depth.)
                euid = os.geteuid()
                if st.st_uid != euid:
                    raise OSError(
                        f"audit file owner mismatch: uid={st.st_uid} expected={euid}"
                    )
                # Force 0o600. Selhání fchmod NEBO mode != 0o600 po fchmod
                # = refuse to write (fail-closed, ne fail-open).
                if (st.st_mode & 0o777) != 0o600:
                    os.fchmod(fd, 0o600)  # raises OSError → propagate
                    st2 = os.fstat(fd)
                    if (st2.st_mode & 0o777) != 0o600:
                        raise OSError(
                            f"audit file mode still {oct(st2.st_mode & 0o777)} after fchmod"
                        )
                data = (line + "\n").encode("utf-8")
                written = 0
                while written < len(data):
                    n = os.write(fd, data[written:])
                    if n <= 0:
                        raise OSError("os.write returned 0 — disk full?")
                    written += n
            finally:
                os.close(fd)
        finally:
            os.close(dir_fd)

    async def record(
        self,
        *,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        args: Any,
        permission: dict[str, Any],
        approval: str | None,
        ok: bool,
        error: str | None,
        result_bytes: int,
        duration_ms: int,
    ) -> None:
        """Zapíše jeden záznam. Pokud je log disabled, no-op. Při I/O chybě
        logguje warning a tichý fallback — audit nesmí zabít agent loop.
        Veškerá redakce + I/O je obalena try/except, aby buggy redactor
        neudusil caller."""
        if not self.enabled:
            return
        try:
            now = _utcnow_iso()
            record = {
                "ts": now,
                "turn_id": str(turn_id) if turn_id is not None else "",
                "tool_call_id": str(tool_call_id) if tool_call_id is not None else "",
                "tool": str(tool_name) if tool_name is not None else "",
                "args": _redact_args(tool_name or "", args),
                "permission": _truncate_permission(permission),
                "approval": approval,
                "ok": bool(ok),
                "error": _redact_string(error, config.AUDIT_FIELD_CAP) if error else None,
                "result_bytes": int(result_bytes),
                "duration_ms": int(duration_ms),
            }
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError, RecursionError) as e:
            log.warning("audit: failed to build/serialize record: %s", e)
            return

        basename = _today_filename(now)
        async with self._lock:
            try:
                await asyncio.to_thread(self._do_write, line, basename)
            except OSError as e:
                log.warning("audit: write failed (%s/%s): %s", self.dir, basename, e)

    def _do_write(self, line: str, basename: str) -> None:
        """Combined ensure_dir + write_sync, oba synchronně v thread executoru."""
        self._ensure_dir_sync()
        self._write_sync(line, basename)


def _truncate_permission(perm: Any) -> dict[str, Any]:
    """Permission má fixní pole; jen oříznem summary/reason které mohou být dlouhé."""
    if not isinstance(perm, dict):
        return {"raw": _generic_truncate(perm, config.AUDIT_FIELD_CAP)}
    cap = config.AUDIT_FIELD_CAP
    out: dict[str, Any] = {}
    for k, v in perm.items():
        if isinstance(v, str):
            out[k] = _redact_string(v, cap)
        else:
            out[k] = v
    return out
