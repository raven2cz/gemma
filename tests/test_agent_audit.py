"""Unit testy pro voice/agent/audit.py (fáze 8).

Pokrývá:
  - happy path zápis + JSONL re-parse
  - per-tool redaction (ask_claude prompt/system, write_file/edit_file content)
  - generic field cap
  - directory + file permissions (0o700 dir, 0o600 file)
  - per-day filename rotation
  - disabled audit (AUDIT_DIR=None → no-op)
  - concurrent writes (asyncio.gather × N)
  - non-serializable args fallback
  - error field truncation
  - I/O chyba → tichý fallback, nezhroucený caller
"""
from __future__ import annotations

import asyncio
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voice.agent import audit, config


@pytest.fixture
def tmp_audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test audit dir + reset module-level caches if any."""
    d = tmp_path / "audit"
    return d


def _make_log(d: Path | None) -> audit.AuditLog:
    return audit.AuditLog(d)


def _read_jsonl(p: Path) -> list[dict]:
    """Načte JSONL soubor řádek po řádku, vrátí list dictů."""
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Basic enabled/disabled behavior
# ---------------------------------------------------------------------------


def test_audit_disabled_when_dir_is_none():
    log = _make_log(None)
    assert log.enabled is False


def test_audit_enabled_when_dir_provided(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    assert log.enabled is True


@pytest.mark.asyncio
async def test_disabled_record_is_noop(tmp_audit_dir: Path):
    log = _make_log(None)
    await log.record(
        turn_id="t1", tool_call_id="tc1", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    # nic nemůžeme zkontrolovat na disku, ale nesmí crashnout
    assert not tmp_audit_dir.exists()


# ---------------------------------------------------------------------------
# Happy path + format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_writes_jsonl_and_creates_dir(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t-abc", tool_call_id="tc-1", tool_name="echo",
        args={"text": "hello"},
        permission={"decision": "auto", "risk": "low", "reason": "echo OK",
                    "requires_explicit": False, "summary": "echo: hello"},
        approval=None, ok=True, error=None, result_bytes=11, duration_ms=4,
    )
    assert tmp_audit_dir.exists()
    files = list(tmp_audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    records = _read_jsonl(files[0])
    assert len(records) == 1
    r = records[0]
    # Sanity checks
    assert r["turn_id"] == "t-abc"
    assert r["tool_call_id"] == "tc-1"
    assert r["tool"] == "echo"
    assert r["args"] == {"text": "hello"}
    assert r["permission"]["decision"] == "auto"
    assert r["approval"] is None
    assert r["ok"] is True
    assert r["error"] is None
    assert r["result_bytes"] == 11
    assert r["duration_ms"] == 4
    # ISO 8601 UTC, končí na Z
    ts = r["ts"]
    assert ts.endswith("Z")
    datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")


@pytest.mark.asyncio
async def test_filename_is_today_iso_date(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    files = list(tmp_audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert files[0].name == f"{today}.jsonl"


# ---------------------------------------------------------------------------
# File / directory permissions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dir_created_with_0o700(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    mode = stat.S_IMODE(tmp_audit_dir.stat().st_mode)
    # umask může 0o700 zmodifikovat na něco užšího, ale group/other bit musí být 0.
    assert mode & 0o077 == 0


@pytest.mark.asyncio
async def test_file_created_with_0o600(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    files = list(tmp_audit_dir.glob("*.jsonl"))
    mode = stat.S_IMODE(files[0].stat().st_mode)
    assert mode & 0o077 == 0


# ---------------------------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_writes_dont_corrupt(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    N = 50

    async def one(i: int) -> None:
        await log.record(
            turn_id=f"t-{i}", tool_call_id=f"tc-{i}", tool_name="echo",
            args={"i": i},
            permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
            approval=None, ok=True, error=None, result_bytes=0, duration_ms=i,
        )

    await asyncio.gather(*(one(i) for i in range(N)))
    files = list(tmp_audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    records = _read_jsonl(files[0])
    assert len(records) == N
    # Všechny IDs unikátní (žádný torn write / duplikace)
    seen_ids = {r["tool_call_id"] for r in records}
    assert seen_ids == {f"tc-{i}" for i in range(N)}


# ---------------------------------------------------------------------------
# Redaction — ask_claude
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_claude_prompt_truncated(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "AUDIT_ARG_PREVIEW", 50)
    log = _make_log(tmp_audit_dir)
    long_prompt = "X" * 5000
    long_system = "Y" * 5000
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="ask_claude",
        args={"prompt": long_prompt, "system": long_system, "max_tokens": 256},
        permission={"decision": "ask", "risk": "medium", "reason": "ok", "requires_explicit": False, "summary": "ask"},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=10,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["prompt"].startswith("X" * 50)
    assert r["args"]["prompt"].endswith("[truncated]")
    assert r["args"]["system"].startswith("Y" * 50)
    assert r["args"]["system"].endswith("[truncated]")
    assert r["args"]["max_tokens"] == 256


# ---------------------------------------------------------------------------
# Redaction — write_file / edit_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_file_content_replaced_with_byte_count(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    body = "abcdefghij" * 100  # 1000 chars / 1000 bytes (pure ASCII)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="write_file",
        args={"path": "foo.txt", "content": body},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["content"] == "<1000 bytes>"
    assert r["args"]["path"] == "foo.txt"


@pytest.mark.asyncio
async def test_edit_file_old_new_redacted_to_byte_count(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """edit_file používá `old_string`/`new_string` (ne `old`/`new`).
    Velký obsah se nahrazuje byte count."""
    monkeypatch.setattr(config, "AUDIT_ARG_PREVIEW", 20)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="edit_file",
        args={"path": "x.py", "old_string": "A" * 1000, "new_string": "B" * 1000},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    # Obsah souboru přesahující preview cap → nahrazen byte count.
    assert r["args"]["old_string"] == "<1000 bytes>"
    assert r["args"]["new_string"] == "<1000 bytes>"
    assert r["args"]["path"] == "x.py"


@pytest.mark.asyncio
async def test_edit_file_small_strings_also_byte_count(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """I malý obsah se nahrazuje byte count — secret typu `.env` řádek
    nebo API key se NIKDY nesmí logovat. Phase 8 iter-2 fix (Codex)."""
    monkeypatch.setattr(config, "AUDIT_ARG_PREVIEW", 200)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="edit_file",
        args={"path": "x.py", "old_string": "foo()", "new_string": "bar()"},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["old_string"] == "<5 bytes>"
    assert r["args"]["new_string"] == "<5 bytes>"
    assert r["args"]["path"] == "x.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("secret_carrier,marker", [
    # run_bash.command s curl Authorization header
    ({"tool": "run_bash", "args": {"command": 'curl -H "Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz0123456789" https://api.github.com'}},
     "ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    # fetch_url s tokenem v query
    ({"tool": "fetch_url", "args": {"url": "https://api.example.com/data?token=verylongsecret123456789xyz&user=foo"}},
     "verylongsecret123456789xyz"),
    # run_bash export API_KEY
    ({"tool": "run_bash", "args": {"command": 'export ANTHROPIC_API_KEY=sk-ant-1234567890abcdef1234567890; do_stuff'}},
     "sk-ant-1234567890abcdef1234567890"),
    # JWT v args
    ({"tool": "fetch_url", "args": {"url": "https://example.com", "headers": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"}},
     "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV"),
])
async def test_secret_value_patterns_scrubbed_in_args(tmp_audit_dir: Path, secret_carrier: dict, marker: str):
    """Phase 8 iter-5 fix (Codex): pattern-based scrubber pro secrety
    uvnitř string hodnot. Authorization headers, API keys, tokens, JWTs
    v args (run_bash.command, fetch_url.url) nesmí leaknout do auditu."""
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc",
        tool_name=secret_carrier["tool"], args=secret_carrier["args"],
        permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    serialized = json.dumps(r)
    assert marker not in serialized, f"secret leaked: {marker}"
    assert "<redacted-secret>" in serialized


@pytest.mark.asyncio
async def test_permission_summary_secrets_scrubbed(tmp_audit_dir: Path):
    """Permission summary obsahuje raw command/URL prefix z classifieru
    (run_bash, fetch_url). Secret v command-line nebo URL musí být scrubbed
    i tam, ne jen v args."""
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="run_bash",
        args={"command": "ls"},
        permission={
            "decision": "ask", "risk": "medium",
            "reason": "command 'curl' not in AUTO allowlist",
            "requires_explicit": False,
            "summary": '$ curl -H "Authorization: Bearer ghp_abcdefghijklmn1234567890opqrstuv"',
        },
        approval="denied", ok=False, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    summary = r["permission"]["summary"]
    assert "ghp_abcdefghijklmn1234567890opqrstuv" not in summary
    assert "<redacted-secret>" in summary


@pytest.mark.asyncio
async def test_edit_file_non_string_content_still_redacted(tmp_audit_dir: Path):
    """Phase 8 iter-3 fix (Codex): non-string `content` / `old_string` /
    `new_string` (dict/list) musí být redaktován bez ohledu na typ. Bez
    tohoto by LLM mohl poslat `{"content": {"x": "secret"}}` a secret
    by prošel přes _generic_truncate (cap=500)."""
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="write_file",
        args={"path": "x.txt", "content": {"hidden": "SUPERSECRETPASSWORD"}},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    content_repr = r["args"]["content"]
    assert "SUPERSECRETPASSWORD" not in content_repr
    assert "non-str" in content_repr or "bytes" in content_repr


@pytest.mark.asyncio
async def test_fetch_url_non_string_body_redacted(tmp_audit_dir: Path):
    """Stejný princip pro fetch_url body/data klíče."""
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="fetch_url",
        args={"url": "https://example.com", "body": ["leaked_token_xyz"]},
        permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    body_repr = r["args"]["body"]
    assert "leaked_token_xyz" not in body_repr


# ---------------------------------------------------------------------------
# Secret-key scrubber
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("key", [
    "api_key", "API_KEY", "apiKey", "x-api-key",
    "Authorization", "authorization",
    "bearer_token", "Bearer",
    "password", "Passwd",
    "secret", "client_secret",
    "session_id", "Cookie",
])
async def test_secret_key_scrubbed(key: str, tmp_audit_dir: Path):
    """Hodnoty pod podezřelými klíči se nahrazují '<redacted>' bez ohledu na tool."""
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="fetch_url",
        args={"url": "https://example.com/", "headers": {key: "sk-supersecret123"}},
        permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["headers"][key] == "<redacted>"


@pytest.mark.asyncio
async def test_secret_at_top_level_args_scrubbed(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="ask_claude",
        args={"prompt": "hi", "api_key": "sk-leak"},
        permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["api_key"] == "<redacted>"
    assert r["args"]["prompt"] == "hi"  # legitimate prompt zůstává


# ---------------------------------------------------------------------------
# DoS defenses: depth + list size
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deeply_nested_args_capped_by_depth(tmp_audit_dir: Path):
    """Hluboké nesting struktury (LLM mohl poslat malicious) se cappuje na _REDACT_MAX_DEPTH."""
    log = _make_log(tmp_audit_dir)
    # Build deeply nested dict
    deep: dict = {"x": "leaf"}
    for _ in range(100):
        deep = {"nested": deep}
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={"deep": deep},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    files = list(tmp_audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    # Hlavní kontrola: žádný RecursionError, zápis prošel.
    r = _read_jsonl(files[0])[0]
    # Někde na hloubce 7 musí být placeholder
    assert "depth-limit" in json.dumps(r["args"])


@pytest.mark.asyncio
async def test_huge_list_truncated(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={"items": list(range(10_000))},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    # List byl ořezán — poslední položka musí být placeholder
    items = r["args"]["items"]
    assert len(items) <= 51  # 50 items + 1 truncation marker
    assert "more items" in items[-1]


# ---------------------------------------------------------------------------
# Generic truncation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_string_truncation(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "AUDIT_FIELD_CAP", 30)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="run_bash",
        args={"command": "ls " + "Z" * 200},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    cmd = r["args"]["command"]
    assert len(cmd) <= 30 + len("…[truncated]")
    assert cmd.endswith("[truncated]")


@pytest.mark.asyncio
async def test_nested_dict_truncated_recursively(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "AUDIT_FIELD_CAP", 10)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="fetch_url",
        args={"url": "https://example.com/", "headers": {"x": "Y" * 500}},
        permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
        approval="approved", ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"]["headers"]["x"].endswith("[truncated]")


# ---------------------------------------------------------------------------
# Permission + approval + error fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permission_summary_and_reason_truncated(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "AUDIT_FIELD_CAP", 20)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={},
        permission={"decision": "auto", "risk": "low", "reason": "R" * 200, "requires_explicit": False, "summary": "S" * 200},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["permission"]["reason"].endswith("[truncated]")
    assert r["permission"]["summary"].endswith("[truncated]")


@pytest.mark.asyncio
async def test_approval_values_recorded(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    for approval in ("approved", "denied", None):
        await log.record(
            turn_id="t", tool_call_id=f"tc-{approval}", tool_name="echo",
            args={},
            permission={"decision": "ask", "risk": "medium", "reason": "", "requires_explicit": False, "summary": ""},
            approval=approval, ok=approval == "approved", error=None,
            result_bytes=0, duration_ms=1,
        )
    records = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))
    assert [r["approval"] for r in records] == ["approved", "denied", None]


@pytest.mark.asyncio
async def test_error_field_truncated(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "AUDIT_FIELD_CAP", 30)
    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=False, error="E" * 1000, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["error"].endswith("[truncated]")


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_dict_args_handled(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    # LLM mohl poslat string args (garbage) — nesmí crash
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args="bogus",  # type: ignore[arg-type]
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    r = _read_jsonl(next(tmp_audit_dir.glob("*.jsonl")))[0]
    assert r["args"] == "bogus"


@pytest.mark.asyncio
async def test_non_serializable_args_does_not_crash(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    class Weird:
        pass
    # default=str na json.dumps převede neserializable na repr
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={"obj": Weird()},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    files = list(tmp_audit_dir.glob("*.jsonl"))
    # Záznam buď byl zapsán (default=str fallback) nebo zahozen — v žádném
    # případě nesmí caller crashnout. Test: pokud soubor neexistuje, je to OK;
    # pokud existuje, musí být validní JSON.
    if files:
        records = _read_jsonl(files[0])
        assert len(records) == 1
        assert "obj" in records[0]["args"]


@pytest.mark.asyncio
async def test_io_error_is_swallowed(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Pokud zápis selže, audit log nesmí crashnout caller (agent loop musí pokračovat)."""
    log = _make_log(tmp_audit_dir)

    def boom(line, target):
        raise OSError("disk full")

    monkeypatch.setattr(log, "_write_sync", boom)
    # Nesmí raise — zachyceno uvnitř record()
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo",
        args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )


# ---------------------------------------------------------------------------
# Symlink hardening
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_symlink_audit_dir_rejected(tmp_path: Path):
    """Pokud je audit dir symlink, mkdir/write se odmítne (OSError uvnitř,
    log warning, agent loop nepadne)."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    log = _make_log(link)
    # Nesmí raise — error se zachytí uvnitř record().
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    # `real` dir nesmí dostat soubor (zápis byl rejected).
    assert not list(real.glob("*.jsonl"))


@pytest.mark.asyncio
async def test_symlink_audit_file_rejected(tmp_audit_dir: Path):
    """Pokud existuje preempted symlink na YYYY-MM-DD.jsonl mířící mimo,
    O_NOFOLLOW musí odmítnout."""
    tmp_audit_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = tmp_audit_dir.parent / "leaked.txt"
    target.write_text("original", encoding="utf-8")
    (tmp_audit_dir / f"{today}.jsonl").symlink_to(target)

    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    # Cílový soubor (mimo audit dir) NESMÍ obsahovat audit zápis.
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_existing_permissive_file_fchmod_to_0o600(tmp_audit_dir: Path):
    """Pokud existuje audit soubor s permissivnější maskou (např. 0o644),
    fchmod ho převede na 0o600 při dalším open."""
    tmp_audit_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = tmp_audit_dir / f"{today}.jsonl"
    existing.write_text("", encoding="utf-8")
    os.chmod(existing, 0o644)
    assert stat.S_IMODE(existing.stat().st_mode) == 0o644

    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    assert stat.S_IMODE(existing.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_fchmod_failure_refuses_write(tmp_audit_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Phase 8 iter-2 (Codex + Gemini): fchmod fail = fail-closed.
    Když je existující soubor 0o644 a fchmod selže (EPERM od jiného owner),
    write se nesmí provést. Předtím audit log silently zapisoval secret data
    do attacker-owned file. Simulujeme přes monkeypatch os.fchmod."""
    tmp_audit_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    existing = tmp_audit_dir / f"{today}.jsonl"
    existing.write_text("", encoding="utf-8")
    os.chmod(existing, 0o644)

    real_fchmod = os.fchmod
    def fail_fchmod(fd, mode):
        raise PermissionError("simulated EPERM (not owner)")
    monkeypatch.setattr(os, "fchmod", fail_fchmod)

    log = _make_log(tmp_audit_dir)
    await log.record(
        turn_id="t", tool_call_id="tc", tool_name="echo", args={"x": "secret"},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    # restore for cleanup (pytest fixtures might rely on real fchmod)
    monkeypatch.setattr(os, "fchmod", real_fchmod)
    # Write byl odmítnut → soubor zůstal prázdný (žádný JSON řádek).
    assert existing.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_parent_dir_symlink_swap_blocked(tmp_path: Path):
    """Phase 8 iter-2 (Gemini #4): post-mkdir parent dir swap musí blokovat
    write. Bez `_dir_ready` cache + dir_fd open by attacker mohl swapnout
    audit dir za symlink mezi prvním write-em a dalšími."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir(mode=0o700)
    # první write proběhne OK do reálného dir
    log = _make_log(audit_dir)
    await log.record(
        turn_id="t1", tool_call_id="tc1", tool_name="echo", args={},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    files_before = list(audit_dir.glob("*.jsonl"))
    assert len(files_before) == 1

    # Attacker swap: smaže dir a nahradí symlinkem na sensitive directory.
    import shutil
    shutil.rmtree(audit_dir)
    sensitive_target = tmp_path / "elsewhere"
    sensitive_target.mkdir()
    audit_dir.symlink_to(sensitive_target)

    # Druhý write musí selhat — symlink check znovu (no cache).
    await log.record(
        turn_id="t2", tool_call_id="tc2", tool_name="echo", args={"data": "secret"},
        permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
        approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
    )
    # Žádný soubor v sensitive_target — write zamítnut.
    assert list(sensitive_target.glob("*.jsonl")) == []


# ---------------------------------------------------------------------------
# Per-day rotation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_records_same_day_share_file(tmp_audit_dir: Path):
    log = _make_log(tmp_audit_dir)
    for i in range(2):
        await log.record(
            turn_id="t", tool_call_id=f"tc-{i}", tool_name="echo",
            args={},
            permission={"decision": "auto", "risk": "low", "reason": "", "requires_explicit": False, "summary": ""},
            approval=None, ok=True, error=None, result_bytes=0, duration_ms=1,
        )
    files = list(tmp_audit_dir.glob("*.jsonl"))
    assert len(files) == 1
    assert len(_read_jsonl(files[0])) == 2


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_config_resolve_audit_dir_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AGENT_AUDIT_DIR", raising=False)
    # Re-import resolver
    from voice.agent.config import _resolve_audit_dir
    d = _resolve_audit_dir()
    assert d is not None
    assert "agent-audit" in str(d)


@pytest.mark.parametrize("val", ["off", "OFF", "0", "false", "no", ""])
def test_config_resolve_audit_dir_disabled(val: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AGENT_AUDIT_DIR", val)
    from voice.agent.config import _resolve_audit_dir
    assert _resolve_audit_dir() is None


def test_config_resolve_audit_dir_custom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("AGENT_AUDIT_DIR", str(tmp_path / "custom"))
    from voice.agent.config import _resolve_audit_dir
    d = _resolve_audit_dir()
    assert d == tmp_path / "custom"
