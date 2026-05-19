"""TmuxAdapter - drive `claude` interactive v tmux session (experimental).

Detection markers (ověřeno z claude-code source 2.1.141):
- `❯` (U+276F figures.pointer): user input prompt indicator
- `●` (U+25CF) na Linux / `⏺` (U+23FA) na macOS: BLACK_CIRCLE, both
  for tool_use markers AND assistant message prefix (`● <response>`)
- `✻ <Verb> for Ns`: turn completion footer (claude-code/src/constants/
  turnCompletionVerbs.ts). Verbs: Baked|Brewed|Churned|Cogitated|Cooked|
  Crunched|Sautéed|Worked. Detection: regex `✻ \w+ for \d+\w*s`.

⚠️ TOS Risk Disclaimer:
Anthropic Consumer Terms zakazují "automated/non-human access" mimo API key.
Tmux pseudo-TTY driver je technicky interactive (stdout je TTY), ale
Anthropic má diskreci klasifikovat jako Agent SDK use. Plán neslibuje žádný
billing benefit - viz docs/plans/claude_tmux_adapter.md TOS disclaimer.

Architektura:
- Per-session `_TmuxSession` instance s asyncio.Lock + state machine
- States: READY (waiting for ask) → RUNNING (active turn) → READY |
  CANCELING (cancel_event signaled) → READY (with canceled=True) |
  DEAD (timeout / external kill / crash, TERMINAL state - no recreate)
- Persistent metadata v `.gemma_local/claude_sessions.json` s integrity check
  (phrase_hash). Reattach valid only s matching metadata; orphan = kill.
- Permission mode immutable per Claude process (codex iter-2 #8) -
  consult↔edit toggle vyžaduje kill + new session.

Tmux primitives:
- new-session -d -s claude_<hash> -x cols -y rows -c <workdir> <argv>
- send-keys -t <id> -l "<prompt>"
- send-keys -t <id> Enter
- capture-pane -p -e -S - -t <id>   (= entire scrollback s ANSI)
- has-session -t <id>
- kill-session -t <id>
- set-option -t <id> history-limit 100000

Parser: pyte terminal emulator (parsing/tui_state.TuiState), incremental
transcript collector (žádný 100-line truncation).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import (
    AbstractClaudeAdapter,
    AdapterCapabilities,
    ProgressCallback,
    SessionState,
)
from ..config import AdapterConfig
from ..exceptions import (
    AdapterConfigError,
    SessionBusy,
    SessionDead,
    SessionNotFound,
    SubprocessError,
)
from ..parsing.ansi import strip_ansi
from ..parsing.tui_state import TuiState
from ..progress import ProgressEvent, SessionInfo
from ..result import ClaudeResult, Mode

log = logging.getLogger("claude_bridge.tmux_mode")


# Mode=edit tools allowlist - same as print_mode.
_EDIT_MODE_TOOLS = "Read,Edit,Write,Bash,Glob,Grep"

# Poll interval pro capture-pane během RUNNING state.
_POLL_INTERVAL_SEC = 0.25

# Kolik idle iterací bez screen change = considered "done waiting".
# Při _POLL_INTERVAL_SEC=0.25, 3 iterace = 0.75s idle = done.
_IDLE_THRESHOLD = 3

# Default tmux session window size. Velký aby long lines nepřepadly.
_TMUX_COLS = 200
_TMUX_ROWS = 50

# Turn completion footer detection - z claude-code/src/constants/turnCompletionVerbs.ts.
# Match: `✻ <Verb> for <duration>`, např. `✻ Churned for 6s`, `✻ Cooked for 1m 32s`.
_TURN_COMPLETION_RE = re.compile(
    r"✻\s+(?:Baked|Brewed|Churned|Cogitated|Cooked|Crunched|Saut[eé]ed|Worked)"
    r"\s+for\s+\d+"
)


@dataclass
class _TmuxSession:
    """In-memory state pro jednu tmux session.

    Lifecycle:
        READY → RUNNING → READY (success or canceled)
                       → DEAD (timeout/kill/crash) [TERMINAL]

    DEAD je terminal - žádné silent recreate (codex iter-2 #13).
    """
    session_id: str
    workdir: Path | None
    model: str
    permission_mode: Mode    # immutable per Claude process (codex iter-2 #8)
    created_at: float
    last_active: float
    destructive_approved: bool = False
    turn_count: int = 0
    state: SessionState = "READY"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    transcript_buffer: list[str] = field(default_factory=list)  # accumulated capture
    last_capture: str = ""   # pro diff append
    tui: TuiState | None = None
    # Cancellation request flag - set v ask() když cancel_event signaled.
    _cancel_requested: bool = False


class TmuxAdapter:
    """AbstractClaudeAdapter implementace co drive-uje `claude` v tmux session.

    Long-lived sessions per session_id (typicky 1:1 s user Claude konverzací).
    Session_id je adapter-allocated unless caller passne explicit (= continue
    existing).
    """

    name = "tmux"

    def __init__(self, config: AdapterConfig) -> None:
        if shutil.which(config.tmux_bin) is None:
            raise AdapterConfigError(
                f"tmux binary {config.tmux_bin!r} not in PATH"
            )
        try:
            import pyte  # noqa: F401
        except ImportError as e:
            raise AdapterConfigError(
                f"pyte required for TmuxAdapter: {e}"
            ) from e

        self.config = config
        self.capabilities = AdapterCapabilities.tmux_mode()
        self._sessions: dict[str, _TmuxSession] = {}
        self._metadata_path: Path | None = None
        if config.metadata_dir:
            self._metadata_path = Path(config.metadata_dir) / "claude_sessions.json"
        # Dedicated tmux socket - izolace od user's default tmux server (codex
        # iter-3 CRITICAL #1). Bez `-L` by jsme sdíleli env tmux serveru +
        # mohli zabít user's sessions. Socket name odvozený od session prefix.
        self._tmux_socket_name = f"claude_bridge_{config.tmux_session_prefix.rstrip('_')}"

    def _build_scrubbed_env(self) -> dict[str, str]:
        """Filtruj parent env na allowlist + LC_*. Aplikováno na tmux subprocess
        i na spawned claude (přes `new-session -e`)."""
        allow = frozenset(self.config.env_allowlist)
        out: dict[str, str] = {}
        for k, v in os.environ.items():
            if k in allow or k.startswith("LC_"):
                out[k] = v
        return out

    # ──────────────── tmux subprocess primitives ────────────────

    async def _tmux(self, *args: str, capture: bool = True,
                    timeout: float = 5.0) -> tuple[int, bytes, bytes]:
        """Spustí tmux <args> na našem dedicated socket (`-L <socket>`).

        Izolace od user's tmux server (codex iter-3 CRITICAL #1). Tmux subprocess
        + spawned claude běží v scrubbed env (allowlist filter).
        """
        env = self._build_scrubbed_env()
        proc = await asyncio.create_subprocess_exec(
            self.config.tmux_bin, "-L", self._tmux_socket_name, *args,
            stdout=asyncio.subprocess.PIPE if capture else None,
            stderr=asyncio.subprocess.PIPE if capture else None,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, b"", b"timeout"
        return proc.returncode or 0, stdout or b"", stderr or b""

    async def _has_session(self, session_id: str) -> bool:
        """tmux has-session -t <id> → True pokud session existuje."""
        rc, _, _ = await self._tmux("has-session", "-t", session_id)
        return rc == 0

    async def _capture_pane(self, session_id: str) -> str:
        """tmux capture-pane -p -e -S - -t <id> → entire scrollback s ANSI.

        `-S -` = start from beginning of history (pro získání full transcript).
        `-e` = include escape sequences (pyte je interpretuje).
        Pokud session neexistuje, vrátí prázdný string.
        """
        rc, stdout, _ = await self._tmux(
            "capture-pane", "-p", "-e", "-S", "-", "-t", session_id,
        )
        if rc != 0:
            return ""
        return stdout.decode("utf-8", errors="replace")

    async def _send_keys(self, session_id: str, text: str, *, literal: bool = True) -> bool:
        """tmux send-keys -t <id> -l "<text>" → True pokud OK.

        `literal=True` (default) použije `-l` flag aby tmux nezpracovával
        special sequences (Enter, Ctrl-C, atd.). Použij `literal=False` pro
        explicit key names (Enter, Escape, C-c).

        SECURITY (sonnet review H2): newline v literal text mode by způsobil
        accidental Enter (= predčasný submit v Claude TUI). Replace `\\r\\n`
        a `\\n` na single space PŘED send. Multi-line prompty MUSÍ caller
        explicitly použít separate send + literal=False C-j (Shift+Enter).
        """
        if literal:
            # Strip CRLF/LF aby se neproste interpretovaly jako Enter
            text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        args = ["send-keys", "-t", session_id]
        if literal:
            args.append("-l")
        args.append(text)
        rc, _, _ = await self._tmux(*args)
        return rc == 0

    async def _send_enter(self, session_id: str) -> bool:
        """Send Enter key (submit Claude input)."""
        rc, _, _ = await self._tmux("send-keys", "-t", session_id, "Enter")
        return rc == 0

    async def _send_escape_twice(self, session_id: str) -> bool:
        """Send Esc Esc - Claude TUI cancel current operation."""
        rc, _, _ = await self._tmux(
            "send-keys", "-t", session_id, "Escape", "Escape",
        )
        return rc == 0

    async def _kill_tmux_session(self, session_id: str) -> bool:
        """tmux kill-session -t <id> → True pokud killed."""
        rc, _, _ = await self._tmux("kill-session", "-t", session_id)
        return rc == 0

    async def _spawn_tmux_session(
        self,
        session_id: str,
        argv: list[str],
        cwd: Path,
    ) -> bool:
        """tmux new-session -d -s <id> -x cols -y rows -c <cwd> <argv...>

        Returns True pokud spawn succeeded. Po spawn nastaví history-limit
        pro full scrollback retention (codex iter-2 #11).
        """
        rc, _, stderr = await self._tmux(
            "new-session", "-d", "-s", session_id,
            "-x", str(_TMUX_COLS), "-y", str(_TMUX_ROWS),
            "-c", str(cwd),
            *argv,
        )
        if rc != 0:
            log.warning("tmux new-session failed: %s", stderr.decode(errors="replace"))
            return False
        # Set large history limit pro incremental transcript collection
        await self._tmux(
            "set-option", "-t", session_id, "history-limit",
            str(self.config.tmux_history_limit),
        )
        return True

    # ──────────────── Session lifecycle ────────────────

    def _build_claude_argv(
        self,
        *,
        model: str,
        mode: Mode,
        workdir: Path | None,
        system: str | None,
    ) -> list[str]:
        """Sestaví argv pro `claude` (BEZ `-p` flag = interactive mode).

        Stejné permission/tool semantiky jako print mode:
        - consult: --permission-mode plan --tools ""
        - edit:    --permission-mode acceptEdits --tools <R/E/W/B/G/G> --add-dir <wd>
        """
        # POZN: `--no-session-persistence` je valid JEN s `--print` (= -p) mode.
        # V interactive mode (= tmux) musí pryč, jinak claude exit=1 hned po
        # spawnu s "Error: --no-session-persistence can only be used with --print".
        argv = [
            self.config.claude_bin,
            "--model", model,
        ]
        if mode == "consult":
            argv += [
                "--permission-mode", "plan",
                "--tools", "",
            ]
        else:  # edit
            if workdir is None:
                raise ValueError("mode=edit requires workdir")
            argv += [
                "--permission-mode", "acceptEdits",
                "--tools", _EDIT_MODE_TOOLS,
                "--add-dir", str(workdir),
            ]
        if system:
            argv += ["--append-system-prompt", system]
        return argv

    def _generate_session_id(self) -> str:
        """Generate unique session_id with `claude_` prefix."""
        return f"{self.config.tmux_session_prefix}{secrets.token_hex(8)}"

    async def _new_session(
        self,
        *,
        model: str,
        mode: Mode,
        workdir: Path | None,
        system: str | None,
    ) -> _TmuxSession:
        """Spawn new tmux + claude session.

        Raises SubprocessError pokud spawn failed.
        """
        session_id = self._generate_session_id()
        argv = self._build_claude_argv(
            model=model, mode=mode, workdir=workdir, system=system,
        )
        cwd = workdir if mode == "edit" and workdir else Path(tempfile.gettempdir())
        ok = await self._spawn_tmux_session(session_id, argv, cwd)
        if not ok:
            raise SubprocessError(f"failed to spawn tmux session {session_id}")
        now = time.time()
        session = _TmuxSession(
            session_id=session_id,
            workdir=workdir,
            model=model,
            permission_mode=mode,
            created_at=now,
            last_active=now,
            tui=TuiState(cols=_TMUX_COLS, rows=_TMUX_ROWS),
        )
        self._sessions[session_id] = session

        # Wait for claude TUI ready (initial prompt rendering)
        await self._wait_for_ready(session, timeout_sec=20.0)
        await self._persist_metadata()
        return session

    async def _wait_for_ready(self, session: _TmuxSession, *, timeout_sec: float) -> None:
        """Block until Claude TUI shows ready prompt (or timeout)."""
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            raw = await self._capture_pane(session.session_id)
            if session.tui is not None:
                session.tui.feed(raw)
                if session.tui.is_ready():
                    return
            await asyncio.sleep(_POLL_INTERVAL_SEC)
        log.warning("session %s did not reach ready state in %.1fs",
                    session.session_id, timeout_sec)

    # ──────────────── Persistent metadata (codex iter-2 #9) ────────────────

    def _compute_phrase_hash(self, session_id: str, approval_phrase: str = "") -> str:
        """SHA256 přes (session_id + phrase + secret). Caller předá phrase
        když user explicit approved; jinak prázdný hash.

        Secret loadne z `.gemma_local/secret` nebo vygenerujeme.
        """
        secret = self._load_or_create_secret()
        data = f"{session_id}\x00{approval_phrase}\x00{secret}".encode()
        return "sha256:" + hashlib.sha256(data).hexdigest()

    def _load_or_create_secret(self) -> str:
        """Per-instance secret pro phrase_hash. Pokud `.gemma_local/secret`
        existuje, načti; jinak vygeneruj a ulož s mode 0600."""
        if not self._metadata_path:
            return "no_metadata_dir"  # adapter without persistence
        secret_path = self._metadata_path.parent / "secret"
        try:
            return secret_path.read_text().strip()
        except FileNotFoundError:
            secret_path.parent.mkdir(parents=True, exist_ok=True)
            secret = secrets.token_hex(32)
            secret_path.write_text(secret)
            secret_path.chmod(0o600)
            return secret

    async def _persist_metadata(self) -> None:
        """Atomic write všech sessions do metadata file (tmpfile + rename)."""
        if not self._metadata_path:
            return
        data = {
            "version": "v1",
            "sessions": {
                sid: {
                    "session_id": sid,
                    "owner": "gemma",
                    "workdir": str(s.workdir) if s.workdir else None,
                    "model": s.model,
                    "permission_mode": s.permission_mode,
                    "created_at": s.created_at,
                    "last_active": s.last_active,
                    "approval": (
                        {
                            "approved_at": s.last_active,
                            "phrase_hash": self._compute_phrase_hash(sid, "ano povoluju"),
                            "approval_version": "v1",
                        } if s.destructive_approved else None
                    ),
                    "turn_count": s.turn_count,
                    "state": s.state,
                }
                for sid, s in self._sessions.items()
            },
        }
        try:
            self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._metadata_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(self._metadata_path)
        except Exception as e:
            log.warning("failed to persist metadata: %s", e)

    async def reattach_persisted_sessions(self) -> None:
        """Při startu načti persisted sessions, ověř integrity, registruj.

        Per codex iter-2 #9 + iter-3 #5:
        - Metadata corrupt/malformed → fail-closed: kill ALL adapter sessions
          v tmux (= unsafe state, nevíme co je trustworthy)
        - tmux has-session neexistuje → smazat z metadata, skip
        - Integrity FAIL (phrase_hash mismatch) → kill tmux + smazat metadata
        - Tmux session bez metadata (orphan) → kill + ignore

        Schema: top-level {"version": "v1", "sessions": {sid: {...}}}
        Session: {session_id, owner, workdir, model, permission_mode,
                  created_at, last_active, approval?, turn_count, state}
        """
        if not self._metadata_path or not self._metadata_path.exists():
            return

        # Fail-closed parse
        try:
            data = json.loads(self._metadata_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("metadata file corrupt - kill all adapter sessions: %s", e)
            await self._kill_all_adapter_sessions()
            return

        # Schema validation
        if not isinstance(data, dict):
            log.warning("metadata not a dict - fail-closed")
            await self._kill_all_adapter_sessions()
            return
        if data.get("version") != "v1":
            log.warning("metadata version unsupported: %r - fail-closed",
                        data.get("version"))
            await self._kill_all_adapter_sessions()
            return
        sessions_dict = data.get("sessions")
        if not isinstance(sessions_dict, dict):
            log.warning("metadata.sessions not a dict - fail-closed")
            await self._kill_all_adapter_sessions()
            return

        valid_ids: set[str] = set()
        for sid, meta in sessions_dict.items():
            if not isinstance(sid, str) or not sid.startswith(
                    self.config.tmux_session_prefix):
                continue  # invalid sid
            if not isinstance(meta, dict):
                log.warning("metadata for %s not a dict - skip", sid)
                continue
            if not await self._has_session(sid):
                continue  # session zmizela, skip
            permission_mode = meta.get("permission_mode")
            if permission_mode not in ("consult", "edit"):
                log.warning("invalid permission_mode for %s - kill", sid)
                await self._kill_tmux_session(sid)
                continue
            if permission_mode == "edit":
                approval = meta.get("approval") or {}
                if not isinstance(approval, dict):
                    log.warning("approval not dict for %s - kill", sid)
                    await self._kill_tmux_session(sid)
                    continue
                expected_hash = self._compute_phrase_hash(sid, "ano povoluju")
                if approval.get("phrase_hash") != expected_hash:
                    log.warning("integrity check failed for %s - killing orphan", sid)
                    await self._kill_tmux_session(sid)
                    continue
            # Valid - register
            wd = meta.get("workdir")
            self._sessions[sid] = _TmuxSession(
                session_id=sid,
                workdir=Path(wd) if wd else None,
                model=meta.get("model", "claude-opus-4-7"),
                permission_mode=permission_mode or "consult",
                created_at=meta.get("created_at", time.time()),
                last_active=meta.get("last_active", time.time()),
                destructive_approved=bool(meta.get("approval")),
                turn_count=meta.get("turn_count", 0),
                state="READY",
                tui=TuiState(cols=_TMUX_COLS, rows=_TMUX_ROWS),
            )
            valid_ids.add(sid)

        # Kill orphan tmux sessions (claude_* without metadata entry).
        # POZN: tmux operuje na našem dedicated socket (-L), takže list-sessions
        # vrátí JEN naše sessions, nikdy user's default tmux server.
        rc, stdout, _ = await self._tmux("list-sessions", "-F", "#{session_name}")
        if rc == 0:
            for line in stdout.decode(errors="replace").splitlines():
                sid = line.strip()
                if (sid.startswith(self.config.tmux_session_prefix)
                        and sid not in valid_ids):
                    log.warning("killing unsafe orphan session: %s", sid)
                    await self._kill_tmux_session(sid)

    async def _kill_all_adapter_sessions(self) -> None:
        """Fail-closed cleanup: zabít všechny sessions s naším prefix
        na našem dedicated socket. Použito pokud metadata file je corrupt
        a nemůžeme rozhodnout které sessions jsou trustworthy."""
        rc, stdout, _ = await self._tmux("list-sessions", "-F", "#{session_name}")
        if rc != 0:
            return
        for line in stdout.decode(errors="replace").splitlines():
            sid = line.strip()
            if sid.startswith(self.config.tmux_session_prefix):
                log.warning("fail-closed cleanup: killing %s", sid)
                await self._kill_tmux_session(sid)
        self._sessions.clear()

    # ──────────────── ask() implementation ────────────────

    async def ask(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str,
        mode: Mode,
        workdir: Path | None,
        session_id: str | None = None,
        timeout_sec: float = 600.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ClaudeResult:
        """Pošli prompt do existing nebo new tmux session, počkej na response."""
        if not prompt or not prompt.strip():
            return ClaudeResult(
                ok=False, mode=mode, error="empty prompt", adapter="tmux",
            )

        # Get or create session
        session: _TmuxSession
        if session_id is not None and session_id in self._sessions:
            session = self._sessions[session_id]
            # Permission mode mismatch? (codex iter-2 #8) - immutable invariant
            if session.permission_mode != mode:
                raise AdapterConfigError(
                    f"session {session_id} has permission_mode={session.permission_mode!r}, "
                    f"but ask() requested mode={mode!r}. Permission mode is immutable per "
                    f"Claude process - kill session + start new for toggle."
                )
            # Health check
            health = await self.health_check(session_id)
            if health == "DEAD":
                raise SessionDead(
                    f"session {session_id} is DEAD (terminal state). "
                    f"Start a fresh session - no silent recreate."
                )
        elif session_id is not None:
            raise SessionNotFound(f"session {session_id!r} not found")
        else:
            session = await self._new_session(
                model=model, mode=mode, workdir=workdir, system=system,
            )

        # Per-session mutex (codex iter-1 #4)
        if session.lock.locked():
            raise SessionBusy(
                f"session {session.session_id} is RUNNING; "
                f"concurrent ask() not supported"
            )

        async with session.lock:
            return await self._do_ask(
                session=session,
                prompt=prompt,
                timeout_sec=timeout_sec,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )

    async def _do_ask(
        self,
        *,
        session: _TmuxSession,
        prompt: str,
        timeout_sec: float,
        cancel_event: asyncio.Event | None,
        progress_callback: ProgressCallback | None,
    ) -> ClaudeResult:
        """Inside-mutex implementation: send prompt, poll capture, extract response."""
        session.state = "RUNNING"
        session._cancel_requested = False
        start = time.monotonic()

        # Marker pro response slicing - kde byl prompt v transcript
        marker = f"__GEMMA_TURN_{session.turn_count}__"
        prompt_with_marker = f"{marker}\n{prompt}"

        # Capture pre-prompt screen state pro before/after diff
        before_capture = await self._capture_pane(session.session_id)

        # Send prompt + Enter
        if not await self._send_keys(session.session_id, prompt_with_marker, literal=True):
            session.state = "DEAD"
            await self._persist_metadata()
            return ClaudeResult(
                ok=False, mode=session.permission_mode,
                error="tmux send-keys failed", adapter="tmux",
                session_id=session.session_id,
            )
        # Krátká pauza po send-keys (claude TUI buffer)
        await asyncio.sleep(0.1)
        if not await self._send_enter(session.session_id):
            session.state = "DEAD"
            await self._persist_metadata()
            return ClaudeResult(
                ok=False, mode=session.permission_mode,
                error="tmux Enter send failed", adapter="tmux",
                session_id=session.session_id,
            )

        # Emit "started" progress
        await self._emit_progress(progress_callback, ProgressEvent(
            stage="started",
            message=f"Claude {session.model} session {session.session_id[:12]}",
            session_id=session.session_id,
            model=session.model,
        ))

        # Polling loop: capture + parse + emit + check done
        # Done detection: turn completion footer `✻ <Verb> for Ns`.
        # POZOR: scrollback v reused session může obsahovat starý completion
        # marker z předchozího turnu. Proto musíme count baseline PŘED send
        # a wait pokud count INCREASED (nový marker = aktuální turn done).
        deadline = start + timeout_sec
        thinking_emitted = False
        # Count `●`/`⏺` markers PRED send (baseline)
        # POZN: raw capture obsahuje ANSI codes mezi znaky → strip_ansi PŘED count.
        pre_capture_clean = strip_ansi(before_capture)
        pre_marker_count = pre_capture_clean.count("● ") + pre_capture_clean.count("⏺ ")
        # Count completion footers PRED send (baseline pro race fix)
        pre_completion_count = len(_TURN_COMPLETION_RE.findall(pre_capture_clean))
        while time.monotonic() < deadline:
            # Cancel check (codex iter-3 HIGH #4: confirm Claude actually stopped)
            if cancel_event is not None and cancel_event.is_set():
                session._cancel_requested = True
                session.state = "CANCELING"
                await self._send_escape_twice(session.session_id)
                # Wait pro confirm: Claude must show idle prompt nebo new ✻ marker
                # do 5s. Jinak označíme session DEAD (Claude ignoroval Esc).
                cancel_deadline = time.monotonic() + 5.0
                cancel_confirmed = False
                while time.monotonic() < cancel_deadline:
                    await asyncio.sleep(0.25)
                    raw = await self._capture_pane(session.session_id)
                    if not raw:
                        continue
                    raw_clean = strip_ansi(raw)
                    # Buď spinner zmizel + ready prompt, NEBO completion footer
                    if session.tui is not None:
                        session.tui.feed(raw)
                    new_completion = len(_TURN_COMPLETION_RE.findall(raw_clean))
                    if new_completion > pre_completion_count:
                        cancel_confirmed = True
                        break
                    if session.tui is not None and session.tui.is_ready() \
                            and not session.tui.is_thinking():
                        cancel_confirmed = True
                        break
                duration_ms = int((time.monotonic() - start) * 1000)
                if cancel_confirmed:
                    session.state = "READY"
                    return ClaudeResult(
                        ok=False, mode=session.permission_mode,
                        error="canceled", canceled=True, adapter="tmux",
                        session_id=session.session_id, duration_ms=duration_ms,
                    )
                else:
                    # Claude ignoroval Esc → session DEAD (terminal, no retry).
                    session.state = "DEAD"
                    await self._persist_metadata()
                    return ClaudeResult(
                        ok=False, mode=session.permission_mode,
                        error="cancel timeout - Claude ignored Esc, session DEAD",
                        canceled=True, adapter="tmux",
                        session_id=session.session_id, duration_ms=duration_ms,
                    )

            await asyncio.sleep(_POLL_INTERVAL_SEC)
            raw = await self._capture_pane(session.session_id)
            if not raw:
                continue
            session.last_capture = raw
            if session.tui is None:
                session.tui = TuiState(cols=_TMUX_COLS, rows=_TMUX_ROWS)
            session.tui.feed(raw)

            # Emit progress for new tool_uses. POZOR: `● <text>` v TUI je
            # *response prefix* + tool_use markers (oba pre `●`). Tool_use
            # follow je `Read 1 file (ctrl+o to expand)` etc - specific tool
            # names z claude builtins. Filter podle known tool names list,
            # skip pokud "tool_name" je něco co Claude do response napsal.
            _KNOWN_TOOLS = {"Read", "Write", "Edit", "Bash", "Glob", "Grep",
                            "NotebookEdit", "Task", "TodoWrite", "WebFetch",
                            "WebSearch", "BashOutput", "KillShell"}
            for tu in session.tui.poll_tool_uses():
                if tu.tool_name in _KNOWN_TOOLS:
                    msg = (f"{tu.tool_name} {tu.args_preview}"
                           if tu.args_preview else tu.tool_name)
                    await self._emit_progress(progress_callback, ProgressEvent(
                        stage="tool_use",
                        tool_name=tu.tool_name,
                        message=msg,
                    ))

            # Emit thinking progress (once per turn)
            if session.tui.is_thinking() and not thinking_emitted:
                await self._emit_progress(progress_callback, ProgressEvent(
                    stage="thinking", message="přemýšlí…",
                ))
                thinking_emitted = True

            # Done detection: turn completion footer `✻ <Verb> for Ns`.
            # MUSÍ být new marker (count > pre_completion_count) aby jsme
            # nezachytili starý marker z reused session scrollback.
            # Strip ANSI před regex (raw obsahuje color codes mezi znaky).
            # Codex iter-3 HIGH #3: ●-count fallback ODSTRANĚN - byl unreliable
            # (● je response start marker, ne completion). Jediný spolehlivý
            # signál je `✻ <Verb> for Ns` footer (post-completion).
            raw_clean = strip_ansi(raw)
            current_completion_count = len(_TURN_COMPLETION_RE.findall(raw_clean))
            if current_completion_count > pre_completion_count:
                # Found new done marker. Krátká dodatečná pauza pro complete render.
                await asyncio.sleep(0.3)
                # Refresh capture
                raw_final = await self._capture_pane(session.session_id)
                if raw_final:
                    session.last_capture = raw_final
                break
        else:
            # timeout - session DEAD (codex iter-2 #13, no respawn)
            session.state = "DEAD"
            await self._persist_metadata()
            duration_ms = int((time.monotonic() - start) * 1000)
            return ClaudeResult(
                ok=False, mode=session.permission_mode,
                error=f"timeout after {timeout_sec:.0f}s - session DEAD, start fresh",
                timeout=True, adapter="tmux",
                session_id=session.session_id, duration_ms=duration_ms,
            )

        # Extract response text from transcript buffer
        text = self._extract_response_text(session, before_capture, marker)

        session.state = "READY"
        session.last_active = time.time()
        session.turn_count += 1
        await self._persist_metadata()

        duration_ms = int((time.monotonic() - start) * 1000)
        return ClaudeResult(
            ok=True,
            mode=session.permission_mode,
            text=text,
            model=session.model,
            session_id=session.session_id,
            duration_ms=duration_ms,
            tool_uses=tuple(
                tu.tool_name for tu in (session.tui.poll_tool_uses() if session.tui else [])
            ),
            adapter="tmux",
        )

    def _extract_response_text(
        self,
        session: _TmuxSession,
        before_capture: str,
        marker: str,
    ) -> str:
        """Extract assistant text z capture po našem prompt markeru.

        Reálný Claude TUI layout po user message:
            ❯ <user prompt s marker>
            (prázdná řádka)
            ● <Claude's response text>
            (prázdná řádka)
            ✻ Crunched for Ns
            ─────────────
            ❯  (empty next input)

        Strategy:
        1. Najdi marker v capture (= náš user prompt řádek)
        2. Skip past user prompt block - hledej PRVNÍ `● ` po marker
        3. Collect text od `● ` až do `✻ Churned`, `─────`, nebo `❯ ` empty
        4. Strip leading `● ` prefix, ANSI, trailing whitespace.
        """
        after = session.last_capture or ""
        idx = after.find(marker)
        if idx == -1:
            # Fallback: diff before/after
            new_content = after[len(before_capture):] if len(after) > len(before_capture) else after
            return strip_ansi(new_content).strip()
        text_after_marker = after[idx + len(marker):]
        clean = strip_ansi(text_after_marker)
        lines = clean.split("\n")

        # Find assistant response start: first line beginning with `● ` or `⏺ `
        response_start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("● ") or stripped.startswith("⏺ "):
                response_start_idx = i
                break
        if response_start_idx == -1:
            # No `● ` marker found - return what's between marker and next ❯
            result_lines: list[str] = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith("❯ ") or stripped == "❯":
                    break
                result_lines.append(line)
            return "\n".join(result_lines).strip()

        # Collect response lines until terminator
        result_lines = []
        first = True
        for line in lines[response_start_idx:]:
            stripped = line.lstrip()
            # Terminators: ✻ Churned/Crunched, ────, next ❯
            if stripped.startswith("✻ ") \
                    or stripped.startswith("─") \
                    or stripped.startswith("❯ ") or stripped == "❯":
                break
            if first:
                # Strip leading `● ` or `⏺ ` from first line
                if stripped.startswith("● "):
                    stripped = stripped[2:]
                elif stripped.startswith("⏺ "):
                    stripped = stripped[2:]
                result_lines.append(stripped)
                first = False
            else:
                result_lines.append(line)
        return "\n".join(result_lines).strip()

    async def _emit_progress(
        self,
        cb: ProgressCallback | None,
        event: ProgressEvent,
    ) -> None:
        """Safe progress emit (callback NESMÍ shodit adapter)."""
        if cb is None:
            return
        try:
            await cb(event)
        except Exception:
            log.exception("progress callback failed: %s", event.stage)

    # ──────────────── Session management API ────────────────

    async def list_sessions(self) -> list[SessionInfo]:
        """Vrátí všechny adapter-managed sessions."""
        return [
            SessionInfo(
                session_id=s.session_id,
                created_at=s.created_at,
                last_active=s.last_active,
                workdir=str(s.workdir) if s.workdir else None,
                model=s.model,
                permission_mode=s.permission_mode,
                destructive_approved=s.destructive_approved,
                state=s.state,
                turn_count=s.turn_count,
            )
            for s in self._sessions.values()
        ]

    async def get_session(self, session_id: str) -> SessionInfo | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return SessionInfo(
            session_id=s.session_id,
            created_at=s.created_at,
            last_active=s.last_active,
            workdir=str(s.workdir) if s.workdir else None,
            model=s.model,
            permission_mode=s.permission_mode,
            destructive_approved=s.destructive_approved,
            state=s.state,
            turn_count=s.turn_count,
        )

    async def clear_session(self, session_id: str) -> bool:
        """`/clear` semantics - wipe history via Claude TUI slash command.

        Claude interactive ma `/clear` co resetuje conversation history.
        Pošle ho do session, počká na re-ready.
        """
        s = self._sessions.get(session_id)
        if s is None or s.state == "DEAD":
            return False
        async with s.lock:
            await self._send_keys(session_id, "/clear", literal=True)
            await asyncio.sleep(0.1)
            await self._send_enter(session_id)
            if s.tui:
                s.tui.reset()
            await self._wait_for_ready(s, timeout_sec=10.0)
        return True

    async def kill_session(self, session_id: str) -> bool:
        """Permanent kill - tmux kill-session + remove z registru."""
        ok = await self._kill_tmux_session(session_id)
        s = self._sessions.pop(session_id, None)
        if s is not None:
            s.state = "DEAD"
        await self._persist_metadata()
        return ok

    async def health_check(self, session_id: str) -> SessionState:
        """Detect dead/orphan sessions."""
        s = self._sessions.get(session_id)
        if s is None:
            return "DEAD"
        if s.state == "DEAD":
            return "DEAD"
        if not await self._has_session(session_id):
            s.state = "DEAD"
            return "DEAD"
        return s.state

    async def close(self) -> None:
        """Adapter shutdown.

        Default policy: tmux sessions ZŮSTÁVAJÍ živé (per #9 invariant -
        sessions persist gemma restart, reattachneme příště). Caller kdy
        chce hard cleanup, volá `kill_session(sid)` explicitly per ID.
        """
        await self._persist_metadata()
