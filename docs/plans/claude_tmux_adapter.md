# Plán: Claude Mode + Bridge Adapter Pattern + Tmux Implementation

## ⚠️ TOS Risk Disclaimer (codex iter-1 critical #1)

Tmux/PTY driving `claude` CLI bez `-p` je **experimentální**. Anthropic
Consumer Terms zakazují "automated/non-human access" mimo API key /
explicitně povolené cesty. I když claude-code/main.tsx:803 detekuje
interactive přes `process.stdout.isTTY` a tmux pseudo-TTY proto vrátí
"interactive", Anthropic má diskreci klasifikovat náš driver jako
Agent SDK use (= $100 pool) nebo ban.

**Důsledky pro plán:**
- TmuxAdapter je **opt-in feature, experimental, default OFF**
- Print mode (`-p`) zůstává **default** dokud Anthropic explicit nepotvrdí
- UI musí jasně signalizovat: "Experimental: drives Claude via tmux,
  subject to Anthropic ToS interpretation"
- **NIKDY** nezvyšovat default na tmux automaticky - user explicit decision
- **NEpřidávat** `claude -p` uvnitř tmux session jako fallback (bypass attempt)
- Plán neslibuje že to bude "obejít" Agent SDK pool - popisuje JEN technický
  mechanism a uživatel jej používá na vlastní riziko

---

## Velký obrázek: Claude Mode jako first-class UX

**User decision (2026-05-19)**: Místo per-turn aktivačních frází ("použij Opus")
přidáme **Claude mode** jako rovnoprávnou alternativu k `chat` a `agent`. User
explicit přepne, zůstává v něm dokud neexitne. Long-lived tmux session
přirozeně poskytuje storage pro tento mode.

**Mode struktura:**
- `chat` - Gemma only, žádné nástroje (= stávající)
- `agent` - Gemma + lokální nástroje (fs, shell, web, hue) (= stávající, BEZ ask_claude)
- `claude` - direct conversation s Claude (Opus/Sonnet/Haiku) přes tmux session
  - žádný Gemma routing per turn, žádný agent loop
  - server přesměruje request přímo do `ClaudeAdapter.ask()`
  - long-lived tmux session per (workdir, model) = context kontinuita
  - approval flow (mode=edit + destructive) zůstává

**Vstup do Claude mode** (všechny tři):
1. **Voice fráze**: "přepni se na Opus" / "přepni na Claude Sonnet" / "udělej přes Opus"
   - Server detekuje fráze, switchne mode na claude + nastaví model
   - První user message po switchi jde rovnou do Claude (ne ztracený)
2. **UI dropdown/button**: V topbaru mode selector
   - Chat | Agent | Claude (rozbalovací: Opus / Sonnet / Haiku)
   - Click = mode switch
3. **Persistence**: Server-side uloží last mode + model do `.gemma_local/ui_state.json`
   - Při dalším otevření UI naskočí stejný mode kde uživatel skončil

**Exit z Claude mode**:
- Voice: "konec Clauda" / "přepni na Gemmu" / "zpět na chat"
- UI: kliknutí na jiný mode button
- Session zůstává živá (tmux server běží nezávisle) → rychlý re-entry

**Smazat `ask_claude` tool** v agent módu:
- Clean separation: Claude přístupný JEN přes Claude mode
- Aktivační fráze "použij Opus" v agent módu → server přepne mode na claude
  a zopakuje query (ne ztracený delegate)
- Zjednoduší: žádný classifier `_cls_ask_claude`, žádný loop override pro
  mode=edit, žádný ExecuteContext.progress_emitter wiring pro ask_claude,
  žádný router mode_hint detection

**ClaudeModePermissionGate (codex iter-1 critical #2):**
Destructive approval gate ze starého `_cls_ask_claude` (mode=edit + "ano povoluju")
NESMÍ zmizet. Replace v server cestě před adapter.ask():

```
POST /api/turn s mode=claude
  → server.claude_mode_handler():
    1. Check claude_session_state[user_session]:
       - permission_mode: "consult" | "edit"  (per-session, persistent)
       - destructive_approved: bool (jednou na session, opt-in)
    2. Pokud incoming user_message indikuje edit intent ("vytvoř/uprav/spusť"):
       a) permission_mode == "consult" → upgrade na "edit" vyžaduje approval
       b) destructive_approved == False → emit approval_required event,
          UI ukáže "Allow Claude to edit/run in this session?" + fráze
          "ano povoluju" required pro persistent toggle
       c) Po approve: state["destructive_approved"] = True na session lifecycle
    3. Adapter spawn s `--permission-mode acceptEdits` JEN pokud edit + approved
    4. Jinak `--permission-mode plan` (read-only)
```

Per-session toggle (= jednou na vstupu do Claude mode user explicit povolí
edit) je UX-lepší než per-turn fráze v dlouhém rozhovoru. Bezpečnostní semantika
zachována: bez explicit "ano povoluju" Claude nikdy nezíská Write/Bash.

UI:
- Při vstupu do Claude mode: badge "🔒 Read-only" / "✏️ Edit allowed"
- Toggle "Allow edit + Bash" button → modal s fráze input
- Voice: "povol editaci" / "ano povoluju editovat" → state toggle

**INVARIANT: permission_mode is immutable per Claude process (codex iter-2 critical #1):**

`--permission-mode plan|acceptEdits` se nastavuje při SPAWN procesu a NELZE
změnit za běhu. Per-session toggle "Allow edit" tedy NEMÚŽE upgradnout existující
session - adapter MUSÍ killnout `plan` session a spawnnout novou s `acceptEdits`.
Symetricky downgrade edit→consult vyžaduje kill+respawn.

Důsledky pro design:
- Toggle "Allow edit" v UI MUSÍ user upozornit "změna ukončí současný kontext
  a vytvoří novou session" + confirm dialog
- Alternativa: paralelní consult/edit session IDs pro stejný (workdir, model).
  User si vybere kterou pokračovat. Více session management overhead.
- **Default: kill+respawn s explicit warning** v MVP. Paralelní sessions later.

Test scenarios pro tohle:
- `test_approve_edit_kills_consult_session`: ASK approval po session start → kill+spawn
- `test_revoke_edit_kills_session`: explicit revoke / exit → kill edit process
- `test_no_silent_permission_change`: send-keys nemůže změnit permission_mode

**Persistent metadata store (codex iter-2 critical #2):**

Session metadata MUSÍ být persistentně uložená a integrity-checked. Jinak
po server restartu by gemma mohla ztratit `destructive_approved` flag zatímco
tmux session s `acceptEdits` přežije = ghost edit capability bez approval state.

Schema `.gemma_local/claude_sessions.json` (atomic write přes tmpfile + rename):
```json
{
  "version": "v1",
  "sessions": {
    "claude_a1b2c3d4": {
      "session_id": "claude_a1b2c3d4",
      "owner": "gemma",                  // identifies our spawned sessions
      "workdir": "/home/box/git/project",
      "model": "claude-opus-4-7",
      "permission_mode": "acceptEdits",  // immutable for life of process
      "permission_argv": ["--permission-mode", "acceptEdits", "--tools", "Read,Edit,Write,Bash,Glob,Grep", "--add-dir", "/home/box/git/project"],
      "created_at": 1715944800.0,
      "last_active": 1715948400.0,
      "approval": {
        "approved_at": 1715944900.0,
        "phrase_hash": "sha256:<phrase + session_id + secret>",
        "approval_version": "v1"
      },
      "turn_count": 5
    }
  }
}
```

Reattach algoritmus při gemma startu:
1. Načíst `claude_sessions.json` (pokud neexistuje, prázdný state)
2. Pro každou file-listed session:
   a) `tmux has-session -t <id>` → pokud false: smazat z metadata, skip
   b) Verify integrity: pokud session měla `acceptEdits` argv, ověř že phrase_hash
      odpovídá uloženému přístupu pro tuto session+secret
   c) Pokud integrity check FAIL → kill tmux session + smazat metadata (unsafe orphan)
   d) Jinak: registrovat v adapteru jako "available for continue"
3. Pro každou tmux `claude_*` session co NENÍ v metadata → unsafe orphan, kill

Test scenarios:
- `test_reattach_with_valid_metadata`: gemma restart → existing approved edit session
  pokračuje
- `test_reattach_unsafe_orphan_killed`: tmux session bez metadata → kill při startup
- `test_reattach_corrupted_metadata`: tampered phrase_hash → kill session
- `test_secret_rotation_invalidates_old_approvals`: secret v `.gemma_local/secret`
  rotated → všechny old `approval.phrase_hash` invalid → sessions vyžadují re-approval

**Tmux session lifecycle (codex iter-1 high #3 - context contamination):**

Naivní per-(workdir, model) cache by způsobila context contamination - jeden
projekt má mnoho různých úkolů, secrets, stale instrukce. Claude Code docs
doporučují `/clear` při změně tasku. Plán:

- **Default per-conversation new session** - každé "New Claude Conversation"
  v UI = nová tmux session (= clean context)
- **Explicit "Continue previous" volba** - UI listuje aktivní sessions
  (z `tmux ls | grep claude_`), user explicitně vybere kterou pokračovat
- Per-session metadata: `created_at`, `last_active`, `workdir`, `model`,
  `permission_mode`, `destructive_approved`
- tmux server (`tmux ls`) běží nezávisle na gemmě → sessions přežijí gemma restart
- Při startu gemmy: scan `tmux ls`, najdi `claude_*` sessions, registruj je
  v adapteru jako "available for continue"
- UI "List sessions" view - user vidí všechny + tlačítka Continue/Kill
- `/clear` command (voice "vyčisti session" / UI button) → adapter pošle
  `/clear` do tmux session → wipe history bez session restart
- "Restart session" → kill + spawn fresh (= full reset)
- Idle timeout cleanup: 24h bez aktivity → session candidate pro auto-kill
  (user může disable v configu)

---

## Motivace

Po **2026-06-15** Anthropic rozdělí Max plán billing:
- **Interactive `claude` CLI v terminálu** (stdout TTY, bez `-p`) → normal Max limity
- **`claude -p` print mode** (= náš dnešní bridge) → separátní Agent SDK pool $100/měs (Max 5x)

Plán přidává **experimentální** alternativu - interactive driver přes pseudo-TTY
(tmux). Plán neslibuje žádný billing benefit; Anthropic má diskreci klasifikovat
jakoukoli automatizaci jako Agent SDK use. Viz **TOS Risk Disclaimer** na začátku.

**User explicitly:**
1. Konfigurační přepínač mezi `print` (= `-p`) a `tmux` adaptérem
2. Identický interface - adaptery zaměnitelné bez code change
3. Tmux adapter long-lived session (lepší context kontinuita)
4. Knihovna v samostatné `src/` struktuře - později pro avatar-engine reuse
5. **Bulletproof testy** s reálným `claude` CLI - API se bude měnit, testy musí drift chytit
6. **Plan → codex review → impl** (žádná spěchaná implementace)

## Klíčové zjištění z `/home/box/git/github/claude-code/src/`

**`main.tsx:803`** - interactive detection:
```typescript
const isNonInteractive = hasPrintFlag || hasInitOnlyFlag || hasSdkUrl || !process.stdout.isTTY;
const isInteractive = !isNonInteractive;
```
→ Spustit `claude` v pseudo-terminal (tmux session) **bez** `-p` flag → `process.stdout.isTTY === true` → interactive mode → Max plan billing.

**Permission modes** (`types/permissions.ts`):
- `acceptEdits` - auto-approve Edit/Write/Bash
- `bypassPermissions` - vše auto
- `dontAsk` - silent fail místo ptaní
- `plan` - read-only, no edits
- `default` - interactive prompts

Pro tmux:
- mode="consult" → spustit s `--permission-mode plan --tools ""`
- mode="edit" → spustit s `--permission-mode acceptEdits --tools "Read,Edit,Write,Bash,Glob,Grep" --add-dir <workdir>` (= shoda s print mode)

**No `--include-partial-messages` ani `--output-format stream-json` v interactive** (ty jsou jen pro `-p`). Tmux musí parsovat **TUI output** (Ink/React terminal UI). To je hlavní výzva.

## Architektura

### Layout - samostatná knihovna v src/

```
gemma/
  src/
    claude_bridge/                    # standalone package, pip-installable
      pyproject.toml                  # min deps, asyncio + ptyprocess/tmux
      README.md
      claude_bridge/
        __init__.py                   # public API surface
        base.py                       # AbstractClaudeAdapter + Protocol
        result.py                     # ClaudeResult dataclass
        progress.py                   # ProgressEvent dataclass + types
        adapters/
          __init__.py
          print_mode.py               # PrintModeAdapter (= refaktor stávající `-p`)
          tmux_mode.py                # TmuxAdapter (long-lived session)
        parsing/
          __init__.py
          stream_json.py              # NDJSON parser (shared)
          ansi.py                     # ANSI escape stripping
          tui_state.py                # claude TUI state machine (prompt detection)
        config.py                     # AdapterConfig, factory
        exceptions.py                 # ClaudeBridgeError taxonomy
      tests/
        unit/
          test_print_mode.py
          test_tmux_mode.py
          test_ansi.py
          test_tui_state.py
        integration/                  # marker claude_cli
          test_print_real.py
          test_tmux_real.py
          test_adapter_parity.py      # OBĚMA adapterům dáme stejné inputy, ověř identický result
  voice/                              # existing gemma
    agent/
      claude_bridge.py                # KILL - kód přesunut do src/claude_bridge/
      tools/
        claude.py                     # importuje z src/claude_bridge, factory podle config
      config.py                       # CLAUDE_BRIDGE_MODE env var
  voice/webapp/static/index.html      # UI dropdown: Bridge mode = print | tmux
```

**pyproject.toml** pro `src/claude_bridge`:
```toml
[project]
name = "claude_bridge"
version = "0.1.0"
dependencies = []   # core: asyncio only, žádné runtime deps pro print adapter

[project.optional-dependencies]
tmux = ["pyte>=0.8"]  # required jen pro tmux adapter (terminal emulator)

[tool.setuptools.packages.find]
where = ["."]
```

avatar-engine bude moci: `pip install -e <gemma>/src/claude_bridge` nebo git submodule.

### Adapter interface (`base.py`) - rozšířený per codex iter-1 high #6

```python
from typing import Protocol, Callable, Awaitable, Literal
from dataclasses import dataclass
from pathlib import Path
import asyncio
from .result import ClaudeResult
from .progress import ProgressEvent

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
Mode = Literal["consult", "edit"]
SessionState = Literal["READY", "RUNNING", "CANCELING", "DEAD"]


@dataclass(frozen=True)
class AdapterCapabilities:
    """Co adapter podporuje. UI/server může na základě toho enable/disable
    features. Klíčové pro budoucí adaptery (HTTP API, OAuth proxy, ...)."""
    supports_cost: bool             # total_cost_usd v ClaudeResult?
    supports_progress: bool         # progress_callback emit eventů?
    supports_persistent_context: bool  # long-lived session?
    supports_session_list: bool     # adapter může listovat existing sessions?
    supports_clear: bool            # /clear bez restartu?
    requires_tty: bool              # vyžaduje pseudo-TTY (= tmux nebo pty)?


@dataclass(frozen=True)
class SessionInfo:
    """Metadata o adapter session (jen relevant pro persistent adaptery)."""
    session_id: str                 # adapter-specific (např. "claude_<hash>")
    created_at: float               # unix epoch
    last_active: float
    workdir: Path
    model: str
    permission_mode: Mode
    destructive_approved: bool
    state: SessionState
    turn_count: int = 0             # kolik turnů už proběhlo


class AbstractClaudeAdapter(Protocol):
    """Společný interface pro print_mode i tmux adapter."""
    name: str  # "print" | "tmux"
    capabilities: AdapterCapabilities

    async def ask(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str,
        mode: Mode,
        workdir: Path | None,
        session_id: str | None = None,   # None = nová session (per-conv)
        timeout_sec: float = 600.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ClaudeResult: ...

    # Session lifecycle (no-op v print adapteru)
    async def list_sessions(self) -> list[SessionInfo]: ...
    async def get_session(self, session_id: str) -> SessionInfo | None: ...
    async def clear_session(self, session_id: str) -> bool:
        """`/clear` semantics - wipe history, keep session alive."""
        ...
    async def kill_session(self, session_id: str) -> bool:
        """Permanent kill - tmux: kill-session, print: noop."""
        ...
    async def health_check(self, session_id: str) -> SessionState:
        """Detect dead/orphan sessions."""
        ...

    async def close(self) -> None:
        """Adapter cleanup. Print: noop. Tmux: shutdown all sessions
        (or release without kill - lifecycle policy v configu)."""
        ...
```

**Print adapter capabilities:** supports_cost=True, supports_progress=True,
supports_persistent_context=False, supports_session_list=False,
supports_clear=False, requires_tty=False.

**Tmux adapter capabilities:** supports_cost=False (interactive neukazuje cost),
supports_progress=True (best-effort z TUI), supports_persistent_context=True,
supports_session_list=True, supports_clear=True, requires_tty=True.

### `ClaudeResult` (`result.py`)

```python
@dataclass(frozen=True)
class ClaudeResult:
    ok: bool
    text: str = ""                      # finální assistant odpověď
    model: str = ""                     # claude-opus-4-7 / claude-sonnet-4-6 / ...
    mode: Mode = "consult"
    session_id: str | None = None
    total_cost_usd: float | None = None  # None pro tmux (nedostupné v interactive)
    duration_ms: int = 0
    tool_uses: tuple[str, ...] = ()
    exit_code: int | None = None
    error: str | None = None
    stderr_preview: str | None = None
    timeout: bool = False
    canceled: bool = False
    # tmux-specific
    adapter: str = ""                    # "print" | "tmux"
```

### `ProgressEvent` (`progress.py`)

```python
ProgressStage = Literal["started", "thinking", "tool_use", "tool_result", "text", "cost"]

@dataclass(frozen=True)
class ProgressEvent:
    stage: ProgressStage
    message: str = ""
    tool_name: str | None = None
    input: dict | None = None
    text: str = ""
    ok: bool | None = None
    session_id: str | None = None
    model: str | None = None
    cost_usd: float | None = None
    duration_ms: int | None = None
```

### PrintModeAdapter - refaktor

Přesunout stávající `voice/agent/claude_bridge.py` logiku do `src/claude_bridge/adapters/print_mode.py`:
- argv builder (`-p --output-format stream-json --include-partial-messages`)
- subprocess spawn s PTY-free pipes
- stream-json parser
- env scrub
- timeout/cancel/killpg

Implementuje `AbstractClaudeAdapter` interface. **Žádná funkční změna** - jen package move + interface fit. Stávající testy by měly projít prakticky beze změny (jen import paths).

### TmuxAdapter - nová implementace

**State machine per session (codex iter-1 high #4, iter-2 high recovery policy):**

```
       ┌─ ask() ──→ RUNNING ──── prompt+complete ──→ READY
READY ─┤
       └─ ask() during RUNNING ──→ raise SessionBusy (mutex blocks)

RUNNING ─── cancel_event ──→ CANCELING ──── Esc+Esc ─→ READY (canceled=True)
RUNNING ─── timeout       ──→ DEAD (kill bez auto-respawn)
RUNNING ─── tmux pid gone ──→ DEAD

DEAD ─── (terminal state, no auto-recovery) → next ask raises SessionDead
                                                UI musí explicit nabídnout
                                                fresh session start
```

**Recovery policy: NIKDY silent recreate** (codex iter-2 high):
Pro long-lived context musíme tichá history loss vyloučit. Pokud session
je DEAD, adapter:
1. Persistně označí session jako DEAD v metadata file
2. Další `ask(session_id=X)` na DEAD session → raise `SessionDead`
3. Server emit `session_dead` event do UI s context preview (last N turns)
4. UI nabídne: "Start fresh session" (default) | "Try resume" (experimental)
5. Fresh start = nový session_id, nová tmux session, čistý context

Implementační detaily:
- `asyncio.Lock` **per session_id** - žádné dvě `ask()` paralelně na stejnou
  session. Druhý request raises `SessionBusy` (žádné silent queueing).
- State stored v `Session` dataclass, transitions přes `async with state_lock:`
- `health_check()` před každým ask: `tmux has-session -t <id>` + capture-pane
  test; pokud broken → mark DEAD persist, raise `SessionDead`.
- `terminate_timeout()` při timeout/cancel doesn't try respawn - DEAD je terminal.

**Lifecycle:**

```
1. session = await adapter._get_or_create_session(
       session_id=session_id, workdir=workdir, model=model,
       permission_mode=pm,  # acceptEdits / plan
   )
2. async with session.lock:
   2a. session.state = "RUNNING"
   2b. tmux send-keys -t <sid> -l "<prompt>"
   2c. tmux send-keys -t <sid> Enter
3. **Incremental transcript collector** (codex iter-2 high #1 - 30KB output):
   Spojuje per-tick capture-pane diffs do persistent transcript buffer. Jen
   posledních 100 řádků by ztratilo velké odpovědi.
   - Před send-keys: marker = capture-pane current bottom line
   - Polling iterace (await asyncio.sleep(0.25)):
     - capture-pane -p -e -S - (entire scrollback s ANSI)
     - tmux session měl `set-option history-limit 100000` při spawnu
     - feed do pyte.Stream/Screen pro per-tick state (spinner, modal, ready)
     - APPEND nový obsah do `self._transcript_buffer[session_id]` (diff oproti
       last capture, deduplicate header lines)
   - emit progress events (callback) z pyte state
   - check cancel_event → state="CANCELING", send Esc Esc, break
   - check ready marker + idle > 0.8s → break (default, configurable)
4. Extract response z transcript_buffer:
   - Slice mezi marker (před send) a aktuální ready prompt
   - Strip ANSI escapes (re-feed přes pyte alebo regex)
   - Strip TUI chrome (borders, status lines, footer)
   - Return clean assistant text v ClaudeResult.text
5. state = "READY", uvolnit lock
```

**Parser: pyte terminal emulator (codex iter-1 high #5):**

Místo regex na raw capture-pane output použijeme `pyte` library - full
terminal emulator co simuluje VT100/xterm. Output je deterministický 2D screen
grid s atributy (foreground, bold, ...) bez křehkosti regex parsingu.

```python
import pyte

screen = pyte.Screen(cols=200, rows=50)
stream = pyte.Stream(screen)

# Feed raw bytes z tmux capture-pane -e (s ANSI)
raw = await self._capture_raw(session_id)
stream.feed(raw.decode("utf-8", errors="replace"))

# screen.display = list[str] - každý řádek 200 chars wide, bez ANSI
lines = screen.display

# Tool_use detection: hledej "⏺" v lines, parse name + args
# Spinner detection: posledních N capture s rotujícími braille symbols
# Ready prompt: bottom rows mají typický input box layout (víc-řádkové,
#   detection přes screen.cursor pozici + content kolem)
```

**Klíčové výzvy a jejich řešení:**

1. **Prompt detection** - pyte screen.cursor pozice + okolní content. Backup:
   idle > N ms (configurable, default 800 ms).
2. **Spinner detection** - pyte feed posledních N capture, detect rotation
   braille symbols na cursor řádku.
3. **Tool_use parse** - match `⏺ Tool(args)` pattern v cleaned screen text.
4. **Long prompts** - tmux send-keys limit (cca 4 KB). Split na chunky,
   mezi `-l` calls krátký sleep.
5. **Multi-line prompts** - newline ≠ submit. Replace `\n` na C-j (Ctrl+J)
   nebo bracketed paste mode.
6. **Send-keys race** - verify after send: capture-pane musí obsahovat
   poslaný text (echo check).
7. **Approval modal drift** - pokud Claude přesto ukáže approval prompt
   (mimo acceptEdits), state machine detect → emit `tool_progress("modal")`
   + raise unless `auto_yes` flag set.
8. **Cancel** - `tmux send-keys -t <id> Escape Escape`. Pokud tool_use je
   blocking I/O (long Bash), claude TUI musí Esc respektovat. Fallback:
   timeout → kill+spawn.
9. **Resize handling** - capture window má fixed size (cols=200, rows=50).
   Long line wrap může zmást parser. Test: prompty/odpovědi co produkují
   wrapped content.
10. **Recovery z dead session** - `health_check()` pre-ask. Pokud DEAD →
    raise `SessionDead`, server na to UI reaguje "session expired, start new".
11. **Cleanup** - `__aexit__` + atexit hook zabíjí JEN sessions co adapter
    spawnoval, NEKILLuje user's other tmux sessions.

### Config & Factory (`config.py`)

```python
class BridgeMode(str, Enum):
    PRINT = "print"
    TMUX = "tmux"

@dataclass
class AdapterConfig:
    mode: BridgeMode = BridgeMode.PRINT  # default until tmux ready
    claude_bin: str = "claude"
    tmux_bin: str = "tmux"
    # ... more

def create_adapter(config: AdapterConfig) -> AbstractClaudeAdapter:
    if config.mode == BridgeMode.TMUX:
        from .adapters.tmux_mode import TmuxAdapter
        return TmuxAdapter(config)
    return PrintModeAdapter(config)
```

V gemma:
- `voice/agent/config.py`:
  ```python
  CLAUDE_BRIDGE_MODE: str = os.environ.get("AGENT_CLAUDE_BRIDGE_MODE", "print")
  ```
- `voice/agent/tools/claude.py`: factory().ask(...) místo přímé volání ask_claude_oneshot

### UI switch

`voice/webapp/static/index.html` + `app.js`:
- Settings panel → "Claude bridge" radio: `Print mode (-p)` (default) / `Tmux interactive session (experimental)`
- POST do `/api/config/claude_bridge_mode` (server persistne do `.gemma_local/`)
- Server respektuje env var override

## Testing strategy - bulletproof

User mandát: **API se bude měnit, testy musí drift chytit**.

### Layer 1 - Unit (no claude, no tmux)
- `test_ansi.py`: known ANSI sekvence → strip správně
- `test_tui_state.py`: state machine s mock terminál bufferem (replay zachycených tmux capture-pane snapshotů uložených jako fixtures)
- `test_print_mode.py`: existing tests refaktorované, fake subprocess
- `test_progress.py`: dataclass invariants

### Layer 2 - Integration (real tmux, fake claude)
- Spawn real tmux session se shell scriptem co předstírá claude TUI
- Verify send-keys + capture-pane chain funguje
- Test cancel/timeout

### Layer 3 - claude_cli marker (real claude + real tmux)
**Klíčové dle user mandátu** - verifikace proti živému CLI. Pokrytí
edge cases (codex iter-1 high #5 - bulletproof):

```python
@pytest.mark.claude_cli
class TestTmuxAdapterReal:
    # Basic functionality
    async def test_simple_arithmetic(self):
        """2+2 → 4, žádný tool_use"""

    async def test_edit_creates_file(self, tmp_workdir):
        """vytvořit hello.py přes Write tool"""

    async def test_consult_no_fs_access(self):
        """mode=consult → Claude nemá Read tool, řekne 'nemůžu'"""

    # Long-lived session semantics
    async def test_long_session_context(self, tmp_workdir):
        """3× ask v rámci jedné session - 3. má pamatovat 1. a 2."""

    async def test_clear_wipes_history(self, tmp_workdir):
        """ask 'remember X', /clear, ask 'co bylo X?' → Claude nepamatuje"""

    async def test_session_survives_gemma_restart(self, tmp_workdir):
        """spawn session, simulate adapter restart, reattach,
        Claude pořád pamatuje pre-restart history"""

    # Concurrency + state machine
    async def test_concurrent_asks_serialized(self, tmp_workdir):
        """dvě paralelní ask() na stejnou session → druhá blokuje na lock,
        ne race v send-keys"""

    async def test_dead_session_no_silent_recreate(self, tmp_workdir):
        """externí kill session → next ask raises SessionDead.
        Adapter NESMÍ silent respawn (kontext by se ztratil bez varování)."""

    async def test_session_dead_event_emitted(self, tmp_workdir):
        """DEAD session → UI obdrží session_dead event s context preview"""

    # Cancel + timeout
    async def test_cancel_via_event(self, tmp_workdir):
        """cancel během běhu → Esc Esc, session zůstane READY"""

    async def test_timeout_marks_session_dead(self, tmp_workdir):
        """sleep timeout > timeout_sec → session DEAD (terminal), no respawn"""

    # Progress events
    async def test_progress_events_emitted(self, tmp_workdir):
        """thinking + tool_use events viditelné během běhu"""

    async def test_progress_has_tool_args(self, tmp_workdir):
        """tool_use eventy mají enriched message (Read /file.py, ne jen Read)"""

    # Edge cases per codex iter-1
    async def test_approval_modal_drift(self, tmp_workdir):
        """pokud Claude zobrazí approval prompt (mimo acceptEdits),
        adapter ho detekuje + raise nebo auto_yes flag"""

    async def test_wrapped_long_command(self, tmp_workdir):
        """Bash command s 300+ znakovou výstupní řádku → pyte správně wrapuje,
        parser extract clean text"""

    async def test_multiline_prompt(self, tmp_workdir):
        """3-řádkový prompt s explicit newlines → Claude vidí celý,
        ne submit po prvním newline"""

    async def test_30kb_output_no_truncation(self, tmp_workdir):
        """ask za 30KB markdown → ClaudeResult.text == celý,
        scrollback z tmux capture-pane stačí"""

    async def test_resize_safe(self, tmp_workdir):
        """tmux resize-window během běhu nesmí zlomit capture parser"""

    async def test_cli_error_screen(self, tmp_workdir):
        """zavolat s neexistujícím modelem → CLI error v capture,
        adapter detekuje + vrátí ClaudeResult.ok=False s error message"""

@pytest.mark.claude_cli
class TestAdapterParity:
    """SAME inputy do print i tmux → equivalent ClaudeResult.
    Tolerujeme rozdíly: cost_usd (tmux nemá), session_id, duration."""
    @pytest.mark.parametrize("scenario", [
        "simple_qa",
        "single_file_create",
        "multi_file_refactor",
        "long_thinking_task",
        "consult_qa",
    ])
    async def test_parity(self, scenario, tmp_workdir):
        ...

@pytest.mark.claude_cli
class TestClaudeModePermissionGate:
    """Critical security: bez 'ano povoluju' fráze nesmí Claude získat Write/Bash."""
    async def test_consult_default_no_edit(self, tmp_workdir):
        """vstup do Claude mode bez approve → mode=consult → Write tool fail"""

    async def test_edit_requires_explicit_phrase(self, tmp_workdir):
        """toggle 'Allow edit' bez fráze 'ano povoluju' → 400 phrase mismatch"""

    async def test_edit_persistent_for_session(self, tmp_workdir):
        """jednou approved → další turny v session mohou editovat bez fráze"""

    async def test_destructive_approval_per_session_not_per_turn(self, tmp_workdir):
        """20 turnů po approve, žádné další fráze prompt → 1 fráze stačí na session"""
```

### Layer 4 - Stress tests
- Tmux session lifecycle: 20 sessions paralelně, cleanup verifikace
- Long-lived: 1 session, 50 turns, paměť stabilní
- Recovery: kill tmux PID externě → adapter detect + recreate

### CI strategy
- Default pytest: jen unit + integration (= žádný real claude)
- `pytest -m claude_cli`: spustit lokálně před commit, vyžaduje Max plan auth
- Output fixtures: ukládat zachycené capture-pane snapshoty do `tests/fixtures/tmux_snapshots/` → unit testy můžou replay-ovat

## Implementační fáze

1. **Fáze 0 - Setup library skeleton** (1 hodina práce, bez behaviorální změny)
   - Vytvořit `src/claude_bridge/` strukturu + pyproject.toml
   - Přesunout `voice/agent/claude_bridge.py` → `src/claude_bridge/adapters/print_mode.py`
   - Definovat `AbstractClaudeAdapter` interface
   - `voice/agent/tools/claude.py` importuje z nové lokace, žádná funkční změna
   - Spustit full test suite → musí být beze ztrát
   - **Commit checkpoint**

2. **Fáze 1 - Tmux adapter MVP**
   - tmux session lifecycle (new-session / kill-session)
   - send-keys + capture-pane primitivy
   - ANSI stripping
   - Single-shot ask (žádné progress eventy zatím)
   - Real claude_cli test simple_qa pass
   - **Commit checkpoint**

3. **Fáze 2 - TUI state machine + prompt detection**
   - Detect ready prompt, spinner, tool_use linky
   - Fixture-based unit tests s capturováním snapshotů
   - Real test: tool_use viditelný v capture-pane
   - **Commit checkpoint**

4. **Fáze 3 - Progress events**
   - Polling capture-pane diff → ProgressEvent emit
   - Real test: progress callback dostává thinking + tool_use eventy
   - **Commit checkpoint**

5. **Fáze 4 - Long-lived sessions + clear/restart**
   - Per-conversation new session default
   - List/continue UI flow + persistent metadata
   - `/clear` + `restart` operations
   - Reattach po gemma restartu (scan `tmux ls`)
   - **Commit checkpoint**

6. **Fáze 5 - Claude Mode UX + ClaudeModePermissionGate**
   - Server endpoint mode=claude bypassuje agent loop
   - Per-session permission state (consult/edit toggle + destructive_approved)
   - Server gate před adapter.ask() (replace ask_claude classifier)
   - UI mode selector + model picker + voice fráze entry/exit
   - Persist last mode v `.gemma_local/ui_state.json`
   - Smazat starý ask_claude tool + jeho testy + classifier + loop override
   - **Commit checkpoint**

7. **Fáze 6 - Adapter parity tests + edge cases + stress**
   - Parity suite (oba adaptery, shared scenarios)
   - Cancel/timeout/error scenarios pro tmux
   - Stress: 20 paralelních sessions, 50-turn long-lived, recovery
   - Edge cases: approval modal drift, wrapped output, multiline prompt,
     30 KB output, resize, dead session
   - **Commit checkpoint**

8. **Fáze 7 - Codex review iter + opt-in default + TOS disclaimer**
   - Per memory mandát: codex review cyklus dokud critical/high = 0
   - Tmux adapter zůstává **opt-in, experimental, default OFF** (codex iter-1 #1)
   - UI mode selector ukazuje "Experimental" badge u tmux
   - README + UI tooltip s TOS risk disclaimer
   - Print mode zůstává default

**Default přepnutí pravidla (codex iter-1 #7):**
- Tmux je default-OFF dokud:
  1. Všechny fáze hotové (NEjak po fázi 3)
  2. Real CLI drift suite zelená 1 týden+
  3. Anthropic explicit potvrdí TOS legitimitu **nebo** user explicit opt-in
     vědomi rizika
- Print mode zůstává **vždy dostupný** jako fallback (nikdy nesmazat)

## Open questions pro codex review

1. **Permission auto-approve**: V interactive režimu nestačí `acceptEdits` pro Bash? Zkontrolovat zda Claude přesto v některých případech ptá.
2. **Long-lived session security**: Pokud session žije přes více user requestů, leak kontextu mezi taskama. Reset session při změně workdir?
3. **tmux dependency**: Vyžaduje tmux v PATH. Co když user nemá? Detection at startup + fail-fast s navodným error.
4. **PTY alternativa**: Místo tmux použít Python `pty` modul nebo `ptyprocess` package. Méně dependency, ale tmux je battle-tested pro screen scraping.
5. **Cost tracking**: Tmux mode nemá `total_cost_usd` (claude interactive ho neuvádí). Můžeme jen aproximovat z token counts pokud Claude exposuje. User to ale potřebuje k UI display.
6. **REJECTED** ~~Stream-json fallback s `claude -p` uvnitř tmux session~~ -
   to by byl billing bypass attempt (Anthropic by oprávněně klasifikoval jako
   Agent SDK use). Plán explicitně zakazuje na řádku 18.
7. **Concurrency**: Lze poslat dvě paralelní ask() na jednu session? Pravděpodobně ne - single-tracked. Mutex per session.
8. **Recovery z dead session**: Pokud `tmux has-session` false (kill externí) → recreate, ale ztratíme history. Acceptable degradation?

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Anthropic změní TUI layout → naše TUI parser broken | Layer 3 real CLI testy chytí drift; cherry-picked fixtures přidat při každé Claude CLI version bump |
| Anthropic explicit zakáže tmux/pseudo-TTY driving | Print mode adapter pořád funkční jako fallback; user může přepnout |
| tmux v různých distrech různě se chová | Testovat na ubuntu + arch (user má arch) v CI; document requirements |
| Long-lived session leak (memory, context) | Idle timeout cleanup, max sessions cap |
| Send-keys lost (race) | Verify after send: capture-pane musí obsahovat poslané text |
| Cost neviditelný v tmux | Display "N/A (interactive)" v UI, user ví že je v Max plánu |

## Codex review findings - adresované

### Iter-1 (7 findings, vše adresované)

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | CRITICAL | TOS compliance tmux automation | Tmux = opt-in experimental, default OFF; print zůstává default; UI disclaimer; nepřidávat `-p` v tmux jako fallback |
| 2 | CRITICAL | Claude mode obchází destructive approval | ClaudeModePermissionGate v server cestě před adapter.ask(); per-session toggle "Allow edit" s "ano povoluju" frází |
| 3 | HIGH | Context contamination v long-lived | Default per-conversation new session; explicit "Continue" UI; /clear command; idle timeout cleanup |
| 4 | HIGH | Tmux mutex + state machine v MVP | asyncio.Lock per session, READY/RUNNING/CANCELING/DEAD state machine od Fáze 1, health_check pre-ask |
| 5 | HIGH | Parser křehký (regex) | pyte terminal emulator místo regex; edge case test suite (wrap, multiline, 30KB output, modal drift, resize, CLI error) |
| 6 | HIGH | Interface málo abstraktní | AdapterCapabilities, SessionInfo, list_sessions/clear/kill/health_check metody |
| 7 | HIGH | Default přepnut moc brzy | Tmux opt-in až po Fázi 6 + real drift suite zelená 1 týden+ + (Anthropic OK NEBO user explicit risk awareness) |

### Iter-2 (6 nových findings, vše adresované)

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 8 | CRITICAL | permission_mode immutable per process | Invariant: kill+respawn pro toggle consult↔edit; explicit warning v UI před toggle; test_no_silent_permission_change |
| 9 | CRITICAL | Reattach ztrácí approval state | Persistent metadata `.gemma_local/claude_sessions.json` s phrase_hash + integrity check; unsafe orphan kill při startup |
| 10 | CRITICAL | Billing workaround wording | Odstraněn z motivace; UI label "Tmux interactive session (experimental)"; open question `claude -p` v tmux REJECTED |
| 11 | HIGH | Capture ztratí dlouhé odpovědi | Incremental transcript collector (per-tick diff append); tmux history-limit=100000; test_30kb_output assertuje proti truncation |
| 12 | HIGH | pyte dependency mismatch | pyproject `[project.optional-dependencies] tmux = ["pyte>=0.8"]`; fail-fast pokud tmux mode bez dep |
| 13 | HIGH | Dead/timeout recovery nekonzistentní | Jednotná politika: DEAD je terminal, NIKDY silent recreate. SessionDead raise → UI nabídne fresh start s context preview |

---

## Codex review prompt template

```
Review tento plán pro Claude Bridge Adapter pattern. Focus:
- Security boundary: ASK destructive flow musí zůstat zachován v obou adapterech
- TOS compliance: je tmux/interactive driver legitimate dle Anthropic policy?
- Test coverage adequacy: chytí změny v Claude CLI API drift?
- API stability: adapter interface dostatečně abstrahuje?
- Edge cases: race conditions, session leaks, ANSI parsing failures?
- Sessio long-lived design: jak řešit cross-task context contamination?

Critical/high findings only. Konkrétní akce nebo OK = LGTM.
```
