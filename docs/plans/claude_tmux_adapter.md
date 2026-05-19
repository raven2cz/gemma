# Plán: Claude Bridge Adapter Pattern + Tmux Implementation

## Motivace

Po **2026-06-15** Anthropic rozdělí Max plán billing:
- **Interactive `claude` CLI v terminálu** (stdout TTY, bez `-p`) → normal Max limity
- **`claude -p` print mode** (= náš dnešní bridge) → separátní Agent SDK pool $100/měs (Max 5x)

Pro user s Max 5x je $100/měs pool nedostatečný pro Opus implementační tasky. Tmux/PTY-driven interactive mode by mohl spadat pod "interactive in terminal" kategorii → normal Max limity.

**User explicitly:**
1. Konfigurační přepínač mezi `print` (= `-p`) a `tmux` adaptérem
2. Identický interface — adaptery zaměnitelné bez code change
3. Tmux adapter long-lived session (lepší context kontinuita)
4. Knihovna v samostatné `src/` struktuře — později pro avatar-engine reuse
5. **Bulletproof testy** s reálným `claude` CLI — API se bude měnit, testy musí drift chytit
6. **Plan → codex review → impl** (žádná spěchaná implementace)

## Klíčové zjištění z `/home/box/git/github/claude-code/src/`

**`main.tsx:803`** — interactive detection:
```typescript
const isNonInteractive = hasPrintFlag || hasInitOnlyFlag || hasSdkUrl || !process.stdout.isTTY;
const isInteractive = !isNonInteractive;
```
→ Spustit `claude` v pseudo-terminal (tmux session) **bez** `-p` flag → `process.stdout.isTTY === true` → interactive mode → Max plan billing.

**Permission modes** (`types/permissions.ts`):
- `acceptEdits` — auto-approve Edit/Write/Bash
- `bypassPermissions` — vše auto
- `dontAsk` — silent fail místo ptaní
- `plan` — read-only, no edits
- `default` — interactive prompts

Pro tmux:
- mode="consult" → spustit s `--permission-mode plan --tools ""`
- mode="edit" → spustit s `--permission-mode acceptEdits --tools "Read,Edit,Write,Bash,Glob,Grep" --add-dir <workdir>` (= shoda s print mode)

**No `--include-partial-messages` ani `--output-format stream-json` v interactive** (ty jsou jen pro `-p`). Tmux musí parsovat **TUI output** (Ink/React terminal UI). To je hlavní výzva.

## Architektura

### Layout — samostatná knihovna v src/

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
      claude_bridge.py                # KILL — kód přesunut do src/claude_bridge/
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
dependencies = []   # zero deps - asyncio only, tmux přes subprocess
```

avatar-engine bude moci: `pip install -e <gemma>/src/claude_bridge` nebo git submodule.

### Adapter interface (`base.py`)

```python
from typing import Protocol, Callable, Awaitable, Literal
from pathlib import Path
import asyncio
from .result import ClaudeResult
from .progress import ProgressEvent

ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]
Mode = Literal["consult", "edit"]

class AbstractClaudeAdapter(Protocol):
    """Společný interface pro print_mode i tmux adapter.

    Single-shot API (ask) je primární. Adapter může uvnitř držet long-lived
    state (tmux session) ale z venku vypadá vždy stejně.
    """
    name: str  # "print" | "tmux"

    async def ask(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str,
        mode: Mode,
        workdir: Path | None,
        timeout_sec: float = 600.0,
        cancel_event: asyncio.Event | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ClaudeResult: ...

    async def close(self) -> None:
        """Cleanup. Print: noop. Tmux: kill session(s)."""
        ...
```

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

### PrintModeAdapter — refaktor

Přesunout stávající `voice/agent/claude_bridge.py` logiku do `src/claude_bridge/adapters/print_mode.py`:
- argv builder (`-p --output-format stream-json --include-partial-messages`)
- subprocess spawn s PTY-free pipes
- stream-json parser
- env scrub
- timeout/cancel/killpg

Implementuje `AbstractClaudeAdapter` interface. **Žádná funkční změna** — jen package move + interface fit. Stávající testy by měly projít prakticky beze změny (jen import paths).

### TmuxAdapter — nová implementace

**Lifecycle:**

```
1. ask(prompt, mode, workdir) → adapter určí session_key
   = hash(workdir, mode, model)  # per-(dir,mode,model) session
2. Pokud session_key není v _sessions:
   - tmux new-session -d -s claude_<key> -x 200 -y 50 \
       -c <workdir> \
       'claude --permission-mode <pm> --tools <t> --model <m>'
   - Poll capture-pane until prompt ready
3. tmux send-keys -t claude_<key> -l "<prompt>"
   tmux send-keys -t claude_<key> Enter
4. Poll capture-pane každých 250ms:
   - Detekce thinking spinner ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ → emit "thinking" progress
   - Detekce tool_use lines (Ink output má specific format) → emit
   - Detekce "approved" prompts → auto-respond (acceptEdits to měl pokrýt)
   - Detekce prompt re-appearance + idle > 500ms = response complete
5. Extract response: capture-pane -S - vrátí historii, slice mezi
   poslední user prompt a aktuální prompt
6. Strip ANSI, vrátit ClaudeResult
7. Session zůstane open pro další ask v rámci stejného session_key
   (lepší context kontinuita - Claude pamatuje předchozí turns)
```

**close():** `tmux kill-session -t claude_<key>` pro každou aktivní session.

**Klíčové výzvy a řešení:**

1. **Prompt detection** — Claude interactive UI má specifický input boxíček.
   Strategy: Look for `>` character at column 1 nebo special Unicode ready markers (capture-pane bez -e dá raw text bez ANSI). Backup: idle timeout > 2s = done.

2. **Spinner detection** — Braille spinner postupně rotuje. Detect `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` symbols → emit progress("thinking").

3. **Tool_use parse** — Claude TUI vypisuje `⏺ Tool name(args)` nebo podobné. Regex match. Méně bohaté než stream-json ale podstatu pokryje (tool_name + truncated args).

4. **Long prompts** — tmux send-keys limit cca 4 KB per call. Větší dělit na chunky, mezi `-l` calls čekat krátce.

5. **Multi-line prompts** — newline v promptu = předčasné submit. Strategy: replace `\n` na **Shift+Enter** sekvenci (claude TUI to bere jako newline bez submitu). `tmux send-keys ... C-j` (Ctrl+J) nebo escape sekvence.

6. **Race podmínky** — send-keys před prompt ready. Strategy: vždy capture-pane + verify ready marker přijít před send-keys.

7. **ANSI parsing** — `pyte` library nebo custom regex `r'\x1b\[[?]?[0-9;]*[a-zA-Z]'`. Strip mouse/cursor, keep text content.

8. **Approval modals** — pokud Claude přesto požádá (mimo acceptEdits), v default behaviour fail-fast. Optional `auto_yes` flag.

9. **Cancel** — `tmux send-keys -t <session> Escape Escape` posle Esc a cancelne probíhající tool_use bez zabití session.

10. **Cleanup** — `__aexit__` + atexit hook zabíjí všechny `claude_*` tmux sessions z naší instance.

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
- Settings panel → "Claude bridge" radio: `Print mode (-p)` / `Tmux session (Max limits)`
- POST do `/api/config/claude_bridge_mode` (server persistne do `.gemma_local/`)
- Server respektuje env var override

## Testing strategy — bulletproof

User mandát: **API se bude měnit, testy musí drift chytit**.

### Layer 1 — Unit (no claude, no tmux)
- `test_ansi.py`: known ANSI sekvence → strip správně
- `test_tui_state.py`: state machine s mock terminál bufferem (replay zachycených tmux capture-pane snapshotů uložených jako fixtures)
- `test_print_mode.py`: existing tests refaktorované, fake subprocess
- `test_progress.py`: dataclass invariants

### Layer 2 — Integration (real tmux, fake claude)
- Spawn real tmux session se shell scriptem co předstírá claude TUI
- Verify send-keys + capture-pane chain funguje
- Test cancel/timeout

### Layer 3 — claude_cli marker (real claude + real tmux)
**Klíčové dle user mandátu** — verifikace proti živému CLI:

```python
@pytest.mark.claude_cli
class TestTmuxAdapterReal:
    async def test_simple_arithmetic(self):
        """2+2 → 4, žádný tool_use"""
    
    async def test_edit_creates_file(self, tmp_workdir):
        """vytvořit hello.py přes Write tool"""
    
    async def test_long_session_context(self, tmp_workdir):
        """3× ask v rámci jedné session - 3. má pamatovat 1. a 2."""
    
    async def test_cancel_via_event(self, tmp_workdir):
        """cancel během běhu → Claude přeruší, ale session zůstane"""
    
    async def test_progress_events_emitted(self, tmp_workdir):
        """thinking + tool_use events viditelné během běhu"""
    
    async def test_consult_no_fs_access(self):
        """mode=consult → Claude nemá Read tool, řekne 'nemůžu'"""
    
    async def test_large_output_design_doc(self, tmp_workdir):
        """30+ KB markdown vrácený celý, ne truncated"""

@pytest.mark.claude_cli
class TestAdapterParity:
    """SAME inputy do print i tmux → equivalent ClaudeResult.
    Tolerujeme rozdíly: cost_usd (tmux nemá), session_id, duration."""
    @pytest.mark.parametrize("scenario", [
        "simple_qa",
        "single_file_create",
        "multi_file_refactor",
    ])
    async def test_parity(self, scenario, tmp_workdir):
        ...
```

### Layer 4 — Stress tests
- Tmux session lifecycle: 20 sessions paralelně, cleanup verifikace
- Long-lived: 1 session, 50 turns, paměť stabilní
- Recovery: kill tmux PID externě → adapter detect + recreate

### CI strategy
- Default pytest: jen unit + integration (= žádný real claude)
- `pytest -m claude_cli`: spustit lokálně před commit, vyžaduje Max plan auth
- Output fixtures: ukládat zachycené capture-pane snapshoty do `tests/fixtures/tmux_snapshots/` → unit testy můžou replay-ovat

## Implementační fáze

1. **Fáze 0 — Setup library skeleton** (1 hodina práce, bez behaviorální změny)
   - Vytvořit `src/claude_bridge/` strukturu + pyproject.toml
   - Přesunout `voice/agent/claude_bridge.py` → `src/claude_bridge/adapters/print_mode.py`
   - Definovat `AbstractClaudeAdapter` interface
   - `voice/agent/tools/claude.py` importuje z nové lokace, žádná funkční změna
   - Spustit full test suite → musí být beze ztrát
   - **Commit checkpoint**

2. **Fáze 1 — Tmux adapter MVP**
   - tmux session lifecycle (new-session / kill-session)
   - send-keys + capture-pane primitivy
   - ANSI stripping
   - Single-shot ask (žádné progress eventy zatím)
   - Real claude_cli test simple_qa pass
   - **Commit checkpoint**

3. **Fáze 2 — TUI state machine + prompt detection**
   - Detect ready prompt, spinner, tool_use linky
   - Fixture-based unit tests s capturováním snapshotů
   - Real test: tool_use viditelný v capture-pane
   - **Commit checkpoint**

4. **Fáze 3 — Progress events**
   - Polling capture-pane diff → ProgressEvent emit
   - Real test: progress callback dostává thinking + tool_use eventy
   - **Commit checkpoint**

5. **Fáze 4 — Long-lived sessions**
   - Per-(workdir, mode, model) session cache
   - Context kontinuita test
   - **Commit checkpoint**

6. **Fáze 5 — Config switch + UI**
   - factory + env var
   - UI dropdown + persist
   - Default přepnout na `tmux` až bude stabilní (otestováno v Fázi 3)
   - **Commit checkpoint**

7. **Fáze 6 — Adapter parity tests + edge cases**
   - Parity test suite proti oběma adapterům
   - Cancel/timeout/error scenarios pro tmux
   - Stress testy
   - **Commit checkpoint**

8. **Fáze 7 — Codex review + iter**
   - Per memory mandát: codex review cyklus dokud critical/high = 0
   - Pak merge do main

## Open questions pro codex review

1. **Permission auto-approve**: V interactive režimu nestačí `acceptEdits` pro Bash? Zkontrolovat zda Claude přesto v některých případech ptá.
2. **Long-lived session security**: Pokud session žije přes více user requestů, leak kontextu mezi taskama. Reset session při změně workdir?
3. **tmux dependency**: Vyžaduje tmux v PATH. Co když user nemá? Detection at startup + fail-fast s navodným error.
4. **PTY alternativa**: Místo tmux použít Python `pty` modul nebo `ptyprocess` package. Méně dependency, ale tmux je battle-tested pro screen scraping.
5. **Cost tracking**: Tmux mode nemá `total_cost_usd` (claude interactive ho neuvádí). Můžeme jen aproximovat z token counts pokud Claude exposuje. User to ale potřebuje k UI display.
6. **Stream-json fallback**: Co kdybychom v tmux interactive shell ručně spustili `claude -p` jako sub-process **uvnitř** té tmux session? Naivně tato vrstva nedělá billing-relevant work, ale nejistý jestli Anthropic to akceptuje jako "interactive".
7. **Concurrency**: Lze poslat dvě paralelní ask() na jednu session? Pravděpodobně ne — single-tracked. Mutex per session.
8. **Recovery z dead session**: Pokud `tmux has-session` false (kill externí) → recreate, ale ztratíme history. Acceptable degradation?

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Anthropic změní TUI layout → naše TUI parser broken | Layer 3 real CLI testy chytí drift; cherry-picked fixtures přidat při každé Claude CLI version bump |
| Anthropic eventually closes TOS workaround | Print mode adapter pořád funkční jako fallback; user může přepnout |
| tmux v různých distrech různě se chová | Testovat na ubuntu + arch (user má arch) v CI; document requirements |
| Long-lived session leak (memory, context) | Idle timeout cleanup, max sessions cap |
| Send-keys lost (race) | Verify after send: capture-pane musí obsahovat poslané text |
| Cost neviditelný v tmux | Display "N/A (interactive)" v UI, user ví že je v Max plánu |

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
