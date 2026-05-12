# Agent Mode — Implementation Plan (v3, sign-off)

Branch: `feature/agent-mode`
Status: **approved, ready for Phase 1 implementation**

## 1. Goal

Přidat voice webappu **agentní režim** = tool-calling smyčka kolem stejného
Gemma modelu, hlasové i textové schvalování, sub-agent Claude (Code CLI, MAX
plan) pro složitější úlohy. Současně přidat **skills** (read-only network
tooly) i do CHAT mode, aby web search / fetch fungovaly bez přepínání režimu.

## 2. Klíčová rozhodnutí (po user feedbacku + codex review)

| Téma | Rozhodnutí | Detail |
|------|-----------|--------|
| Endpoint | **Rozšířit `/api/turn`**, ne nový `/api/agent/stream` | Codex: jinak duplikuju TTS/cancel/audio queue logiku. Přidám pole `mode` (chat/agent/claude). |
| Model | Stejný pro chat i agent | User si volí. UI varování při Gemma 3. |
| Sandbox | WORKDIR = `Path.cwd().resolve()` při startu | Konstantní |
| Read uvnitř WORKDIR | AUTO | bez ask |
| Read mimo WORKDIR | **Allowlist veřejných cest = AUTO, zbytek = ASK** | `/etc/os-release`, `/etc/lsb-release`, `/proc/cpuinfo`, `/proc/meminfo`, `/proc/version`, `/etc/hostname` |
| Read limit | 256 KiB default + symlink resolve + reject special files (`/proc`, `/dev`, sockety, fifa) | Codex doporučení |
| Bash | **AUTO allowlist** (`pwd ls rg find git status/diff/log/show cat`), vše ostatní ASK | Codex: blacklist je security theatre |
| Bash spec | cwd=WORKDIR, timeout, output cap (1 MiB), env scrub | Codex |
| Write/edit uvnitř WORKDIR | AUTO | |
| Write/edit mimo WORKDIR | ASK | path resolve před exekucí |
| TTS off mode | UI modal je canonical, hlas je alternativní | Approval event vždy do frontendu |
| Skills v chat mode | `web_search` + `fetch_url`, max 1 round, stejný ToolRegistry | Tool calls do conversation history |
| Mode state | **Frontend localStorage** (server stateless) | Codex: méně stavu = méně bugů |
| Mode triggers | „přepni do agenta" / „chat mode" / „přes Claude/Opus/Sonnet" / „Claude mode" | Regex CS+EN |
| Claude bridge | Per-session subprocess, persistent, `--add-dir WORKDIR` | Codex: ne global, dráinovat stderr, log session_id |
| Claude reload | Neukončovat subprocess, jen odpojit consumer | Pokračující proces má cancel endpoint |
| OpenHue | **Plán B (CLI wrapper) první**, MCP později jako adapter | Codex: MCP overkill pro 6 toolů |
| Tool transcript | Collapsible karty + tool calls v conversation history | Codex: schema hned ve fázi 1 |

## 3. Audit kódové báze

### `voice/webapp/server.py` současný stav
- `/api/turn` (řádek ~1063) — **THE** endpoint. NDJSON streaming, sentence chunker,
  TTS pipeline, cancel logika, audio queue. **Tady rozšíříme**, nový endpoint nevytváříme.
- `/api/chat` (řádek ~740) — single-turn JSON (non-streaming). Asi necháme, ale neřeší se.
- `SYSTEM_PROMPTS` (103) + `MARKDOWN_SYSTEM_PROMPTS` (953) — per-lang. Rozšířit o
  skills text.
- TTS toggle infra už hotová.
- Audio cleanup, cancel, lang detection, lang lock — vše v `turn` endpointu.

### Frontend `static/app.js`
- `runTurn()` posílá do `/api/turn`, parsuje NDJSON.
- `state.messages` = conversation history (user + assistant).
- `addMessage()`, `persistMessages()` — DOM + localStorage.
- **Nutno doplnit:** `tool_call` / `tool_result` events, `approval_required` event,
  collapsible tool karty, mode toggle.

### Existující kód pro inspiraci
- `~/git/github/avatar-engine/avatar_engine/bridges/claude.py` — persistentní
  Claude Code subprocess s stream-json. Pattern zjednodušíme: jeden session per
  browser tab, lifecycle vázaný na turn id.

### OpenHue audit
- `/usr/bin/openhue` v0.23-1 (AUR) nainstalován, `~/.openhue/config.yaml` má bridge+key.
- Binárka teď reportuje „not configured yet" i pro `--help` — pravděpodobně chybí
  ENV proměnná nebo flag. **Před fází 5 ověřit `openhue setup` + `openhue get lights`.**
- MCP support (`openhue mcp`) přidán v novějším release — možná naše verze nemá.
- **Cesta: CLI wrapper jako MVP**, MCP adapter pak za stejné Tool interface.

## 4. Architektura (revised)

```
voice/
├── webapp/
│   ├── server.py          [edit: /api/turn rozšířen o mode + agent runner + approval events]
│   └── static/
│       ├── app.js         [edit: mode toggle, tool cards, approval modal, voice approval STT]
│       ├── style.css      [edit: tool card + modal styly]
│       └── index.html     [edit: mode toggle, modal markup]
├── agent/                 [NEW]
│   ├── __init__.py
│   ├── config.py          # WORKDIR, max_turns, output_cap, claude_unrestricted flag
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py        # Tool dataclass, ToolRegistry, JSON schema helpers
│   │   ├── fs.py          # list_files, read_file, write_file, edit_file, glob, grep
│   │   ├── shell.py       # run_bash s allowlistem
│   │   ├── web.py         # fetch_url, web_search (SKILL)
│   │   └── home.py        # OpenHue CLI wrapper (Plán B první)
│   ├── permissions.py     # decide(tool_name, args) -> AUTO|ASK|DENY (path-aware)
│   ├── loop.py            # agent loop: ollama → tool_calls → exec → feedback
│   ├── claude_bridge.py   # persistent `claude -p`, per-session
│   ├── router.py          # explicit Claude routing detect (regex)
│   └── messages.py        # conversation history schema (tool_call/tool_result roles)
└── ...
```

**Klíčový princip dle codex:** `approval` NEDÁVÁM do `agent/` jako modul.
Approval orchestrace je odpovědnost `/api/turn` handleru — `permissions.decide()`
vrátí ASK, handler emituje `approval_required` NDJSON event, čeká na response
z frontendu (přes pollable endpoint nebo přes WebSocket? — viz otevřená otázka).

### Conversation history schema (codex point: navrhnout HNED v fázi 1)

```python
# Message types stored client-side + sent to server
{"role": "user", "content": "rozsviť světla"}
{"role": "assistant", "content": "Jistě, zapínám…", "tool_calls": [
    {"id": "tc_1", "name": "set_room", "args": {"room": "obyvak", "on": true}}
]}
{"role": "tool", "tool_call_id": "tc_1", "content": "{\"ok\": true, \"affected\": 4}"}
{"role": "assistant", "content": "Hotovo, čtyři světla v obýváku."}
```

UI tool karta = `tool_calls` + matching `tool` message render do jednoho boxu.

### Datový tok (revised)

```
POST /api/turn { mode: "chat"|"agent"|"claude", messages: [...], model, ... }
  │
  ├── mode == "claude" (explicit routing detected on FE nebo regex zde)
  │    └── claude_bridge.stream() ──► NDJSON text events ──► TTS chunker
  │
  ├── mode == "agent"
  │    └── loop.run():
  │         while turn < MAX_TURNS:
  │            ollama.chat(messages, tools=[ALL])
  │            for tc in response.tool_calls:
  │               perm = permissions.decide(tc)
  │               if perm == ASK:
  │                  emit("approval_required", {id, summary, risk})
  │                  await approval_response()  ──► voice STT nebo UI button
  │               if perm == DENY or rejected: result = "denied by user"
  │               else: result = tool.execute(tc.args)
  │               emit("tool_call", tc); emit("tool_result", result)
  │               messages.append(tool_message)
  │            if no tool_calls: emit text deltas → done
  │
  └── mode == "chat" (default)
       └── loop.run(max_tool_rounds=1, tools=SKILLS only)
            (jinak stejné jako agent, ale jen 1 round)
```

## 5. Fázový plán (revised, 9 fází)

### Fáze 1 — Tool-calling infrastructure + conversation schema

- [ ] `voice/agent/tools/base.py`: `Tool(name, description, parameters_schema, execute_fn)`.
- [ ] `voice/agent/messages.py`: schema pro `tool_call` / `tool_result` v history.
- [ ] `voice/agent/permissions.py`: `decide(tool, args, workdir) -> AUTO|ASK|DENY`.
- [ ] `voice/agent/loop.py`: agent loop s `max_turns`, `max_wall_time`, `output_cap`.
- [ ] `voice/agent/config.py`: WORKDIR detection, env config (CLAUDE_UNRESTRICTED).
- [ ] Testovací tool `echo(text)` — bez side-effects.
- [ ] **Rozšířit `/api/turn`** o `mode` parametr a agent runner větev.
- [ ] NDJSON events: `tool_call`, `tool_result`, `approval_required`, `approval_response` (FE→BE).
- [ ] Frontend: mode toggle (chat/agent), tool karty (collapsible) v transkriptu.
- [ ] Frontend: `state.messages` rozšíření o `tool_calls` + `tool` role.

**DoD:** „Echo: hello" v agent mode → vidím collapsible tool kartu s call+result.
Conversation history obsahuje `tool_call` + `tool_result` messages. CHAT mode beze
změny chování (jen prep work pro fázi 4).

### Fáze 2 — File-ops tooly + sandbox bezpečnost

- [ ] `tools/fs.py`: `list_files`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`.
- [ ] Path normalization: `resolve()`, kontrola na symlinky.
- [ ] Reject special files (`/proc`, `/dev`, sockety, fifa, block devices).
- [ ] Read size limit 256 KiB + offset/limit pro velké soubory.
- [ ] Read mimo WORKDIR: allowlist (`/etc/os-release`, `/etc/lsb-release`,
      `/proc/cpuinfo`, `/proc/meminfo`, `/proc/version`, `/etc/hostname`) = AUTO,
      zbytek = ASK.
- [ ] Write/edit uvnitř WORKDIR = AUTO; mimo = ASK.
- [ ] Unit testy: symlink escape, `../../etc/passwd` traversal, `/proc/self/environ` reject.

**DoD:** Read v WORKDIR auto, read v `~` ask. Symlink na `/etc/shadow` odmítnut.

### Fáze 3 — Shell + approval flow (UI canonical, voice optional)

- [ ] `tools/shell.py`: `run_bash(command)` s **allowlistem** pro AUTO:
      `pwd`, `ls`, `rg`, `git status/diff/log/show`, `find` bez `-delete/-exec`,
      `cat` s resolved path check.
      Vše ostatní = ASK.
- [ ] Subprocess: `cwd=WORKDIR`, timeout (default 30s), stdout/stderr cap (1 MiB),
      `env` scrub (žádné cred env vars dovnitř).
- [ ] Server emit `{"type":"approval_required", "id":"...", "tool":"run_bash",
      "summary":"rm src/test.py", "risk":"file deletion"}`.
- [ ] Frontend modal vždy zobrazit (canonical), Allow/Deny tlačítka.
- [ ] Voice mode = navíc TTS čte summary + spustí STT 5s capture.
- [ ] STT regex: `\b(ano|jo|jasn[ěe]|povol|fajn|ok)\b` = approve;
      `\b(ne|stop|zruš|nechci)\b` = deny; jinak = deny (timeout).
- [ ] Destructive tooly (write outside WORKDIR, rm/destroy patterns) → vyžaduje
      explicit „ano povoluju" frázi (codex: STT nespolehlivá pro 1 slovo).
- [ ] **Bez voice timeoutu**: voice STT běží neomezeně, modal blokuje turn dokud
      nepřijde answer (voice nebo UI klik). User si může v klidu rozmyslet.
- [ ] Během approval: mic input pauznutý pro běžný transcript, frontend nesubmituje
      nový turn. Druhý mic kanál naslouchá jen approve/deny frázím.

**DoD:** Allowlist příkazy běží bez ask, ostatní vyvolají modal/voice. TTS off
mode = jen modal funguje.

### Fáze 4 — Skills: web_search + fetch_url (chat mode!)

- [ ] `tools/web.py`: `fetch_url(url)` (httpx + BeautifulSoup), `web_search(query)`.
- [ ] Web search backend: **Brave Search API** (free tier 2000/měsíc).
      Env var `BRAVE_SEARCH_API_KEY`. Pokud chybí klíč → tool nedostupný,
      do system promptu se nepřidá.
- [ ] Skill registry: `SKILLS = ["fetch_url", "web_search"]`.
- [ ] CHAT mode (`max_tool_rounds=1`): tools=SKILLS, system prompt extend.
- [ ] Tool calls v chat ukládat do history (codex: jinak další turn nezná zdroj).

**DoD:** V chat mode „kolik je 47. prezident USA, najdi to" → web_search →
final text. Funguje s TTS on i off. Po refresh: tool karty viditelné v historii.

### Fáze 5 — OpenHue (CLI wrapper první, MCP odložit)

- [ ] **Pre-flight:** ověřit `openhue get lights` na současné verzi. Pokud
      nefunguje, upgrade nebo dokumentovat.
- [ ] `tools/home.py`: 6 CLI wrapper toolů (`list_lights`, `list_rooms`,
      `list_scenes`, `set_light`, `set_room`, `activate_scene`).
- [ ] Permissions: smart home = AUTO (není destructive na systému).
- [ ] (Pozdější fáze, mimo MVP) MCP adapter za stejné `Tool` interface.

**DoD:** „Rozsviť světla v obýváku" → openhue set room → světla svítí.

### Fáze 6 — Claude sub-agent (MAX plan)

- [ ] `claude_bridge.py`: per-session subprocess `claude -p --input-format
      stream-json --output-format stream-json --verbose --include-partial-messages`.
- [ ] Args: `--add-dir WORKDIR`, `--permission-mode acceptEdits`,
      `--model claude-opus-4-7` (default), temp `--settings` (sandbox).
      Volitelný switch v UI: Opus 4.7 ↔ Sonnet 4.6 per session.
- [ ] **Nikdy** `--dangerously-skip-permissions` (pokud user neexplicit override).
- [ ] Env override `CLAUDE_UNRESTRICTED=1` → bez `--add-dir`.
- [ ] Stderr drain async task (codex: jinak proces zamrzne).
- [ ] Per-session bridge mapping (session_id → subprocess).
- [ ] Reload tolerance: pokračující turn dokončí, jen odpojí consumer.
- [ ] Cancel endpoint: `POST /api/turn/:id/cancel` posílá interrupt do subprocess.
- [ ] Log: session_id, cost/usage z stream-json eventů.
- [ ] Stream parser: text deltas → TTS chunker (existující), tool eventy → UI cards.

**DoD:** „Naplánuj refaktor X přes Claude" → Claude streamuje plán → TTS čte.
Reload page during stream → backend dokončí, žádný zombie process.

### Fáze 7 — Router + mode switching

- [ ] `router.py`: regex pre-filter pro user message první větu.
  - `přepni do agenta|agentní (mode|režim)|spusť agenta` → mode=AGENT
  - `chat (mode|režim)|vrať se do chatu|normální (mode|režim)` → mode=CHAT
  - `přes (claude|opus\w*|sonnet)|claude mode` → explicit Claude routing
- [ ] Mode persistence: frontend `localStorage` (server stateless, codex).
- [ ] Default po refresh = CHAT (bezpečnost).
- [ ] UI: badge ukazuje aktivní mode + visual cue (color border? subtle).

**DoD:** Hlasem nebo textem „přepni do agenta" → mode změněn (FE updates badge,
další turn jede agent runner).

### Fáze 8 — **Reliability** (codex nová fáze)

- [ ] Cancellation napříč celým loopem: agent loop, běžící tool subprocesses,
      Claude bridge, MCP klienti (až přijde).
- [ ] `max_turns` = **bez limitu** (user volba). Místo limitu = `max_wall_time`
      (default 10 min) jako pojistka proti runaway. `max_output_bytes` =
      4 MiB per tool result.
- [ ] Structured error recovery: tool error → vrátí strukturovaný error
      tool_result modelu, loop pokračuje (nepadá celý turn).
- [ ] Test suite: symlink traversal, path escape, bash allowlist coverage,
      approval timeout, max_turns enforcement.
- [ ] Audit log: každý tool call → `webapp.log` s timestamp + args + outcome.

**DoD:** Stop tlačítko zabije běžící bash subprocess + Claude stream + agent
loop atomicky. Tool error nezastaví turn (loop pokračuje, model dostane chybu).

### Fáze 9 — Polish

- [ ] Acoustic mode beep (krátký tón při hlasovém přepnutí mode).
- [ ] Barge-in: během dlouhých tool exekucí uživatel může přerušit hlasem.
- [ ] Audio filler („moment, hledám…") během tool exekucí > 2s.
- [ ] Auto-degradace: po destruktivní akci na 5 min default = ASK (i pro
      allowlist příkazy).

## 6. Finální rozhodnutí (user sign-off)

| # | Otázka | Rozhodnutí |
|---|--------|-----------|
| 1 | Read mimo WORKDIR | **Allowlist veřejných cest = AUTO**, zbytek = ASK. Allowlist: `/etc/os-release`, `/etc/lsb-release`, `/proc/cpuinfo`, `/proc/meminfo`, `/proc/version`, `/etc/hostname` |
| 2 | Web search backend | **Brave Search API** (free tier 2000/měsíc). Klíč v env `BRAVE_SEARCH_API_KEY`. User musí dodat klíč před fází 4. |
| 3 | Claude default model | **Opus 4.7** (`claude-opus-4-7`). UI switch na Sonnet 4.6 per session. |
| 4 | Approval voice fráze | Běžné = „ano"/„jo"/„ok"/„povol"/„jasně". **Destruktivní = vyžadovat „ano povoluju"** explicit. |
| 5 | Approval timeout | **Bez timeoutu**. Modal blokuje turn dokud user neresolve (UI klik nebo voice). |
| 6 | `max_turns` | **Bez limitu**. Místo toho `max_wall_time` = 10 min jako pojistka. |

Před fází 4 (web search) potřebuju **BRAVE_SEARCH_API_KEY**. User si ho založí na
[brave.com/search/api](https://brave.com/search/api/) (free tier).

## 7. Top 3 rizika (codex)

1. **Bash sandbox falešně bezpečný** — řešíme allowlistem pro AUTO, ne
   blacklistem pro ASK. Default je ASK.
2. **Duplicitní endpoint** — řešíme rozšířením `/api/turn`, ne novým.
3. **Tool history fragmentace** — schéma fázi 1, jeden zdroj pravdy.

## 8. Test plan

| Test | Fáze | Co ověřuje |
|------|------|-----------|
| `test_permissions.py::test_workdir_paths` | 2 | Write uvnitř WORKDIR = AUTO |
| `test_permissions.py::test_outside_read_asks` | 2 | Read mimo WORKDIR = ASK |
| `test_permissions.py::test_symlink_escape` | 2 | Symlink mimo WORKDIR = ASK |
| `test_permissions.py::test_special_files_reject` | 2 | /proc/self/environ = DENY |
| `test_permissions.py::test_bash_allowlist` | 3 | `ls` AUTO, `rm` ASK |
| `test_permissions.py::test_bash_redirect_asks` | 3 | `cmd > file` = ASK |
| `test_agent_loop.py::test_echo_tool` | 1 | Loop end-to-end |
| `test_agent_loop.py::test_max_turns` | 8 | Strop respektován |
| `test_agent_loop.py::test_tool_error_continues` | 8 | Tool error ≠ crash |
| `test_router.py::test_claude_routing` | 7 | „přes Claude" → claude branch |
| `test_router.py::test_mode_triggers` | 7 | „přepni do agenta" → AGENT |
| `test_skills_chat.py::test_websearch_in_chat` | 4 | Skill v chat mode |
| `test_claude_bridge.py::test_persistent_warmup` | 6 | 2. zpráva ne cold-start |
| `test_claude_bridge.py::test_stderr_drain` | 6 | Žádný hang při stderr fill |
| `test_messages.py::test_history_with_tool_calls` | 1 | Schema kompatibilní s Ollama |
