# Gemma

Lokální český hlasový asistent s agentním režimem + Claude Code jako sub-mode. LLM přes Ollama, STT přes whisper.cpp, TTS přes Chatterbox. Vše běží na jednom GPU.

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-13.2-green.svg" alt="CUDA 13.2">
  <img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti-76b900.svg" alt="RTX 5070 Ti">
  <img src="https://img.shields.io/badge/lang-cs%20%2B%20en-red.svg" alt="CS + EN">
</p>

## Co umí

Webová aplikace, ve které mluvíš na mikrofon (nebo píšeš), model ti odpoví hlasem + textem. Tři režimy přepínané v UI:

- **chat** klasická konverzace, bez toolů. TTS po větách during streamingu.
- **agent** model má 13 nástrojů (soubory, shell, web, Philips Hue, Claude consult). Klasifikátor rozhoduje AUTO / ASK / DENY. Approval modalem nebo hlasem.
- **claude** přímý dialog s Claude Code CLI přes vlastní adapter library. Print mode (per-turn subprocess) nebo Tmux mode (persistent session, kontext napříč turny). Workdir sandbox + claude vlastní `acceptEdits` permission policy. Volba modelu (Opus / Sonnet / Haiku) v UI.

## Nástroje v agent módu

| Tool | Co dělá |
|------|---------|
| `read_file` | Přečte soubor s čísly řádků (1-indexed, paged přes offset/limit) |
| `list_files` | Vypíše adresář |
| `glob` | Najde soubory podle vzoru (`**/*.py`) |
| `grep` | Hledá regex v souborech (ripgrep když je k dispozici) |
| `write_file` | Atomicky zapíše. Mimo workdir vyžaduje destruktivní frázi |
| `edit_file` | Přepíše přesný substring (single-match) |
| `run_bash` | Pustí shell příkaz. Allowlist (`ls`, `git status`, `cat`, …) auto, jinak ptá |
| `fetch_url` | HTTP GET. Blokuje RFC1918 + loopback + redirecty (SSRF) |
| `web_search` | Brave Search API |
| `light_list` | Vypíše Philips Hue světla |
| `light_set` | Změní stav světla (jméno, barva, jas) |
| `ask_claude` | Consult-only Claude Code subprocess (bez FS, bez shellu, jen vrátí text) |
| `echo` | Test pipeline |

## Claude mode

Samostatný režim oddělený od `ask_claude` toolu. Místo jednorázového consultu ti dá **plnohodnotného Claude Code agenta** s workdir editací.

| Vlastnost | Detail |
|-----------|--------|
| Sandbox | `--add-dir WORKDIR`, claude se nedostane ven |
| Permission policy | claude vlastní `--permission-mode acceptEdits` (sám blokuje destruktivní shell) |
| Session continuity | `claude_session_id` se ukládá do `~/.local/state/gemma/claude_ui_state.json` (mimo workdir, aby agent mode nemohl session_id přepsat) |
| Model switch | Volba Opus / Sonnet / Haiku v UI. Při změně se stávající session zabije a spawne se nová (model je immutable per claude process) |
| Reset | Klik na session badge → kill tmux session + clear state |
| Adapter | Default `print` (per-turn `claude -p` subprocess s `--output-format stream-json`). Opt-in `tmux` přes `AGENT_CLAUDE_BRIDGE_MODE=tmux` (persistent tmux session, kontext drží napříč turny, parsing TUI přes pyte) |

Žádná textová approval fráze (`ano povoluju`) v claude módu už není — vstup do režimu = implicit consent. Pokud někdy přibude friction pro nevratné operace, půjde to přes UI modal Allow/Deny, ne text prefix.

Audit a observability: každý turn loguje `claude emit stage=… count=N`, `_read_stream START/EOF/RESULT`, `claude gen YIELD #N`. Diagnostika progress eventů je na INFO leveli per design (silent failures = mrtvé UI, viz [feedback memory](#dev-notes)).

## Bezpečnost agenta

Každý tool má klasifikátor, který vrátí jedno ze tří:

- **AUTO** projde bez ptaní (čtení v workdir, `git status`, `ls`, …).
- **ASK** UI ukáže modal s tlačítky Allow/Deny. Lze odpovědět hlasem, kliknutím nebo napsat frázi.
- **DENY** zamítnuto okamžitě (cesty mimo workdir, `/proc`, `/sys`, syntax error, SSRF target, …).

Destruktivní akce v agent módu (`rm`, `sudo`, `chmod`, write mimo workdir) vyžadují **explicitní frázi „ano povoluju"** — to platí pro AGENT mode (voice fallback k modal kliknutí). V claude módu žádná fráze není.

Match na frázi je přísný: rovnost na normalizovaném textu, žádný substring/contains (chrání proti „tak jsem řekl ano povoluju nikdy"). V konfliktu vyhrává deny.

Každý tool call jde do append-only JSONL audit logu (decision, args hash, duration, výsledek). Adresář přes `AGENT_AUDIT_DIR`.

## TTS v agent + claude módu

Default `tts_scope=final`: nahlas se přečte jen finální odpověď. Mezikola jsou ticho; během dlouhých toolů hraje krátký filler („moment...") přes Web Speech API. Lze přepnout na `off`.

Code bloky se nikdy nečtou nahlas. Sentence chunker je vyseparuje a pošle do UI jako `kind: "code", speakable: false`. System prompt to modelu explicitně dovoluje: žádný markdown styling, ale ` ```jazyk … ``` ` fences jsou OK (a u `read_file` výstupu má strip-nout line-number prefixy `     1\t…`).

Před TTS synth se uvolní Ollama LLM z VRAM (`keep_alive=0` na `/api/generate`). Bez toho gemma4-26b (10 GB) + Chatterbox TTS (3 GB) přetlačí 16 GB RTX 5070 Ti při activations peak a první synth OOM-ne. Cena: další turn re-loadne LLM (3-5 s).

## Spuštění

```bash
# Ollama už musí běžet (systemd)
ollama create gemma4-e4b-32k -f modelfiles/gemma4-e4b-32k.Modelfile

# Spouštěč. WORKDIR = aktuální adresář (sandbox root pro agenta).
./scripts/agent.sh

# Nebo symlink do PATH a pak `gemma` z libovolného PWD:
ln -s $PWD/scripts/gemma ~/bin/gemma
gemma                # port 8080
gemma --port 9000
gemma --dangerous    # ASK -> AUTO (destructive stále vyžaduje frázi)

# Claude mode v tmux adapteru (persistent sessions):
AGENT_CLAUDE_BRIDGE_MODE=tmux gemma
```

Webapp na `http://127.0.0.1:8080`. Skript sám detekuje a zabije starou zombie webapp na portu (`/proc/$pid/stat` race-guard přes starttime). Cizí proces na portu odmítne s chybou.

## Externí služby

**Claude mode + `ask_claude` tool** spawnou `claude` CLI binary. Auth si CLI řeší sám: pokud máš `claude auth login` udělaný (OAuth do keychain), funguje. Pokud preferuješ API klíč, dej `ANTHROPIC_API_KEY` do env. Server žádný klíč sám nenastavuje, jen pasivně pasuje env do subprocesu. CLI binary lze přepsat přes `CLAUDE_CLI_BIN` (default `claude`).

- `ask_claude` (agent tool) — consult-only, `--permission-mode plan --tools ""` v prázdném temp dir, bez FS/shellu, jen vrátí text. Model default `claude-opus-4-7`, override `AGENT_CLAUDE_MODEL`.
- claude mode — full bridge přes `src/claude_bridge/` package (viz [Struktura](#struktura)). Edit mode v WORKDIR, full tools, session continuity.

Od 2026-06-15 jede `claude -p` (= náš bridge) z separátního Anthropic Agent SDK quotu (≠ hlavní Max limit), takže intensive agent usage neukousává Max plán.

**`web_search`** volá Brave Search API přímo (REST), klíč potřebuje. Účet zdarma má 2000 dotazů měsíčně:

```bash
# Env var:
export BRAVE_SEARCH_API_KEY="BSA-..."

# Nebo soubor 0600 (server ho přečte při startu):
echo "BSA-..." > ~/.brave-search-api
chmod 600 ~/.brave-search-api
```

Server odmítne načíst soubor, který je world/group readable (`mode & 0o077`). Override cesty přes `BRAVE_SEARCH_API_KEY_FILE`.

**Philips Hue** (`light_list`, `light_set`) — bridge IP a app key v env, viz `voice/agent/tools/hue.py` pro detail.

## Modely

| Tag | Velikost | Kontext | Poznámka |
|-----|----------|---------|----------|
| `gemma4-e4b-32k` | 9.6 GB | 32K | default, rychlé |
| `gemma4-26b-32k` | 17 GB | 32K | MoE, dobré na kód |
| `gemma4-31b-8k` | 20 GB | 8K | nejvyšší kvalita, CPU offload |
| `gemma4-31b-gguf` | 18.8 GB | 8K | Unsloth Dynamic 2.0 quant |
| `qwen3-14b-32k` | 9.3 GB | 32K | fallback, tool calling kompatibilní |
| `gemma3-12b-32k` | 8 GB | 32K | starší, **bez tool calling** (nepoužívat v agent módu) |

Všechny Modelfiles jsou v `modelfiles/`. Default v UI je `gemma4-26b-32k`.

V Claude módu jsou k dispozici tři modely Anthropic přes adapter: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`.

## Hardware

RTX 5070 Ti (16 GB, Blackwell SM_120, CC 12.0), 64 GB RAM, CUDA 13.2, Arch Linux. Ollama 0.21+ (kvůli tool calling fixu v Gemma 4).

Pro tmux adapter v Claude módu: `tmux` ≥ 3.0 + Python `pyte` (oboje v `voice/.venv-tts`).

## Konfigurace Ollamy

V `/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_MODELS=/home/box/git/github/gemma/models/ollama"
Environment="OLLAMA_KEEP_ALIVE=4m"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
ProtectHome=no
ReadWritePaths=/home/box/git/github/gemma
```

## Struktura

```
gemma/
├── modelfiles/        Ollama Modelfile (gemma4 + qwen3 + gemma3)
├── plans/             Design dokumenty per feature/fáze
├── prompts/           System prompty
├── scripts/           gemma, agent.sh, chat_voice.sh, record_ref.sh
├── src/
│   └── claude_bridge/ Standalone adapter library (PrintMode + Tmux)
│       ├── claude_bridge/
│       │   ├── adapters/    print_mode.py, tmux_mode.py
│       │   ├── parsing/     stream_json.py, ansi.py, tui_state.py
│       │   ├── base.py      Protocol + ProgressCallback
│       │   ├── config.py    AdapterConfig + BridgeMode + factory
│       │   ├── exceptions.py
│       │   ├── progress.py  ProgressEvent + SessionInfo
│       │   └── result.py    ClaudeResult dataclass
│       └── tests/     unit (parser, adapters) + integration (live claude CLI)
├── voice/
│   ├── agent/         Agent loop, permissions, audit, 13 toolů
│   ├── webapp/        FastAPI + frontend (static/app.js, style.css)
│   └── *.wav          Voice reference
└── tests/             972 testů + integration suite
```

## Testy

```bash
# Default (mocked Ollama, mocked TTS, mocked claude CLI)
./voice/.venv-tts/bin/python -m pytest tests/ -m "not integration"

# Integration proti reálné Ollamě (musí běžet + tool-capable model)
./voice/.venv-tts/bin/python -m pytest tests/integration/ -m integration

# Claude bridge unit (parser + adapter contract)
./voice/.venv-tts/bin/python -m pytest src/claude_bridge/tests/unit/

# Claude bridge integration (live `claude` CLI - potřebuje OAuth nebo API key)
./voice/.venv-tts/bin/python -m pytest src/claude_bridge/tests/integration/
```

Mocked test suite ověří kontrakt s Ollamou (arguments jako dict, ne string, jinak HTTP 400 na další round-tripu). E2E testy běží proti reálnému uvicornu na náhodném portu, ne ASGITransport, protože ten nespouští lifespan + bufferuje streamy.

## Endpointy (zkrácený přehled)

| Endpoint | Metoda | Co dělá |
|----------|--------|---------|
| `/api/turn` | POST | NDJSON stream pro chat / agent / claude turn |
| `/api/turn/{tid}/approval/{ap_id}` | POST | Schválení agent tool callu |
| `/api/turn/{tid}/audio/{seq}.wav` | GET | TTS audio chunk |
| `/api/turn/{tid}/messages` | GET | Snapshot agent history po dokončení turn |
| `/api/claude_ui_state` | GET | Claude session state (session_id, model) |
| `/api/claude_ui_state/reset` | POST | Kill tmux session + clear state |
| `/api/approval_config` | GET | Approve/deny phrases pro frontend |
| `/api/models` | GET | Seznam Ollama modelů |
| `/api/voices` / `/api/refs` | GET | TTS voice/reference allowlist |
| `/api/health` | GET | Stav komponent (TTS, Ollama, CUDA, workdir) |
| `/api/client_log` | POST | Frontend defensive logging → server log |

## Dev notes

`tool_call.function.arguments` se interně drží jako **dict**, ne JSON string. Ollama native `/api/chat` chce objekt, string způsobí 400 „can't find closing '}' symbol" na druhém round-tripu po každém tool callu. (OpenAI API naopak chce string, takže pokud někdy přibude OpenAI backend, konverze patří do adapteru.)

Synth pipeline je sdílená mezi chat a agent módem. Chat ji volá per-sentence během streamingu, agent ji volá jednou po `agent_done` s celým finálním textem. Helper `_synth_chunk_and_emit` v `voice/webapp/server.py`.

Fráze pro approval (`APPROVE_PHRASES`, `DENY_PHRASES`, `DESTRUCTIVE_APPROVAL_PHRASE`) žijí v `voice/agent/config.py`. Frontend si je tahá přes `GET /api/approval_config`, fallback constants jsou v `app.js` jen pro init před prvním fetchem. Žádný drift.

V Claude módu nikdy nedávej `threading.Event` jako `cancel_event` do `adapter.ask` — bridge očekává `asyncio.Event`. `threading.Event.wait()` je sync-blocking call, který by zamrazil celý event loop. Server proto vyrábí per-turn asyncio mirror přes async polling loop (NE `run_in_executor(None, threading_ev.wait)` — executor thread není cancellable a po normálním dokončení turnu by zůstal navždy blokovaný → po pár desítkách turnů thread pool vyčerpán). Bridge má defensive `TypeError` guard.

`claude_session_id` write má **CAS** proti `expected_prior` aby reset endpoint mid-turn nepřepsal session_id zpět. SessionNotFound (stale ID po přepnutí adapter print↔tmux nebo restart serveru) má auto-recovery: clear + retry s `session_id=None` uvnitř téhož `_CLAUDE_TURN_LOCK`.

## Licence

MIT.
