# Gemma — Lokální český hlasový asistent + agent

**Plně lokální stack pro hlasový chat **i agentní akce** v češtině: Gemma 4 (LLM) + whisper.cpp (STT) + Chatterbox-TTS-Czech (TTS) + voice webapp s 13 nástroji pro práci se soubory, shellem, webem, Philips Hue a Claude API.**

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-13.2-green.svg" alt="CUDA 13.2">
  <img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti-76b900.svg" alt="RTX 5070 Ti">
  <img src="https://img.shields.io/badge/lang-čeština%20%2B%20english-red.svg" alt="CS + EN">
  <img src="https://img.shields.io/badge/tests-852%20passing-brightgreen.svg" alt="852 tests">
</p>

---

## Co to je

Čtyři vrstvy, vše běží lokálně bez cloudu (Claude bridge je volitelný):

| Vrstva | Komponenta | Účel |
|--------|-----------|------|
| **LLM** | Gemma 4 e4b / 26b / 31b přes Ollama (qwen3 fallback) | Konverzace + agentní úlohy + tool calling |
| **STT** | whisper.cpp (large-v3 + Silero VAD) | Mic → text |
| **TTS** | Chatterbox-TTS-Czech (per-věta streaming) | Text → hlas, CS i EN |
| **Webapp** | FastAPI + WebGL orb avatar | Mic ↔ LLM ↔ TTS pipeline, chat/agent/Claude režimy, 13 toolů |

## Featury

### 🤖 Agent mode (hotový)

Plnohodnotná tool-calling smyčka s permission gating, hlasovou approval frází a TTS výstupem na finální odpověď.

**13 toolů:**

| Tool | Použití |
|------|---------|
| `read_file`, `list_files`, `glob`, `grep` | Sandboxed FS read (cat-n format, line numbers, paging) |
| `write_file`, `edit_file` | Atomic write / exact-substring replace; mimo workdir vyžaduje „ano povoluju" |
| `run_bash` | Shell s root-cmd allowlist (AUTO) + ASK pro pipes/redirecty + destruktivní guard (`rm`/`sudo`/`chmod` vyžaduje frázi) |
| `fetch_url`, `web_search` | HTTP GET (SSRF defense: blokované RFC1918/loopback) + Brave Search API |
| `light_list`, `light_set` | Philips Hue smart-home (per-room/light state) |
| `ask_claude` | Bridge na Anthropic API pro tasky kde lokální LLM nestačí |
| `echo` | Test pipeline |

**Bezpečnost:**
- **Permission classifier per tool**: `AUTO` (safe + uvnitř workdir), `ASK` (modal approval), `DENY` (policy violation)
- **Destruktivní akce vyžadují frázi „ano povoluju"** — buď psanou v modalu, nebo **vyslovenou hlasem** přes Whisper STT
- **Sandbox** pro FS: path resolution + `is_inside_workdir` + `is_special_file` (drop `/proc/*`, `/sys/*`, `/dev/*`)
- **Wall-time bounded** (10 min cap), tool-call cap per turn
- **Append-only audit log** (JSONL) — každý tool call s permission decision, args hash, duration, outcome
- **Prompt injection defense** — tool výstupy jsou data, ne instrukce
- **Server-side history sanitization** — drop forged tool messages, canonicalize tool_call args na dict (Ollama spec)

**Agent TTS:**
- Konfigurovatelný `tts_scope`: `final` (default — jen finální odpověď po toolech) / `off`
- Code bloky **nikdy** nečteny nahlas (jdou jako `chunk` event v UI)
- Sdílený synth pipeline s chat módem (žádná duplikace) — `_synth_chunk_and_emit` helper
- Per-tool latency: audio filler („moment…") během dlouhých toolů via Web Speech API
- OOM-resistant: LLM se uvolní z VRAM před TTS synth (gemma4-26b 10 GB + Chatterbox 3 GB se na 16 GB RTX nepomestí současně při activations peak)

**Voice approval:**
- Tlačítko 🎤 přímo v approval modalu (kvůli `<dialog>` inert problému, mic v hlavním composeru je nedostupný)
- Whisper transcribe → `classifyApprovalUtterance(text, requiresExplicit)`:
  - **DENY priority** v konfliktu (safer)
  - Destructive = **strict equal match** (`"ano povoluju"` přesně, ne substring — žádné false positive z citací)
  - Non-destructive = intent (libovolná z `APPROVE_PHRASES`/`DENY_PHRASES`)
- Race guard: snapshot `{turnId, approvalId}` při startu nahrávky; mismatch při finishi → discard
- Server `/api/approval_config` = single source of truth pro fráze (žádný drift mezi server/client)

### 🎙️ Voice webapp (`voice/webapp/`)

- Hlasový a textový vstup s NDJSON streamingem
- Auto-VAD nahrávání (1.5 s ticho = stop) i push-to-talk
- TTS streaming po větách (`voice/sentence_chunker.py`)
- Stop tlačítko atomicky přeruší recording / LLM / TTS / tool execution
- Per-turn cancel + tmpdir, nezasahuje souběžné turny
- WebGL orb avatar reaguje na audio level
- Detekce jazyka + lang lock (po odpovědi se model drží zvoleného jazyka)
- Mode toggle (chat ↔ agent ↔ claude) hlasem („přepni do agentního módu") i tlačítkem
- Collapsible tool karty v transkriptu (status: running / OK / denied / error)

### 🧠 Modely (Ollama tagy)

| Tag | Velikost | Kontext | Použití |
|-----|----------|---------|---------|
| `gemma4-e4b-32k` | ~9.6 GB | 32K | **default**, rychlé, tool-calling, plně v VRAM |
| `gemma4-26b-32k` | ~17 GB | 32K | nejlepší kvalita pro programování, MoE |
| `gemma4-31b-8k` | ~20 GB | 8K | maximální kvalita, CPU offload |
| `gemma4-31b-gguf` | ~18.8 GB | 8K | Unsloth Dynamic 2.0 quant |
| `qwen3-14b-32k` | ~9.3 GB | 32K | fallback (tool-calling kompatibilní) |

Modelfiles jsou v [`modelfiles/`](modelfiles/).

## Hardware baseline

- NVIDIA RTX 5070 Ti (16 GB VRAM, Blackwell SM_120, CC 12.0)
- 64 GB RAM, CUDA 13.2, Arch Linux
- Ollama 0.21+ (Gemma 4 tool-calling fix nutný)

## Rychlý start

```bash
# 1) Naklonuj a připrav modely (Ollama už musí běžet jako systemd)
ollama create gemma4-e4b-32k  -f modelfiles/gemma4-e4b-32k.Modelfile
ollama create gemma4-26b-32k  -f modelfiles/gemma4-26b-32k.Modelfile
# (volitelně další varianty z modelfiles/)

# 2) Voice webapp — spouštěč
./scripts/agent.sh                # WORKDIR = $PWD, port 8080
./scripts/agent.sh --port 9000    # vlastní port
./scripts/agent.sh --dangerous    # ASK rozhodnutí → AUTO (destructive stále vyžaduje frázi)

# Wrapper z libovolného PWD (symlink do PATH):
ln -s $PWD/scripts/gemma ~/bin/gemma
gemma                             # WORKDIR = aktuální adresář, agent dostane sandbox root tady

# Skript sám detekuje a zabije starou zombie webapp na portu; cizí proces odmítne.
```

Webapp běží na `http://127.0.0.1:8080`. Default model `gemma4-e4b-32k`, výchozí TTS scope `final` (čte jen finální odpověď agenta po toolech).

## Systemd konfigurace (již nasazena)

Drop-in v `/etc/systemd/system/ollama.service.d/override.conf`:

```ini
[Service]
Environment="OLLAMA_MODELS=/home/box/git/github/gemma/models/ollama"
Environment="OLLAMA_KEEP_ALIVE=4m"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
ProtectHome=no
ReadWritePaths=/home/box/git/github/gemma
```

## Struktura repa

```
gemma/
├── modelfiles/              # Ollama Modelfile (gemma4 e4b/26b/31b + qwen3)
├── plans/                   # Design dokumenty per feature
│   ├── voice_chat_webapp.md
│   ├── agent_mode.md
│   ├── bilingual_tts.md
│   └── text_input.md
├── prompts/                 # System prompty (cs/en)
├── scripts/                 # gemma, agent.sh (s auto port-free), smoke_test.sh, …
├── voice/
│   ├── agent/               # Tool-calling: loop, permissions, messages, audit
│   │   ├── loop.py          # AgentLoop — main driver
│   │   ├── permissions.py   # AUTO/ASK/DENY classifier per tool
│   │   ├── messages.py      # tool_calls canonicalize (dict args pro Ollama)
│   │   ├── audit.py         # JSONL audit log
│   │   ├── router.py        # Pre-flight heuristic (claude/local/smart-home)
│   │   └── tools/           # 13 toolů (fs, shell, web, hue, claude, echo)
│   ├── webapp/              # FastAPI server + frontend
│   ├── sentence_chunker.py  # TTS sentence chunking (code blocky non-speakable)
│   └── *.wav                # TTS voice reference
├── tests/                   # pytest suite: 852 unit/E2E + integration proti reálné Ollamě
│   └── integration/         # `pytest -m integration` (vyžaduje běžící Ollama)
├── PLAN.md                  # Top-level baseline plán
└── README.md
```

## Testy

```bash
# Default (rychlé, ~17 s, mocked Ollama + TTS)
./voice/.venv-tts/bin/python -m pytest tests/ -m "not integration"

# Integration testy proti reálné Ollamě (vyžaduje běžící server + tool-capable model)
./voice/.venv-tts/bin/python -m pytest tests/integration/ -m integration
```

**852 testů** (unit + E2E přes reálný uvicorn server na náhodném portu — ne ASGITransport, ten bufferuje streamy a způsobuje hangy). TTS preload je v testech stubbován.

E2E pokrytí: agent loop (text/tool/approval/cancel), per-tool permission classification, audit log, full round-trips (write_file/run_bash/light_set/ask_claude), mocked Ollama validates arguments-as-dict kontrakt (regrese guard pro Ollama 400 bug).

## Klíčová architektonická rozhodnutí

- **`tool_call.function.arguments` je vždy DICT** v interní historii, ne JSON string. Ollama native `/api/chat` chce object; string způsobí HTTP 400 „can't find closing '}' symbol" při round-tripu po každém tool callu. (OpenAI Chat Completions naopak chce string — pokud někdy přibude OpenAI backend, konverze patří do adapteru.)
- **Sdílený TTS synth helper** mezi chat a agent módem (`_synth_chunk_and_emit`) — jedna funkce, dvě cesty volání (per-sentence streaming pro chat, one-shot final-only pro agent). Žádná duplikace.
- **Single source of truth pro fráze**: server endpoint `/api/approval_config` vrací `APPROVE_PHRASES`, `DENY_PHRASES`, `DESTRUCTIVE_APPROVAL_PHRASE` z `voice/agent/config.py`. Frontend má fallback constants jen pro init před prvním fetchem.
- **Codex review cyklus per fáze**: po každé větší změně review přes `codex exec` (a `gemini` když je dostupný), iterovat dokud critical/high = 0.

## Plány

- [`plans/voice_chat_webapp.md`](plans/voice_chat_webapp.md) — voice webapp (hotovo)
- [`plans/bilingual_tts.md`](plans/bilingual_tts.md) — bilingual TTS (hotovo)
- [`plans/text_input.md`](plans/text_input.md) — text input + mode toggle (hotovo)
- [`plans/agent_mode.md`](plans/agent_mode.md) — agent mode (hotovo, vč. TTS, voice approval, 13 toolů)

Top-level baseline + hardware audit → [`PLAN.md`](PLAN.md).

## Licence

MIT (viz [LICENSE](LICENSE), bude doplněn).
