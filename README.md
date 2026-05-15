# Gemma

Lokální český hlasový asistent s agentním režimem. LLM přes Ollama, STT přes whisper.cpp, TTS přes Chatterbox. Vše běží na jednom GPU.

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-13.2-green.svg" alt="CUDA 13.2">
  <img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti-76b900.svg" alt="RTX 5070 Ti">
  <img src="https://img.shields.io/badge/lang-cs%20%2B%20en-red.svg" alt="CS + EN">
</p>

## Co umí

Webová aplikace ve které mluvíš na mikrofon (nebo píšeš), model ti odpoví hlasem. Tři režimy:

- **chat** klasická konverzace, žádné tooly
- **agent** model má 13 nástrojů (soubory, shell, web, Philips Hue, Claude API). Před destruktivními akcemi se ptá. Schvaluje se kliknutím, napsáním fráze, nebo hlasem.
- **claude** přepošle dotaz na Anthropic Claude API (pokud máš `ANTHROPIC_API_KEY`).

## Nástroje v agent módu

| Tool | Co dělá |
|------|---------|
| `read_file` | Přečte soubor s čísly řádků |
| `list_files` | Vypíše adresář |
| `glob` | Najde soubory podle vzoru (`**/*.py`) |
| `grep` | Hledá regex v souborech (ripgrep když je) |
| `write_file` | Atomicky zapíše. Mimo workdir vyžaduje frázi |
| `edit_file` | Přepíše přesný substring |
| `run_bash` | Pustí shell příkaz. Bezpečné (`ls`, `git status`) auto, jinak ptá |
| `fetch_url` | HTTP GET. Blokuje RFC1918 a loopback (SSRF) |
| `web_search` | Brave Search API |
| `light_list` | Vypíše Hue světla |
| `light_set` | Změní stav světla (jméno, barva, jas) |
| `ask_claude` | Delegace na Claude API |
| `echo` | Test pipeline |

## Bezpečnost agenta

Každý tool má klasifikátor který vrátí jedno ze tří:

- **AUTO** projde bez ptaní (čtení v workdir, `git status`, ...).
- **ASK** UI ukáže modal s tlačítky Allow/Deny. Lze taky odpovědět hlasem nebo napsat.
- **DENY** zamítnuto okamžitě (cesty mimo workdir, `/proc`, `/sys`, syntax error).

Destruktivní akce (`rm`, `sudo`, `chmod`, write mimo workdir) vyžadují **explicitní frázi „ano povoluju"**. Stačí ji říct hlasem, modal má vlastní 🎤 tlačítko (hlavní mikrofon je v té chvíli inert kvůli `<dialog>::showModal()`). Match je přísný: čistá rovnost na normalizovaném textu, žádné „tak jsem řekl ano povoluju nikdy" false positive. V konfliktu vyhrává deny.

Každý tool call jde do append-only JSONL audit logu (decision, args hash, duration, výsledek).

## TTS v agent módu

Default `tts_scope=final`: nahlas se přečte jen finální odpověď po všech toolech. Mezikola jsou ticho, během dlouhých toolů hraje krátký filler („moment...") přes Web Speech API. Lze přepnout na `off`.

Code bloky se nikdy nečtou nahlas. Sentence chunker je vyseparuje a pošle do UI jako text-only chunk.

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
```

Webapp na `http://127.0.0.1:8080`. Skript sám detekuje a zabije starou zombie webapp na portu (`/proc/$pid/stat` race-guard přes starttime). Cizí proces na portu odmítne s chybou.

## Modely

| Tag | Velikost | Kontext | Poznámka |
|-----|----------|---------|----------|
| `gemma4-e4b-32k` | 9.6 GB | 32K | default, rychlé |
| `gemma4-26b-32k` | 17 GB | 32K | MoE, dobré na kód |
| `gemma4-31b-8k` | 20 GB | 8K | nejvyšší kvalita, CPU offload |
| `gemma4-31b-gguf` | 18.8 GB | 8K | Unsloth Dynamic 2.0 quant |
| `qwen3-14b-32k` | 9.3 GB | 32K | fallback, tool calling kompatibilní |

Všechny Modelfiles jsou v `modelfiles/`.

## Hardware

RTX 5070 Ti (16 GB, Blackwell SM_120, CC 12.0), 64 GB RAM, CUDA 13.2, Arch Linux. Ollama 0.21+ (kvůli tool calling fixu v Gemma 4).

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
├── modelfiles/        Ollama Modelfile (gemma4 + qwen3)
├── plans/             Design dokumenty per feature
├── prompts/           System prompty
├── scripts/           gemma, agent.sh
├── voice/
│   ├── agent/         Agent loop, permissions, audit, 13 toolů
│   ├── webapp/        FastAPI + frontend
│   └── *.wav          Voice reference
└── tests/             852 testů + integration suite
```

## Testy

```bash
# Default (mocked Ollama, mocked TTS)
./voice/.venv-tts/bin/python -m pytest tests/ -m "not integration"

# Integration proti reálné Ollamě (musí běžet + tool-capable model)
./voice/.venv-tts/bin/python -m pytest tests/integration/ -m integration
```

Mocked test suite ověří kontrakt s Ollamou (arguments jako dict, ne string, jinak HTTP 400 na další round-tripu). E2E testy běží proti reálnému uvicornu na náhodném portu, ne ASGITransport, protože ten bufferuje streamy.

## Pár věcí které je dobré vědět

`tool_call.function.arguments` se interně drží jako **dict**, ne JSON string. Ollama native `/api/chat` chce objekt, string způsobí 400 „can't find closing '}' symbol" na druhém round-tripu po každém tool callu. (OpenAI API naopak chce string, takže pokud někdy přibude OpenAI backend, konverze patří do adapteru.)

Synth pipeline je sdílená mezi chat a agent módem. Chat ji volá per-sentence během streamingu, agent ji volá jednou po `agent_done` s celým finálním textem. Helper `_synth_chunk_and_emit` v `voice/webapp/server.py`.

Fráze pro approval (APPROVE_PHRASES, DENY_PHRASES, DESTRUCTIVE_APPROVAL_PHRASE) žijí v `voice/agent/config.py`. Frontend si je tahá přes `GET /api/approval_config`, fallback constants jsou v `app.js` jen pro init před prvním fetchem. Žádný drift.

## Plány

- `plans/voice_chat_webapp.md` voice webapp
- `plans/agent_mode.md` agent mode s 13 tools a voice approval
- `plans/bilingual_tts.md` cs/en TTS swap
- `plans/text_input.md` textový vstup, mode toggle

Top-level baseline v `PLAN.md`.

## Licence

MIT.
