# Gemma — Lokální český hlasový asistent

**Plně lokální stack pro hlasový chat v češtině: Gemma 3/4 (LLM) + whisper.cpp (STT) + Chatterbox-TTS-Czech (TTS) + voice webapp s agentním režimem.**

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+"></a>
  <img src="https://img.shields.io/badge/CUDA-13.2-green.svg" alt="CUDA 13.2">
  <img src="https://img.shields.io/badge/GPU-RTX%205070%20Ti-76b900.svg" alt="RTX 5070 Ti">
  <img src="https://img.shields.io/badge/lang-čeština%20%2B%20english-red.svg" alt="CS + EN">
</p>

---

## Co to je

Tři vrstvy pro běh AI úplně lokálně bez cloudu:

| Vrstva | Komponenta | Účel |
|--------|-----------|------|
| **LLM** | Gemma 3 e4b / 31b / 31b-gguf přes Ollama | Konverzace, agentní úlohy, tool calling |
| **STT** | whisper.cpp (large-v3 + Silero VAD) | Mic → text, primárně česky |
| **TTS** | Chatterbox-TTS-Czech | Text → hlas (CS), volitelně en |
| **Webapp** | FastAPI + WebGL orb avatar | Mic → LLM → TTS pipeline, chat/agent/Claude režimy |

## Hardware baseline

- NVIDIA RTX 5070 Ti (16 GB VRAM, Blackwell SM_120, CC 12.0)
- 64 GB RAM, CUDA 13.2, Arch Linux
- Ollama 0.21+ (Gemma 4 tool-calling fix nutný)

## Featury

### Voice webapp (`voice/webapp/`)
- Hlasový a textový vstup s NDJSON streamingem
- Auto-VAD nahrávání (1.5 s ticho = stop) i push-to-talk
- TTS streaming po větách (sentence chunker)
- Stop tlačítko atomicky přeruší recording / LLM / TTS
- Per-turn cancel + tmpdir, nezasahuje souběžné turny
- WebGL orb avatar reaguje na audio level
- Detekce jazyka + lang lock (po odpovědi se model drží zvoleného jazyka)

### Agent mode (work-in-progress)
- Tool-calling smyčka kolem Ollama `/api/chat`
- Permission gating: AUTO / ASK / DENY classifier per tool
- Approval round-trip (UI modal canonical, voice STT alternativní)
- Wall-time bounded (10 min cap na celý turn)
- Server-side history sanitization (drop forged tool messages)
- Prompt injection defense (tool outputs jsou data, ne instrukce)
- Destructive akce vyžadují explicit frázi „ano povoluju"
- NDJSON eventy: `tool_call`, `tool_result`, `approval_required`, `approval_response`
- Frontend: mode toggle (chat/agent/claude), collapsible tool karty v transkriptu
- Detaily implementace: [`plans/agent_mode.md`](plans/agent_mode.md)

### Modely (Ollama tagy)
| Tag | Velikost | Kontext | Použití |
|-----|----------|---------|---------|
| `gemma4-e4b-32k` | ~9.6 GB | 32K | default, rychlé, tool-calling |
| `gemma4-31b-8k` | ~20 GB | 8K | maximální kvalita, CPU offload |
| `gemma4-31b-gguf` | ~18.8 GB | 8K | Unsloth Dynamic 2.0 quant |

Modelfiles jsou v [`modelfiles/`](modelfiles/).

## Rychlý start

```bash
# 1) Naklonuj a připrav modely (Ollama už musí běžet)
ollama create gemma4-e4b-32k  -f modelfiles/gemma4-e4b-32k.Modelfile
ollama create gemma4-31b-8k   -f modelfiles/gemma4-31b-8k.Modelfile
ollama create gemma4-31b-gguf -f modelfiles/gemma4-31b-gguf.Modelfile

# 2) Smoke test
bash scripts/smoke_test.sh

# 3) Spusť voice webapp (venv musí být připraven, viz voice/webapp/README.md)
./voice/.venv-tts/bin/uvicorn voice.webapp.server:app --host 127.0.0.1 --port 8080
# Otevři http://127.0.0.1:8080

# 4) STT z příkazové řádky
./whisper.cpp/build/bin/whisper-cli \
  -m whisper.cpp/models/ggml-large-v3.bin \
  -vm whisper.cpp/models/ggml-silero-v6.2.0.bin --vad \
  -l cs -f tvuj_soubor.wav
```

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
├── modelfiles/           # Ollama Modelfile pro 3 varianty Gemmy
├── plans/                # Design dokumenty per feature
│   ├── voice_chat_webapp.md
│   ├── agent_mode.md     # ← aktuální velká feature
│   ├── bilingual_tts.md
│   └── text_input.md
├── prompts/              # System prompty (cs/en)
├── scripts/              # smoke_test.sh, chat_voice.sh, record_ref.sh
├── voice/
│   ├── agent/            # Tool-calling infra (loop, permissions, tools)
│   ├── webapp/           # FastAPI server + frontend
│   ├── sentence_chunker.py
│   └── *.wav             # TTS voice reference
├── tests/                # pytest suite (unit + E2E real uvicorn)
├── PLAN.md               # Top-level baseline plán
└── README.md             # Tento soubor
```

## Testy

```bash
./voice/.venv-tts/bin/python -m pytest tests/ --timeout=20
```

E2E testy běží proti reálnému uvicorn serveru na náhodném portu (ne ASGITransport — ten bufferuje streamy a způsobuje hangy). TTS preload je v testech stubbován.

## Plány

- **Voice webapp** → [`plans/voice_chat_webapp.md`](plans/voice_chat_webapp.md) (hotovo)
- **Bilingual TTS** → [`plans/bilingual_tts.md`](plans/bilingual_tts.md) (hotovo)
- **Text input** → [`plans/text_input.md`](plans/text_input.md) (hotovo)
- **Agent mode** → [`plans/agent_mode.md`](plans/agent_mode.md) (Fáze 1 hotová, 8 dalších v práci)

Top-level baseline + hardware audit → [`PLAN.md`](PLAN.md).

## Licence

MIT (viz [LICENSE](LICENSE), bude doplněn).
