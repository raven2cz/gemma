# Gemma 4 Local Stack — Implementation Plan

Cílem je v adresáři `/home/box/git/github/gemma` rozjet tři velikosti Gemma 4 přes
Ollama + opencode s prioritou češtiny a lokální hlasové zpracování (STT + TTS).

## 1. Hardware & software baseline

| Položka | Hodnota |
|---------|---------|
| GPU | NVIDIA RTX 5070 Ti, 16 GB VRAM, Blackwell **SM_120** (CC 12.0) |
| RAM | 64 GB |
| CUDA toolkit | **13.2** (`/opt/cuda`), `compute_120` je v seznamu arch |
| GCC | 15.2.1 (whisper.cpp CUDA build prošel OK) |
| Driver | 595.58.03 |
| Ollama | 0.21.0 (≥ 0.20.2 nutné pro Gemma 4 tool calling fix) |
| opencode | **1.4.7** (pacman `extra/opencode`) |
| Disk | po úklidu 225 GB volných v `/home` |
| Shell | bash, Arch Linux |

## 2. Tři modely

### 2.1 Malý & rychlý — **`gemma4:e4b` → `gemma4-e4b-32k`**
- Oficiální Ollama tag `gemma4:e4b` = ~9.6 GB (Q4_K_M, ne 3–4 GB jak se uvádělo
  v první verzi plánu).
- Odvozený Modelfile `modelfiles/gemma4-e4b-32k.Modelfile` s **`num_ctx 32768`**
  → tag `gemma4-e4b-32k`. Důvod: default 4K kontext zabíjí tool calling v opencode.
- Multimodální (text + obraz), podpora 140+ jazyků včetně češtiny.
- Použití: default model v opencode, rychlé autocomplete, levný agent.

### 2.2 Velký default Ollamy — **`gemma4:31b` → `gemma4-31b-8k`**
- Oficiální tag `gemma4:31b` = alias `gemma4:31b-it-q4_K_M`, **~20 GB**.
- Na 16 GB VRAM nepůjde celý na GPU → Ollama automaticky udělá **CPU offload**
  pro několik posledních layerů. Očekávaná rychlost **~3–7 tok/s**.
- Odvozený Modelfile `modelfiles/gemma4-31b-8k.Modelfile` s **`num_ctx 8192`**
  (256K default je nepoužitelný, KV cache by sežrala 10–22 GB navíc).
- Použití: velký model v opencode, maximální kvalita, akceptujeme pomalost.

### 2.3 Velký GGUF s laděním — **`gemma4-31b-gguf` (UD-Q4_K_XL)**
- Stahováno ručně z `unsloth/gemma-4-31B-it-GGUF`, konkrétně
  `gemma-4-31B-it-UD-Q4_K_XL.gguf` (~18.8 GB, Unsloth Dynamic 2.0 = lepší
  kvalita než běžný Q4_K_M při srovnatelné velikosti).
- Unsloth patchnul Gemma 4 chat template → tool calling s opencode funguje.
- Importováno přes Modelfile, laděn `num_gpu` pro co největší GPU offload.
- Google QAT Q4_0 GGUF repo pro Gemma 4 31B **neexistuje** (ověřeno), proto
  Unsloth UD-Q4_K_XL je nejlepší dostupná volba.
- Použití: alternativa k 2.2, potenciálně lepší kvalita/rychlost.

## 3. Context window — strategie per model

| Model | `num_ctx` | KV cache type | Důvod |
|-------|-----------|---------------|-------|
| `gemma4-e4b-32k` | **32 768** | fp16 (default) | Vejde se do VRAM, 32K stačí na agentní flow v opencode. |
| `gemma4-31b-8k` | **8 192** | `q8_0` (via env) | Váhy 20 GB už přesahují VRAM; agresivní KV úspora. |
| `gemma4-31b-gguf` | **8 192** | `q8_0`, později zkusit `q4_0` | Cílem je dostat maximum layerů na GPU. |

Globálně přes systemd override ollama.service:
```
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

## 4. Ollama systemd konfigurace (hotovo)

Drop-in `/etc/systemd/system/ollama.service.d/override.conf`:
```ini
[Service]
ProtectHome=no
ReadWritePaths=/home/box/git/github/gemma
Environment="OLLAMA_MODELS=/home/box/git/github/gemma/models/ollama"
Environment="OLLAMA_KEEP_ALIVE=4m"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
```
Zajištěno `chmod o+x` na `/home/box`, `/home/box/git`, `/home/box/git/github`,
`/home/box/git/github/gemma` (ollama user nemá read access, jen traverse).

**Keep-alive = 4 min**: model se unloadne 4 minuty po posledním požadavku,
VRAM se uvolní. Okamžité vyhození: `ollama stop <tag>`.

## 5. Pipeline pro stažení modelů

### 5.1 Ollama registry
```bash
ollama pull gemma4:e4b     # ~9.6 GB
ollama pull gemma4:31b     # ~20 GB, pomalý pull
ollama create gemma4-e4b-32k -f modelfiles/gemma4-e4b-32k.Modelfile
ollama create gemma4-31b-8k -f modelfiles/gemma4-31b-8k.Modelfile
```

### 5.2 Ruční GGUF (Unsloth)
```bash
hf download unsloth/gemma-4-31B-it-GGUF gemma-4-31B-it-UD-Q4_K_XL.gguf \
  --local-dir /home/box/git/github/gemma/gguf
```
Modelfile `modelfiles/gemma4-31b-gguf.Modelfile`:
```
FROM /home/box/git/github/gemma/gguf/gemma-4-31B-it-UD-Q4_K_XL.gguf
PARAMETER num_ctx 8192
PARAMETER num_gpu 99          # zkusit vše; při OOM ubírat po 5
PARAMETER temperature 0.7
SYSTEM """Odpovídej česky, pokud uživatel nepožádá o jiný jazyk."""
```
```bash
ollama create gemma4-31b-gguf -f modelfiles/gemma4-31b-gguf.Modelfile
ollama run gemma4-31b-gguf "Ahoj, představ se česky."
```

## 6. opencode integrace (hotovo)

- Config: `~/.config/opencode/opencode.json` (symlink/kopie z `opencode.json`
  projektu).
- Auth: `~/.local/share/opencode/auth.json` s `{"ollama":{"type":"api","key":"ollama"}}`.
- Každý model má `"tools": true` — jinak opencode tool calling neviděl.
- Default model: `gemma4-e4b-32k`.
- Disable reasoning mode (může lámat tool call formát).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "ollama/gemma4-e4b-32k",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": {
        "gemma4-e4b-32k":  { "name": "Gemma 4 E4B 32K",      "tools": true },
        "gemma4-31b-8k":   { "name": "Gemma 4 31B Q4_K_M",   "tools": true },
        "gemma4-31b-gguf": { "name": "Gemma 4 31B UD-Q4_K_XL","tools": true }
      }
    }
  }
}
```

## 7. Hlasové zpracování

### 7.1 STT — whisper.cpp + large-v3 (CUDA SM_120, hotovo)
```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc) --config Release
# CUDA 13.2 replace 120 → 120a-real automaticky, funguje.
bash models/download-ggml-model.sh large-v3           # ~3 GB
bash models/download-vad-model.sh silero-v6.2.0       # ~865 KB
```
Použití:
```bash
./build/bin/whisper-cli -m models/ggml-large-v3.bin -l cs \
  -vm models/ggml-silero-v6.2.0.bin --vad -f vzorek.wav
```

### 7.2 TTS — **Chatterbox-TTS-Czech** (primární)
- HF repo: `Thomcles/Chatterbox-TTS-Czech`, fine-tune na CZ nad Resemble AI
  Chatterbox Multilingual, MIT license (komerční OK), voice cloning z 6s wav.
- Python **3.11** nutné (3.14 na systému je moc nový pro PyTorch 2.9).
- Pro Blackwell sm_120 **PyTorch 2.9.1+cu128**:
```bash
python3.11 -m venv voice/.venv-tts
voice/.venv-tts/bin/pip install torch==2.9.1+cu128 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
voice/.venv-tts/bin/pip install chatterbox-tts --no-deps
voice/.venv-tts/bin/pip install transformers accelerate tqdm scipy numpy \
  soundfile safetensors
# Stáhnout t3_cs.safetensors z Thomcles/Chatterbox-TTS-Czech
hf download Thomcles/Chatterbox-TTS-Czech t3_cs.safetensors \
  --local-dir voice/chatterbox-cs
```
Inference (Python):
```python
from chatterbox import mtl_tts
import torchaudio as ta
from safetensors.torch import load_file as load_safetensors
m = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device="cuda")
m.t3.load_state_dict(load_safetensors("voice/chatterbox-cs/t3_cs.safetensors"))
m.t3.to("cuda").eval()
wav = m.generate("Dobrý den, toto je test českého hlasu.")
ta.save("out.wav", wav, m.sr)
```
- Watermark: Resemble AI Perth watermarker je vždy v každém výstupu.
- Sample rate: 24 kHz (`m.sr`).

Záložní: **VibeVoice** (MS, oficiální CZ, long-form), **XTTS v2**.

### 7.3 Orchestrace — jen CLI
Fáze 1: ruční CLI (`whisper-cli` + `tts` script + `ollama run`). Voice loop
až v fázi 2.

## 8. Struktura projektu (aktuální)

```
/home/box/git/github/gemma/
├── PLAN.md
├── opencode.json                     # projektová kopie
├── modelfiles/
│   ├── gemma4-e4b-32k.Modelfile
│   ├── gemma4-31b-8k.Modelfile
│   └── gemma4-31b-gguf.Modelfile     # TBD po stažení GGUF
├── prompts/
│   └── system_cz.md
├── gguf/                             # 31B UD-Q4_K_XL
├── models/ollama/                    # OLLAMA_MODELS (blobs + manifests)
├── whisper.cpp/                      # buildnuté, large-v3 + silero
├── voice/
│   └── .venv-tts/                    # Python 3.11 venv pro chatterbox
└── logs/
```

## 9. Stav implementace (snapshot)

- [x] Systemd override, ProtectHome=no, env vars, KEEP_ALIVE=4m
- [x] `chmod o+x` na cestě pro ollama user
- [x] Adresářová struktura + git init
- [x] Modelfile pro e4b-32k, 31b-8k
- [x] opencode.json + auth.json
- [x] System prompt CZ
- [x] whisper.cpp CUDA build (SM_120 → 120a)
- [x] Silero VAD model
- [ ] `ollama pull gemma4:e4b` — běží
- [ ] `ollama pull gemma4:31b` — běží
- [ ] whisper large-v3 bin — běží (curl)
- [ ] 31B UD-Q4_K_XL GGUF — běží (hf CLI)
- [ ] PyTorch 2.9.1+cu128 v TTS venv — běží
- [ ] Chatterbox + t3_cs.safetensors — po pytorchu
- [ ] Smoke testy všech 3 LLM modelů + STT + TTS
- [ ] Vytvořit `gemma4-e4b-32k`, `gemma4-31b-8k`, `gemma4-31b-gguf` tagy
- [ ] Ladění `num_gpu` pro GGUF

## 10. Otevřené body pro rozhodnutí

- **KV cache kvantizace**: zatím `q8_0` globálně. Jestli 31B narazí na VRAM,
  zkusit `q4_0` (menší, méně kvalitní). Rozhodnutí po smoke testu.
- **TTS finální volba**: Chatterbox-TTS-Czech je primární. Pokud zní hůř než
  očekáváno, vyzkoušet VibeVoice (MS) a Fish Speech S2 Pro.
- **Voice loop**: zatím mimo scope; až po validaci jednotlivých komponent.

---

**Zdroje klíčové pro tento plán:**
- [Ollama gemma4](https://ollama.com/library/gemma4) — oficiální tagy a velikosti
- [Unsloth Gemma 4 31B GGUF](https://huggingface.co/unsloth/gemma-4-31B-it-GGUF)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Thomcles/Chatterbox-TTS-Czech](https://huggingface.co/Thomcles/Chatterbox-TTS-Czech)
- [OpenCode + Ollama integration](https://docs.ollama.com/integrations/opencode)
- [OLLAMA_KEEP_ALIVE FAQ](https://docs.ollama.com/faq)
- [PyTorch RTX 5070 setup](https://medium.com/@gideont/how-i-got-chatterbox-tts-running-on-an-rtx-5070-pytorch-2-9-cuda-12-8-afc92bb5c10b)
