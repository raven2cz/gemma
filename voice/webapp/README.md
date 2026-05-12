# Voice Chat Web UI

Lokální webovka pro hlasový chat: mic → whisper.cpp (STT) → Ollama (LLM, streamed) → Chatterbox (TTS) → přehrání, s WebGL orb avatarem.

## Spuštění

Systémové požadavky: Ollama na `:11434`, `ffmpeg`, CUDA (5070 Ti), venv `voice/.venv-tts`.

```bash
cd /home/box/git/github/gemma
./voice/.venv-tts/bin/uvicorn voice.webapp.server:app --host 127.0.0.1 --port 8080
```

Otevři <http://127.0.0.1:8080>.

## Ovládání

- **Psaný vstup** — napiš do pole dole, **Enter** odešle, **Shift+Enter** = nový řádek. Text vstup → pouze textová odpověď (bez TTS).
- **Klik na mic** — start nahrávání. V **auto-VAD** módu se nahrávání zastaví automaticky po 1.5 s ticha, v push-to-talk módu klikni znovu. Mic vstup → hlasová odpověď (pokud je `TTS` zapnutý).
- **Space** — shortcut start/stop mic (mimo textarea a ostatní inputy).
- **Stop tlačítko** — přeruší recording, LLM, nebo TTS a vrátí do idle.
- **TTS toggle** (nahoře) — globální master switch hlasové odpovědi. Když vypnutý, mic vstup se přepíše, odpoví textem.
- **Stream TTS toggle** (nahoře) — sekání odpovědi po větách a postupné přehrávání zatímco se zbytek syntézuje. *(Phase B, server ho zatím nepoužívá.)*
- **Model/jazyk/voice dropdown** — změna platí pro další dotaz. Výběr se ukládá do localStorage.
- **Rychlý TTS** — `cfg_weight=0` přes monkey-patch v `tts_cs.py`, ~2× rychlejší.

## TTS policy

Intent se bere z volby vstupu:

| vstup | globální TTS | výsledek |
|-------|--------------|----------|
| text  | on/off       | jen text |
| mic   | on           | text + voice |
| mic   | off          | jen text |

## Bilingual (CZ/EN)

- Dropdown **jazyk**: `🌐 Auto` (default) / `🇨🇿 Čeština` / `🇬🇧 English`.
- **Auto detekce** běží server-side z LLM odpovědi (heuristika: diakritika → CZ, jinak stopwords CZ vs EN). Detekce z user inputu je pouze pro volbu LLM system promptu — whisper běží `-l cs`, takže anglický input má často nespolehlivou transkripci.
- **Force override** (CZ / EN) vynutí oba: LLM system prompt i TTS model.
- **1 TTS model v VRAM**. Při přepnutí jazyka se starý model uvolní a nový natáhne (~5–10 s). Server pošle klientovi `lang_hint` po ~50 znacích LLM streamu → klient spustí `/api/tts/preload` a swap proběhne paralelně se zbytkem streamu.
- **Voice family (doporučené):** klient pošle `voice=<family>` (např. `female`, `v1`) a backend per-turn resolvuje `ref_{family}_{lang}.wav` → `ref_{family}.wav` → `ref_female_{lang}.wav` → `ref_female.wav`. `/api/voices` vrátí dostupné families s lang-coverage chipy. Auto-lang + voice family → každý turn dostane správnou per-lang variantu automaticky.
- **Explicit ref (legacy/power-user):** `ref=<filename.wav>` v požadavku překlene family resolve a použije přesně ten soubor — žádný lang fallback, deterministické. UI to odkrývá pod "advanced" v nastavení.
- **Naming konvence souborů:** `ref_*_cs.wav` (jen CZ), `ref_*_en.wav` (jen EN), `ref_*.wav` (univerzální). `/api/refs?lang=cs|en` filtruje legacy ref listing. Pokud pro daný jazyk neexistuje žádný ref, TTS běží bez voice cloning (Chatterbox default).
- **Znám omezení:** český ref voice čtený EN modelem bude mít lehký CZ přízvuk (cross-lingual clone). Pro učení angličtiny doporučuju přidat `ref_female_en.wav` s nativní EN nahrávkou — backend pak auto-přepne díky voice family.

## Architektura

- `server.py` — FastAPI. Endpoints: `/api/{health,models,refs,transcribe,chat,tts}`. VRAM unload je server-side.
- `static/index.html` — layout.
- `static/style.css` — glassmorphism, animovaný gradient mesh background.
- `static/orb.js` — WebGL2 fragment shader orb (výchozí avatar).
- `static/avatar_api.js` — abstract base class pro další avatary.
- `static/app.js` — state machine, mic + RMS VAD, NDJSON stream parser, TTS playback.

## Přidání nového avatara

1. Vytvoř `static/my_avatar.js`:
   ```js
   import { Avatar } from './avatar_api.js';
   export class MyAvatar extends Avatar {
     static meta = { id: 'wave', label: 'Wave' };
     attach(canvas) { /* ... */ }
     detach() { /* ... */ }
     resize(w, h, dpr) { /* ... */ }
     setPhase(p) { /* idle | listening | thinking | speaking */ }
     setAnalyser(a) { /* volitelné, audio amplitude */ }
   }
   ```
2. V `app.js` přidej do `AVATARS` mapy:
   ```js
   import { MyAvatar } from './my_avatar.js';
   const AVATARS = {
     [GlowingOrb.meta.id]: GlowingOrb,
     [MyAvatar.meta.id]: MyAvatar,
   };
   ```
3. (Budoucí) UI picker avatara — Phase 4.

## Známé pain points

- **První TTS cca 10–20 s** cold-start (Chatterbox + safetensors load) — probíhá na pozadí při startu serveru, takže první `/api/tts` request už najde model natažený. Další volání jsou rychlá (sekundy).
- **Unload LLM** před každým transcribe + TTS. Whisper large-v3 s flash-attn chce ~3 GB souvislé VRAM; TTS (2 GB) + LLM (8–18 GB) se s whisperem na 16 GB kartě nevejdou. LLM reload v dalším `/api/chat` je rychlý (~3–5 s), TTS cold-start by byl 10–20 s — proto necháváme TTS v paměti a uvolňujeme LLM.
- **RMS VAD** je citlivý na šum. V hlučném prostředí vypni auto-VAD a používej push-to-talk.
