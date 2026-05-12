# Plan — Voice Chat Web UI

## Cíl

Lokální webová aplikace pro hlasový chat: mikrofon → STT → LLM → TTS → přehrání, s animovaným vizuálním avatarem (zářící koule), která reaguje na fáze (naslouchá / přemýšlí / mluví). Vše plně pod naší kontrolou (žádný Gradio wrapper), snadno rozšiřitelné.

## Scope & non-goals

**Scope:**
- Single-user lokálně (http://127.0.0.1:8080 jen loopback)
- Reuse existujícího stacku: whisper.cpp, Ollama, Chatterbox TTS (přes `tts_cs.py`)
- Python venv `voice/.venv-tts` (přidáme FastAPI, httpx, python-multipart)
- Vanilla JS frontend (žádný React/build-step — snadná editace)
- Extensibility první třídy: **swappable avatar plugin**, volba modelu, voice ref, fast mode

**Non-goals (pro v1):**
- Multi-user / auth
- Mobilní responsive
- Streaming TTS (Chatterbox chunkuje po větách — pro v1 generujeme celou odpověď před přehráním)
- Persistence historie mezi sessions (jen localStorage)
- Export / search
- I18n (jen čeština v UI, labely Czech hardcoded)

## Systémové závislosti (host)

- Ollama na `:11434` (už máme)
- `whisper-cli` binárka + `ggml-large-v3.bin` + `ggml-silero-v6.2.0.bin` (už máme)
- `ffmpeg` — konverze webm/opus z MediaRecorderu → 16kHz mono WAV pro whisper. **Ověří se při startu serveru.**
- `tts_cs.py` funkční (už máme)
- `voice/.venv-tts` existuje (už máme)

Server při startu zaloguje (a varuje, neshodí se):
- `which ffmpeg` → cesta nebo error
- `curl -s http://localhost:11434/api/tags` → OK nebo varování
- `test -x whisper-cli` → OK nebo varování
- CUDA dostupná (`torch.cuda.is_available()` přes krátký `python -c`)

## Architektura

### Backend (FastAPI, Python)

**Soubor:** `voice/webapp/server.py`

**Endpoints:**
| Metoda | Cesta | Účel |
|---|---|---|
| GET | `/` | servíruje `static/index.html` |
| GET | `/static/*` | CSS/JS/fonty |
| GET | `/api/health` | `{ollama, whisper, ffmpeg, cuda}` stavový JSON |
| GET | `/api/models` | `ollama list` (přes `/api/tags`) → JSON pole tagů |
| GET | `/api/refs` | `ls voice/ref_*.wav` → JSON pole |
| POST | `/api/transcribe` | multipart `audio` (webm/opus) → `{text}` přes whisper-cli |
| POST | `/api/chat` | POST JSON `{model, messages}` → **streamuje NDJSON** (ne SSE) tokenů z Ollamy |
| POST | `/api/tts` | JSON `{text, ref, fast}` → `audio/wav` stream |

**POZOR — streaming protokol:** `EventSource` v browseru je pouze GET bez těla. Naše `/api/chat` je POST s JSON. Proto **nepoužíváme SSE/EventSource**. Místo toho server vrací `Content-Type: application/x-ndjson`, každý token jako `{"token":"..."}\n`. Klient čte přes `fetch` + `response.body.getReader()` + ruční line parser. Simpler, funguje s POST, žádný EventSource trap.

**VRAM choreografie — server-side, ne client-side:**
- Před `/api/transcribe`: server zavolá `_unload_all_llms()` — `GET /api/ps` (zjistí co je v Ollamě načtené), pro každý model `POST /api/generate {model, keep_alive:0, prompt:""}`, krátce počká (~300 ms) než Ollama uvolní VRAM.
- Před `/api/tts`: stejně `_unload_all_llms()`.
- **Klient o VRAM neřeší nic** — žádný `/api/unload` endpoint pro klienta, je to interní.
- Whisper-cli sám uvolní po skončení (není keep-alive).
- Chatterbox TTS: subprocess per `/api/tts` volání — cold-start ~10–20 s při prvním requestu (Chatterbox + safetensors load). Pro v1 akceptováno, v2 zvážit TTS daemon.

**Subprocess cleanup:** u `/api/tts` a `/api/transcribe`: pokud klient odpojí (`request.is_disconnected()` nebo `CancelledError`), pošleme `SIGTERM` subprocessu. Implementace přes `asyncio.subprocess` a `proc.terminate()` v `finally`.

**Logging:** každý endpoint loguje stage timing: `{"stage": "transcribe", "ms": 1800, "bytes_in": ...}`. Pomůže diagnózu pomalosti bez nutnosti printů.

**System prompt:** server injektuje před user messages v `/api/chat` systémový prompt:
```
Odpovídej česky, stručně, v jednom odstavci bez markdown formátování,
emoji a bez odrážek.
```
Prevence markdown/emoji v TTS výstupu.

**Text length cap:** `/api/tts` ořízne `text[:2000]` před voláním `tts_cs.py` (tts_cs si potom chunkuje sám). 800 char cap z `chat_voice.sh` byl pro přehrávání — 2000 jako bezpečný strop.

**Detaily subprocess volání:**
- Whisper: `whisper-cli -m ggml-large-v3.bin -vm ggml-silero-v6.2.0.bin --vad -l cs -f in.wav -nt`
- ffmpeg: `ffmpeg -hide_banner -loglevel error -y -i in.webm -ar 16000 -ac 1 -acodec pcm_s16le out.wav`
- TTS: `./.venv-tts/bin/python tts_cs.py "$TEXT" "$OUT" cuda "$REF" [--fast]`
- Ollama unload: `POST /api/generate {model: tag, keep_alive: 0, prompt: ""}` pro každý tag z `/api/ps`

### Frontend (vanilla JS, Canvas 2D)

**Soubory:**
```
voice/webapp/static/
  index.html
  style.css
  app.js          # state machine, API, streaming, mic
  orb.js          # default Avatar plugin (zářící koule)
  avatar_api.js   # abstract Avatar base class (pro extensibilitu)
```

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│ [model ▾] [voice ▾] [☐ fast TTS] [⨯ clear]  ● health │
├──────────────────────────────────────────────────────┤
│                                                      │
│                 (zářící koule — canvas)              │
│                                                      │
├──────────────────────────────────────────────────────┤
│ USER:      Ahoj, co je to řeka?                      │
│ ASSISTANT: Řeka je přirozený vodní tok...            │ (streamovaně)
│ ...                                                  │
├──────────────────────────────────────────────────────┤
│ status: speaking…          [🎙 hold to talk] [⏹ stop]│
│ ❗ error banner (když se něco nepovede)             │
└──────────────────────────────────────────────────────┘
```

**State machine (app.js):**
```
idle → recording (mic down)
recording → transcribing (mic up)
transcribing → thinking (whisper done, POST /api/chat start)
thinking → speaking (chat stream done, /api/tts done, audio.play())
speaking → idle (audio 'ended')
* → error (červený banner + tlačítko „Zkusit znovu")
```

**Stop button (už v Phase 2, ne v Phase 4):** drží `AbortController`. Přeruší fetch `/api/chat` nebo `/api/tts` i `audio.pause()`. Bez stopu je dev nepoužitelný.

**Mic:** `MediaRecorder` na `audio/webm;codecs=opus`. Poslat jako Blob na `/api/transcribe`. Pozor na velmi krátké nahrávky (<1 s) — pár milisekund před stopem nasbírat posledních chunků (`requestData()` před stop).

**LLM stream:** `fetch` + `response.body.getReader()` + ruční NDJSON parser (rozsekat po `\n`). Tokeny se připisují do poslední `ASSISTANT` bubliny live.

**TTS přehrávání:** `<audio>` element, `src = URL.createObjectURL(blob)`. **Napojení AnalyserNode:**
```js
const ctx = new AudioContext();            // pozor: jen po user gestu
const src = ctx.createMediaElementSource(audioEl);
const analyser = ctx.createAnalyser();
src.connect(analyser);
analyser.connect(ctx.destination);         // KRITICKÉ — bez tohoto nebude slyšet!
avatar.setAnalyser(analyser);
```
AudioContext vytvořit až po první interakci (mic click stačí — splňuje autoplay policy).

**Mic AnalyserNode (listening fáze):**
```js
const src = ctx.createMediaStreamSource(micStream);
const analyser = ctx.createAnalyser();
src.connect(analyser);  // nenapojovat na destination (neozvuk!)
avatar.setAnalyser(analyser);
```

### Avatar plugin — rozhraní

`avatar_api.js`:
```js
export class Avatar {
  /** Připojí avatar k canvasu, spustí render loop. Implementace si vybere getContext('2d' | 'webgl' | 'webgl2'). */
  attach(canvas) {}
  /** Odpojí, zruší RAF, vyčistí zdroje. */
  detach() {}
  /** Hook pro window resize / devicePixelRatio. */
  resize(cssWidth, cssHeight, dpr) {}
  /** Nastavení aktuální fáze. Implementace může crossfade přes ~300 ms. */
  setPhase(phase) {} // 'idle' | 'listening' | 'thinking' | 'speaking'
  /** Volitelně: napoj WebAudio AnalyserNode pro amplitudově-reaktivní efekty. */
  setAnalyser(analyser) {}
  /** Metadata pro UI picker. */
  static meta = { id: 'orb', label: 'Zářící koule' };
}
```

Registrace:
```js
// app.js
import { GlowingOrb } from './orb.js';
const AVATARS = { [GlowingOrb.meta.id]: GlowingOrb };
```

Přidání nového avatara = nový JS soubor s třídou + řádek v registru. V UI dropdown se vyrenderuje automaticky. Budoucí WebGL / Three.js varianty si vyžádají vlastní canvas kontext přes `attach()`.

**DPR a resize:** app.js má jeden `ResizeObserver` + `window.matchMedia('(resolution: ...)')` handler, volá `avatar.resize(w, h, devicePixelRatio)`. Avatar si překreslí backing store.

**Default: GlowingOrb (orb.js)**

Canvas 2D, render loop v `requestAnimationFrame`. Čtyři fáze:

| Fáze | Vizuál |
|---|---|
| idle | pomalé dýchání (0.2 Hz), chladně modro-fialový gradient |
| listening | zelený ring, poloměr pulzuje s mic amplitudou (přes AnalyserNode) |
| thinking | duhový shimmer, rotující konický gradient, jemný šum |
| speaking | amber/oranžové vlny, amplituda z TTS `<audio>` přes AnalyserNode |

Implementační trik: udržovat `currentPhase` + `targetPhase` + `fadeProgress`, interpolovat barvy a parametry mezi nimi po 300 ms.

## File layout

```
voice/webapp/
  server.py              # FastAPI
  requirements.txt       # fastapi, uvicorn[standard], httpx, python-multipart
  static/
    index.html
    style.css
    app.js
    orb.js
    avatar_api.js
  README.md              # spouštění: ./.venv-tts/bin/uvicorn voice.webapp.server:app --host 127.0.0.1 --port 8080
```

## Fáze implementace

**Phase 1 — Backend MVP + minimální použitelný frontend**
- FastAPI skeleton, `/api/health`, `/api/models`, `/api/refs`.
- `/api/transcribe` (ffmpeg + whisper subprocess, server unloads LLMs first).
- `/api/chat` NDJSON streaming proxy na Ollamu (s injektovaným system promptem).
- `/api/tts` (tts_cs subprocess, server unloads LLMs first, text cap 2000 chars).
- Minimální index.html + app.js: **mic tlačítko, model dropdown, stop button, error banner, chat transkript**. Bez orba — jen text UI s tlačítkem. Musí být testovatelné end-to-end.

**Phase 2 — Orb avatar**
- avatar_api.js abstract, orb.js implementace.
- Canvas 2D render loop, čtyři fáze, crossfade transitions.
- AnalyserNode napojení: při recording na mic stream, při speaking na `<audio>`.
- DPR/resize handler.

**Phase 3 — Polish & extensibility**
- Dropdown na voice ref (GET /api/refs).
- Fast TTS toggle.
- Persistence konverzace v localStorage (poslední session).
- Clear conversation.
- Připravit avatar registry pro další pluginy (dokumentovat v README).

## End-of-speech detekce (VAD)

Klíčový UX problém. Tři úrovně:

1. **Phase 1 default — RMS VAD:** jednoduchý amplitudový detektor v browseru. `AnalyserNode.getFloatTimeDomainData()` → RMS → porovnání s prahem. Když RMS < threshold (−50 dBFS) po dobu >1.5 s po první detekci řeči → stop nahrávání automaticky. ~50 řádků, žádné závislosti. Funguje dobře v tichém prostředí, v hluku je noisy.
2. **Phase 3 upgrade — Silero VAD v browseru:** ONNX runtime web + Silero VAD model (~1 MB). Přes [`@ricky0123/vad-web`](https://github.com/ricky0123/vad-web) CDN (ESM import). Robustní proti hluku, detekuje konec věty spolehlivě. Callback `onSpeechEnd(float32Array)` vrátí čisté PCM.
3. **Manuální fallback — push-to-talk tlačítko:** vždycky dostupné, když VAD selže. Držet = nahrává, uvolnit = stop.

UI toggle v headeru: `VAD: [auto ▾]` (auto/push-to-talk).

## Otevřené otázky / kompromisy

1. **VAD threshold tuning:** −50 dBFS a 1.5 s timeout jsou startovní hodnoty. V UI dát slidery (pokročilé).
2. **Mid-stream TTS:** chunkujeme na větách uvnitř tts_cs.py → teoreticky by šlo streamovat i WAV chunky klientovi (`MediaSource API`). Nechávám na v2.
3. **Keep-alive strategie:** cold-start Chatterboxu ~10–20 s je největší UX pain. V2 buď (a) daemonizovat TTS přes FastAPI sub-process držený v paměti, nebo (b) přidat mode kde se po TTS znovu NAČTE malý LLM (E4B) a velký uvolní — záleží na profilu.
4. **Race na Ollama unload:** `POST /api/generate keep_alive:0` je asynchronní. Po volání chvíli počkat (300–500 ms) než znovu něco loadneme, jinak OOM. Server to řeší interně.
5. **Avatar varianty do budoucna:** WebGL shader orb, particle system, 3D model přes Three.js. Pluginovatelnost je v rozhraní od začátku.

## Známé UX pain points (očekávané)

- **První TTS volání pomalé** (~10–20 s cold start). Status label musí ukazovat „TTS model se načítá…" ať uživatel nezmatkuje.
- **Unload + reload LLM mezi turny** (~5–15 s na 12B modelu). Status: „Uvolňuji VRAM…".
- **Krátké nahrávky <1 s** mohou mít špatný webm header — testovat a případně varovat.

## Definice hotového (v1 = konec Phase 3)

- [ ] `uvicorn voice.webapp.server:app` nastartuje a zaloguje health check.
- [ ] `http://127.0.0.1:8080` zobrazí UI s dropdowny, mic button, orb canvas.
- [ ] Kliknu mic, mluvím česky, uvolním → do ~2 s se zobrazí přepis.
- [ ] Vidím, jak se odpověď LLM píše token po tokenu.
- [ ] Po skončení streamu slyším TTS, orb je amber a vlní se s hlasem.
- [ ] Fast TTS toggle viditelně zrychlí (~2×).
- [ ] Výběr modelu v dropdownu funguje a změní chování další otázky.
- [ ] Stop button skutečně přeruší (a) chat generování, (b) TTS přehrávání.
- [ ] Když je Ollama down, UI zobrazí chybový banner s konkrétní hláškou.
- [ ] První TTS request ukáže „TTS model se načítá" indikátor.
