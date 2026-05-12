# Plan: Text input + implicit TTS policy + streaming TTS

Cíl: přidat textový composer (jako ChatGPT/Gemini), zavést konzistentní
pravidlo kdy hrát TTS, a rozběhnout TTS **po větách paralelně s LLM
streamem** aby uživatel nečekal na syntézu celého odstavce.

## UX decisions (schváleno)

**Composer**: dole místo samostatné mic-area je jedno široké pole:
```
┌────────────────────────────────────────────────────────────┐
│  napiš nebo začni mluvit…                    [ 🎙 ] [ ■ ]  │
└────────────────────────────────────────────────────────────┘
```
- `<textarea>` roste do 5–6 řádků, pak scroll.
- Mic a stop zůstávají uvnitř toho pole vpravo.
- Enter = send · Shift+Enter = nový řádek · prázdný send = no-op.
- Během recording / LLM stream / TTS je textarea `disabled` a vizuálně
  tlumená (cursor not-allowed).

**Globální toggle "TTS" / "mluvená odpověď"** v topbaru (vedle
`rychlý TTS`, `auto VAD`):
- Checked (default) = povolit TTS výstup.
- Unchecked = nikdy nehrát TTS, ani pro mic vstup. Ušetří to per-turn
  `/api/tts/preload` swap. **Neušetří** to startup TTS load a health
  závislost — server preloaduje CS TTS při startu a `tts_ready` je
  součást zeleného health dot. (Refactor pro lazy startup by byl
  samostatný plán.)
- **Nepojmenovat jako "hlas"** — label `hlas` už má ref-voice dropdown.

**Implicitní pravidlo** `wantTTS`:
```
wantTTS = globalVoiceOn && inputMode === 'mic'
```
- **Snapshot v okamžiku odeslání** (ne při dokončení streamu). Uložit
  `turn.wantTTS` do lokální proměnné v `runChatAndTTS()`. Pokud uživatel
  během streamu přepne toggle, platí až pro další turn. Konzistentní
  chování bez race.
- `text` vstup → čistě textová odpověď, orb jde do `idle` hned po streamu.
- `mic` vstup + voice on → stávající flow (preload, TTS playback).
- `mic` vstup + voice off → stejné jako text: jen zobrazíme odpověď.

Takže žádný per-message přepínač "pošli hlasem / textem". Volba vstupu
nese intent.

## Frontend changes

### `index.html`
- V `.topbar .controls` přidat třetí `.ctrl.toggle` pro `voice-toggle`
  (checked default). Label: `hlas` nebo `TTS odpověď`.
- V `<footer class="bottombar">` nahradit `.mic-area` za `.composer`:
  ```html
  <form class="composer" id="composer">
    <textarea id="text-input" rows="1" placeholder="napiš nebo stiskni mic…"></textarea>
    <div class="composer-actions">
      <button id="mic-btn" class="mic-btn">…</button>
      <button id="stop-btn" class="stop-btn" hidden>…</button>
    </div>
  </form>
  ```
- Status text posunout jinam: malý caption pod composerem, nebo nahoru
  vedle brandu. (Decision: pod composer, fg-faint, 11px mono.)

### `style.css`
- `.composer`: flex row, glass pill (rgba bg + 1 px border +
  backdrop-filter), radius 24 px, focus-within ring s `--accent`.
- `.composer textarea`: transparent, no border, resize none, auto-grow
  přes JS (min-h 24 px, max-h ~140 px), font-size 14–15 px, Inter.
- `.composer-actions`: pravý shrink, gap 8 px.
- Zmenšit `.mic-btn` na 44 px, `.stop-btn` na 38 px, aby se vešly do
  pillu.
- Disabled stav composeru (během **recording | transcribing |
  thinking | speaking**, tj. všech non-idle fází — viz state machine
  v app.js:4): opacity 0.6, textarea `disabled`, placeholder změnit na
  `čekej…`. `transcribing` je důležitý — plan ho v úvodu opomněl.

### `app.js`
- Nový state field: `inputMode` (`'text' | 'mic'`) — nastavuje se v
  momentě odeslání.
- Nový state field: `voiceEnabled` (bool, default true) — čte toggle +
  localStorage.
- `handleTextSubmit(text)`:
  - trim, empty → no-op.
  - `state.inputMode = 'text'`
  - `state.messages.push({role:'user', content:text})` + `persistMessages()`
    (dnes to dělá mic flow implicitně po transcribe). `runChatAndTTS()`
    bere poslední message z `state.messages` — nic mu neposíláme přes
    argument.
  - `addMessage('user', text)`
  - `runChatAndTTS()`
- `handleMicStop()`: stávající flow, ale `state.inputMode = 'mic'`
  nastavit **před** voláním transcribe, aby byl mode korektní i kdyby
  transcribe selhal.
- `runChatAndTTS()` dostane nový branch:
  ```js
  const wantTTS = state.voiceEnabled && state.inputMode === 'mic';
  // ... stream ...
  if (wantTTS) {
    // preload na lang_hint, TTS po streamu
  } else {
    // skip preload, skip /api/tts
    // ALE: assistant message musí být stále pushed do state.messages
    //      a persistMessages() zavoláno — dnes se to děje v rámci
    //      TTS větve (viz app.js:529). Refactor: přesunout save před
    //      if (wantTTS), aby text-only měl konzistentní persistenci.
    // setPhase('idle') až po streamu
  }
  ```
- Auto-grow textarea: `input` listener, `el.style.height='auto'`, pak
  `el.style.height = Math.min(el.scrollHeight, 140) + 'px'`.
- Keybindings:
  - Enter (no shift) v textarea → submit.
  - Shift+Enter → default (newline).
  - Space: stávající handler (app.js:674) ignoruje jen
    `HTMLInputElement`. Rozšířit guard — Space nesmí triggerovat mic,
    pokud je fokus v `textarea`, `input`, `select`, `button`, nebo
    prvku s `contenteditable`. Cleanější je kontrola
    `e.target.closest('textarea,input,select,button,[contenteditable]')`.
- Restore messages: nic se nemění (localStorage už existuje).
- Voice toggle handler: `state.voiceEnabled = e.target.checked`,
  `localStorage.setItem('voice', state.voiceEnabled)`.
- Při `voiceEnabled=false` během už probíhajícího TTS: NE přerušovat
  (aby toggle během přehrávání nebyl destruktivní). Platí pro další
  turn.

### Mic-only flow (kde se nic nemění)
Mic button start → record → VAD/stop → transcribe → zobrazí user text v
transcriptu (už to umí) → stejný `runChatAndTTS` cesta. Jen nastaví
`inputMode='mic'` před voláním.

## Server changes

Pro pure TTS policy: žádné. TTS je klient-side volání, klient prostě
přeskočí `/api/tts` a `/api/tts/preload`.

**Pro markdown v text módu** (doporučené, ne povinné):
- `/api/chat` bere nový volitelný field `want_tts` (default true pro
  zpětnou kompatibilitu). Pokud `false`, system prompt dostane suffix
  povolující markdown (nahradí "žádný markdown" větu v server.py:96).
  Pokud `true`, zachová se "žádný markdown" (TTS engine markdown syntax
  vyslovuje a zní to blbě).
- Klient posílá `want_tts: state.voiceEnabled && state.inputMode === 'mic'`
  v body `/api/chat`.

## Edge cases & guardrails

1. **Enter v disabled textarea**: vyřešeno přes `disabled` atribut —
   prohlížeč to ignoruje.
2. **Uživatel píše během nahrávání**: textarea je disabled během
   `recording | thinking | speaking`. Po idle se odblokuje.
3. **Stop uprostřed text-only chatu**: musí fungovat `chatAbort.abort()`
   (už existuje) — žádný TTS cancel nepotřeba.
4. **Voice toggle off uprostřed streamu**: TTS ještě není nahrané, takže
   jednoduše neudělá `/api/tts` call na konci. Preload mohl být spuštěn
   — necháme ho doběhnout (už začal VRAM swap).
5. **První load, composer focus**: autofocus do textarea — ale jen na
   desktopu (mobilní klávesnice by vyskočila otravně). `matchMedia`
   check.
6. **Fallback když browser nemá mic access**: mic button `disabled`,
   composer nadále funkční. (Už dnes je to broken — není fallback; při
   této změně to napravíme "zdarma".)
7. **Welcome screen**: text "zmáčkni mikrofon nebo prostě začni mluvit"
   přepsat na "napiš, nebo zmáčkni mic".
8. **Accessibility**: přidat skrytý submit button (`<button type="submit"
   class="sr-only">Odeslat</button>`) pro screen readery a mobilní
   `Enter` na soft-keyboard. Hlavní UI to neovlivní.
9. **Mobile**: `body` má `overflow:hidden` (style.css:23) — po otevření
   soft-keyboard může composer sebrat transkriptu prostor. Omezit
   max-height textarea na mobilu (`@media (max-width: 900px)` → 80 px).
   Testovat skutečně na telefonu, ne jen DevTools.
10. **Server system prompt** (server.py:96) dnes říká modelu "žádný
    markdown". Když máme markdown rendering + text mód bude hlavní pro
    "učení se", relaxovat tento zákaz (aspoň pro text-only turn).
    Vyžaduje malou server změnu: `/api/chat` dostane `want_tts` flag a
    podle něj použije jiný system prompt suffix. **Volitelné** — pokud
    neuděláme, text odpovědi budou plain bez formátování. Doporučuji
    udělat zároveň, jinak se ztratí půlka hodnoty markdown renderu.

## Testing checklist

- [ ] Text → Enter → text-only response, žádný TTS request v
      devtools network.
- [ ] Mic → VAD → TTS hraje.
- [ ] Voice toggle off + mic → transkribuje, odpoví textem, žádný TTS.
- [ ] Voice toggle off + text → jen text.
- [ ] Shift+Enter = newline, Enter = send.
- [ ] Space v textarea píše mezeru, Space mimo textarea toggluje mic.
- [ ] Stop uprostřed text streamu přeruší LLM.
- [ ] Composer disabled vizuálně během recording.
- [ ] localStorage si pamatuje voice toggle po reloadu.
- [ ] Textarea auto-grow funguje, při 7+ řádcích scrolluje uvnitř.
- [ ] Markdown rendering nadále funguje u text-only odpovědí.

## Streaming TTS (paralelní syntéza po větách)

### Motivace

Dnes: LLM stream skončí → celý text → 1× `/api/tts` → čekání 4–8 s
→ playback. Time-to-first-audio = LLM_time + TTS_time (7–16 s).

Nově: LLM tokeny se za běhu sekají po větách → jakmile je hotová první
věta, pošle se do TTS executoru → WAV se hned streamuje klientovi →
playback začíná po ~1–2 s. Zbytek se syntézuje paralelně s přehráváním.

### Nový toggle "stream TTS" (default ON)

V topbaru vedle `TTS` / `auto VAD` / `rychlý TTS`. Když OFF, fallback na
stávající "celý text najednou" flow (užitečné pro debug nebo stabilitu).

### Architektura: unifikovaný endpoint `/api/turn`

Místo `/api/chat` (text) + `/api/tts` (audio) se složí do jednoho
NDJSON streamu. Klient otevře **jedno spojení** a dostává heterogenní
eventy.

**Request body** (JSON):
```json
{
  "model": "...", "ref": "ref_female.wav",
  "fast": false, "lang_override": "auto",
  "prev_lang": "cs",
  "want_tts": true, "stream_tts": true
}
```

**Response NDJSON events**:
- `{"type":"text","delta":"…"}` — LLM token delta (stejné jako dnes, jen
  zabalené v obálce).
- `{"type":"user_lang","lang":"cs"}` — stejné jako dnes.
- `{"type":"lang_hint","lang":"en"}` — stejné (spouští preload).
- `{"type":"audio","seq":0,"url":"/api/turn/<id>/audio/0.wav","chars":72}`
  — připravený audio chunk. Klient ho zařadí do queue.
- `{"type":"lang","lang":"cs"}` — finální detekce.
- `{"type":"done"}` — konec.

Když `want_tts=false` → server nevysílá `audio` eventy, funguje jako
dnešní `/api/chat`.
Když `want_tts=true && stream_tts=false` → server počká, syntézuje
najednou, pošle 1× `audio` event na konci.

### Server-side implementace

**Sentence chunker** (`voice/sentence_chunker.py`, samostatný modul):

Chunker musí pracovat na **speakable text**, ne na raw LLM výstupu
(codex nález #5). Pipeline:
1. **Strip non-speakable**: fenced code bloky (`` ```…``` ``) → emit
   jako samostatný text-only chunk bez TTS (klient má markdown render,
   TTS nemá smysl). Inline code (`` `foo` ``) → slovo "kód" nebo
   prostě content bez backticků, dle chuti. `<think>…</think>` tagy →
   zahoď. Markdown syntax (`**`, `*`, `#`, `_`, `[text](url)`) →
   extrahuj jen text, ne URL.
2. **Sentence split**: nespoléhat na naivní `re.split(r"(?<=[.!?])\s+")`
   (tts_cs.py:352). Použít pravidla:
   - Terminátory: `. ! ? …` (včetně unicode `\u2026`).
   - **Negative lookbehind** pro abbrev: reuse `ABBREV` dict z
     tts_cs.py:261 + EN sada (`Mr.`, `Dr.`, `etc.`, `e.g.`, `i.e.`,
     `vs.`, `Inc.`, `Ltd.`).
   - Jednopísmenné + tečka (`p.`, `č.`, `A.`) → zkratka.
   - Čísla s desetinnou tečkou (`3.14`, `1.5×`) → ne konec věty.
   - Zavírací uvozovky/závorky po terminátoru (`."`, `.)`, `…"`) →
     přilepit k předchozí větě.
3. **Chunk sizing**:
   - První chunk: 1 věta, min 40 znaků, max 120 (TTFB win).
   - Následné: 2–3 věty nebo ~200 znaků, pak flush.
   - Hard cap 400 znaků bez terminátoru → force flush na mezeře.

**Interface**:
```python
def chunk_for_tts(buf: str, *, first: bool, lang: str) -> tuple[list[Chunk], str]:
    """Vrátí (emit_chunks, remainder).

    Chunk = {"text": str, "speakable": bool, "kind": "sentence" | "code" | "other"}

    speakable=False chunky klient zobrazí v transcriptu ale neenqueueuje
    do audio queue."""
```

**Testy**: unit testy s minimálně 20 fixtures pokrývající abbrev, čísla,
code blocky, `<think>`, mixed CZ/EN věty, ellipsis, uvozovky. Protože
tohle je zdroj ~většiny budoucích bugů.

**Pipeline orchestration** v `/api/turn` handleru:
- 2 asyncio tasks + 1 consumer queue + 1 out queue:
  1. **LLM task**: streamuje z Ollamy, pro každý delta: buffer += delta,
     `emit_event({type:text, delta})`, pak `chunks, buf = chunk_for_tts(buf, first=first_done)`.
     Každý chunk se naqueueuje do `tts_queue`. Po konci: `tts_queue.put(SENTINEL)`,
     `emit_event({type:text_done})`.
  2. **TTS consumer task**: `while True: chunk = await tts_queue.get()`;
     pokud SENTINEL → break; submit `run_in_executor(...)`; await future → uloží WAV
     → emit `{type:audio, seq, url}`. Po break: `emit_event({type:done})`.
- **Event ordering je kritické**: `{type:done}` vysílá **až consumer
  task po drainu TTS queue** (ne LLM task). Jinak klient vypne playback
  před posledním audio chunkem. Pořadí: text... → text_done → audio...
  → done. (codex nález #2)
- **Backpressure**: `tts_queue = asyncio.Queue(maxsize=8)`. LLM task při
  plné queue se zablokuje na `put()` — auto-throttle. Bez limitu by
  dlouhá odpověď s code blocky (neemituje audio, ale emituje text)
  byla OK, ale cancel+retry flow by zanechal stale tasks. (nález #4)
- **Merger**: out_queue je oddělená od tts_queue. Oba tasks pushují do
  out_queue, hlavní NDJSON generator čte FIFO. `asyncio.Queue` je
  fair FIFO, takže ordering z pohledu jednoho producera je zachované.
- Zdůvodnění: TTS executor má 1 worker, takže chunky se synthézují
  serially; ale LLM + TTS běží paralelně.

**Per-turn temp dir**: `TMPDIR/turn_<id>/chunk_<seq>.wav`.

**Cleanup je delikátní** (codex nález #3):
- NESMÍ se smazat po `{type:done}`, protože klient ještě nemusel
  stihnout fetchnout poslední audio URL (prefetch race).
- Per-file delete **až po úspěšném GET streamu** (stejně jako stávající
  `/api/tts` v cleanup_stream, server.py:617). Endpoint
  `GET /api/turn/<id>/audio/<seq>.wav` → stream WAV → unlink v `finally`.
- **Cancel path**: server označí turn jako canceled + dá TTL 30 s —
  klient může ještě mít rozpracovaný fetch. Po 30 s force cleanup.
- **Safety net**: background task každých 60 s projde `TMPDIR/turn_*`
  a smaže dir starší 10 min (proti orphan při crashi).
- **Whitelist**: `turn_id` musí matchnout `^[a-f0-9]{16}$`, `seq`
  jen integer, žádné `..` nebo `/`.

**Cancel** (přesný dvouvrstvý design):
- **Low-level** `CANCEL_EVENT` (thread Event) = přeruš právě běžící GPU
  synth. Ponechán beze změny: clearuje se na startu každého synthu
  (tts_cs.py:39, server.py:224), což je správně — je to per-synth interrupt.
- **High-level** per-turn `cancel_flag` (immutable bool wrapper, např.
  `{"canceled": False}` dict, nebo asyncio.Event přímo). Checkuje se na
  **4 místech**, jinak vzniknou race windows:
  1. Před `tts_queue.put_nowait(chunk)` v LLM tasku.
  2. Před `run_in_executor(...)` submit v consumer tasku.
  3. Po návratu z executoru (synth mohl skončit těsně před cancelu).
  4. Před emit `{type:audio}` na out_queue (i když WAV je hotový,
     neposíláme ho).
- Flow cancelu: `/api/tts/cancel?turn=<id>` → nastaví `cancel_flag[turn]`
  + `CANCEL_EVENT.set()`. LLM task se přeruší čekáním na další Ollama
  chunk? → HTTPX connection abort. Consumer task vyprázdní queue a končí.
  Hlavní NDJSON generator pošle `{type:canceled}` a zavře.
- Globální `/api/tts/cancel` (bez turn_id) zruší všechny aktivní turny —
  fallback pro starší klienty.

**Lang lock pro turn** (codex nález #8):

Plan původně říkal "check před prvním submitem", ale to řeší jen cold
start. Problém je pokud `lang_hint` po 50 znacích určí EN, ale po 200
znacích finální detekce řekne CS — první chunky už byly synthézovány
EN modelem. Uživatel uslyší "Ahoj, jak se máš" s anglickou prozódií.

**Řešení**: jakmile máme rozhodnutí (buď `lang_override` != "auto",
nebo `lang_hint` ze streamu), **zamknout** `turn.tts_lang` až do konce
turnu. Další detekce se do něj nepromítne.

Pipeline:
- Pokud `lang_override` != "auto" → lock hned, TTS consumer může začít
  jakmile má chunk.
- Pokud `lang_override == "auto"` → **první chunk počká** na `lang_hint`
  event (server už ho emituje po 50 znacích, server.py:524). Tím se
  obětuje ~1 s TTFB, ale vyhne se mis-synth.
- Finální `{type:lang}` event po dokončení stream je jen informativní
  (pro budoucí turn jako `prev_lang`).

### Klient-side implementace

**Audio queue s A/B bufferingem**:
```js
const audioQueue = [];      // [{seq, url}, ...]
let currentAudio = null;    // HTMLAudioElement
let prefetchAudio = null;   // pre-cached next

function enqueueAudio(url, seq) {
  audioQueue.push({url, seq});
  if (!currentAudio) playNext();
  else prefetchNext();
}

function playNext() {
  const item = audioQueue.shift();
  if (!item) { setPhase('idle'); return; }
  currentAudio = prefetchAudio?.src === item.url ? prefetchAudio : new Audio(item.url);
  currentAudio.addEventListener('ended', playNext);
  reconnectAnalyser(currentAudio);  // orb amplitude
  currentAudio.play();
  prefetchNext();
}

function prefetchNext() {
  const next = audioQueue[0];
  if (!next) return;
  prefetchAudio = new Audio();
  prefetchAudio.preload = 'auto';
  prefetchAudio.src = next.url;
}
```

**Analyser — single element only** (codex nález #6):

`createMediaElementSource(el)` je permanent-bound; element může být
bound jen **jednou**. Druhý `new Audio()` + reconnect = crash nebo
silent fail.

Bezpečný design:
- Jeden `<audio>` element (stávající `#tts-audio`, app.js:558) — už má
  jeden bound source.
- Prefetch dalšího chunku = `fetch(url) → blob → URL.createObjectURL()`.
  Blob URL se uloží, ale **nepřiřazuje se** do druhého Audio elementu.
- Na `ended` event: `audio.src = nextBlobUrl; audio.play()`.
- Po `play()`: revoke starý blob URL (paměť).

**Autoplay policy**: pro text-input mode + voice on (budoucí, dnes
plan říká text→bez TTS, ale pokud bychom to změnili) je potřeba
volat `ensureAudioCtx()` při click na send button, ne až při prvním
audio eventu. Přidat do `handleTextSubmit()`.

**NDJSON parser** v `runChatAndTTS()`: rozšířit switch o `audio` event
→ `enqueueAudio(ev.url, ev.seq)`. `text` event → `appendToLast()`.
Stop button: `POST /api/tts/cancel` + `currentAudio.pause()` +
`audioQueue.length = 0`.

**Fallback (stream_tts=false)**: klient akceptuje jeden `audio` event
na konci, chová se jako dnes.

### Interakce s text input plánem

- `wantTTS` logika se nemění: `wantTTS = voiceEnabled && inputMode === 'mic'`.
  Posílá se jako `want_tts` do `/api/turn`.
- `streamTTS` = další flag, posílá se jako `stream_tts`.
- `runChatAndTTS()` se přejmenovává na `runTurn({wantTTS, streamTTS})`.
- **Kritické pro ortogonalitu fází**: Fáze A zavede `runTurn()`
  interface a flagy, ale vnitřně stále volá staré `/api/chat` + `/api/tts`.
  Fáze B jen přepíše vnitřek `runTurn()` na `/api/turn` NDJSON parser.
  Tím zůstane UX state, persistence, stop semantika čistě ve Fázi A.
  (codex nález #9 — fáze nejsou plně ortogonální bez tohoto rozhraní.)
- Původní `/api/chat` + `/api/tts` endpointy zůstávají pro zpětnou
  kompatibilitu během vývoje; po stabilizaci deprecovat.

### Edge cases

1. **Velmi krátká odpověď** ("Ano."): chunker pod threshold → neflushne
   dokud LLM neskončí → na konci force flush zbylý buffer. Takže "Ano."
   dorazí jako jeden chunk po `done`.
2. **LLM chyba uprostřed**: už máme error event; consumer task vyprázdní
   queue, pošle error, skončí.
3. **TTS chyba na chunku N**: emit `{type:audio_error, seq:N}` →
   klient přeskočí chunk, pokračuje dalším. Nebo: přeruší celou odpověď
   a ukáže toast. (Doporučuji: přeskočit, logovat server-side, ať se
   uživatel neztratí.)
4. **Prosodie přes chunk boundaries** (codex nález #7):
   - 20–50 ms fade-out řeší jen **click artefakty**, ne prosody
     diskontinuitu (důraz, rytmus věty).
   - **Primární řešení**: server po synth přidá **80–120 ms ticha**
     na konec každého chunku (numpy `np.concatenate([audio, silence])`
     v `tts_cs.py`). Simulates natural sentence pause. Zvuk zní
     přirozeně protože mezi větami je i tak pauza.
   - Krátký fade-out (~20 ms) navíc zabrání click při přechodu mezi
     audio elementy.
   - **Neříkat crossfade na klientu** — vyžaduje WebAudio scheduling
     s dekódovaným PCM, smear konsonantů, překomplikované.
5. **Preload swap během aktivního synthu**: consumer task řeší — další
   chunk se submitne až po swap complete.

### Performance odhady

- Typická LLM odpověď: 150 slov ≈ 900 znaků ≈ 4–6 vět ≈ 8 s streamu.
- Chunk 1 (první věta, ~50 znaků): ready po ~1.2 s LLM + ~0.8 s TTS = **2 s do first audio**.
- Současná latence: ~12 s (8 s LLM + 4 s TTS na 900 znacích).
- **Zrychlení ~6× u TTFB**, celková doba beze změny (TTS běží během playbacku).

### Adaptivní chunk sizing (instrument first, adapt later)

User navrhl dynamickou adaptaci. Codex review: **worth instrumenting,
not worth dynamic control v1**. Static thresholdy (první 40–120 znaků,
další 200 znaků, hard cap 400) stačí protože TTS je ~0.5× RT —
underflow je dominantně způsoben cold-start / lang swap / cancel bugy,
ne chunk size.

**V1 — jen metriky** (server logs):
- Per chunk: `synth_ms`, `audio_duration_ms`, `chars`, `queue_depth`.
- Per turn: `first_audio_ms` (TTFB), `total_synth_ms`, `total_audio_ms`,
  `gap_ms` mezi chunky (z playback progress eventů klienta — nový NDJSON
  event `{type:ack_played, seq}` by mohl být nice-to-have, ale neslepovat
  do v1 aby se SLO stabilizovalo).

**V2 — adaptace**:
- EWMA z `synth_ms / audio_ms` (α ≈ 0.3) a `ms_per_char`.
- Adaptovat **mezi turny**, ne každou větu (hysterezis).
- Pravidla:
  - Pokud ratio > 0.8 nebo `queue_audio_ms_ahead < 1000` → grow target
    na 250–350 znaků (méně overhead).
  - Pokud ratio < 0.5 a first-audio > 2500 ms → shrink první chunk
    na 40–80 znaků.
- Storage: per-session (in-memory), ne persistent. Reset na restart.

**V1 je default**, adaptace až když reálně uvidíme problém v metrikách.
Neinvestovat do EWMA kontroleru předtím, než znám její variabilitu.

### Rizika

1. **1-worker TTS executor**: Pokud TTS_time > LLM_streaming_time per
   chunk, queue naroste a playback se čas od času zastaví "na nic"
   mezi chunky. Realistické měření: Chatterbox na 5070 Ti ~0.5× real-time
   (tj. 2 s zvuku za 1 s compute). LLM typicky 30–50 tok/s ~200 znaků/s.
   Takže TTS je obvykle rychlejší než playback → queue se drží prázdná,
   žádné gapy. **Ale** první chunk musí vystačit na dobu, než druhý
   dorazí: 2 s zvuku, ~1.5 s synth druhého → OK.
2. **Gaps mezi chunky**: řešeno server-side silence padding, viz Edge
   cases #4.
3. **Cancel latence**: 4-point check + CANCEL_EVENT + audio.pause();
   musí všechno stihnout < 200 ms, jinak uživatel uslyší zbytek věty.
4. **Temp disk usage**: 5–10 WAV files per turn × mnoho turns. Cleanup
   per-file po GET streamu + 10 min watchdog.
5. **První SLO co praskne** (codex prioritizace): lang swap mid-stream
   (EN→CS), ne throughput. Řešeno lang-lock per turn.

### Rozsah

- Server: ~300 řádků (chunker, orchestration, nový endpoint, cleanup).
- Klient: ~150 řádků (audio queue, blob prefetch, event switch).
- Tests: integračně přes curl + reálný prohlížeč.

## Největší riziko (z codex auditu)

Phase/UI synchronizace. Dnes se `stopBtn.hidden`, `composer.disabled`
(nově), `micBtn.recording`, `chatAbort`, `ttsAbort` řídí ručně ve více
větvích (app.js:448). Přidáním text módu se počet větví zvyšuje. **Doporučuji**:
vytáhnout `setPhase(p)` tak, aby sama nastavovala composer disabled
state, stop button visibility a mic button třídu. Jediná source of
truth.

## Rozsah a pořadí implementace

**Fáze A — UX refactor** (základ):
1. Refactor `setPhase()` — centralizovat composer/stop/mic UI bindy.
2. HTML: composer markup, dva nové topbar toggle (`TTS`, `stream TTS`).
3. CSS: composer pill styling + responsive, disabled state, mobile
   max-height.
4. JS: textarea auto-grow, submit handler, Space shortcut guard.
5. JS: state fields `inputMode`, `voiceEnabled`, `streamTTSEnabled`,
   snapshot flagů na začátku turnu.
6. JS: toggle listeners + localStorage.
7. Welcome text copy, accessibility (sr-only submit), README.

**Fáze B — streaming TTS** (architektura):
8. `voice/sentence_chunker.py` — sentence chunker + testy.
9. `server.py`: nový `/api/turn` endpoint s LLM+TTS orchestrací, per-turn
   temp dir, audio file server, cancel flag.
10. `server.py`: `/api/turn/<id>/audio/<seq>.wav` endpoint.
11. `server.py`: cleanup watchdog (background task).
12. `tts_cs.py`: přidat krátký fade-out (20–50 ms) na konec synth WAV,
    aby chunky navazovaly bez cvaknutí.
13. Client `app.js`: přejmenovat `runChatAndTTS()` → `runTurn()`,
    rozšířit NDJSON switch o `audio` event, audio queue + blob prefetch.
14. Server markdown toggle (když `want_tts=false`, povolit markdown v
    system promptu).

**Fáze C — stabilizace**:
15. Fallback: `/api/chat` + `/api/tts` zůstávají pro `stream_tts=false`
    nebo odpojené. Po ověření stability v prohlížeči deprecovat.
16. Integrační test: 3 realné scénáře — krátká CZ odpověď, dlouhá EN
    odpověď s code blockem, přerušení uprostřed 3. chunku.

**Odhad diffu**: ~500 řádků (Fáze A ~200 + B ~300).

## Testy

Nový adresář `tests/` v rootu repa. Python tests spouštíme přes
`./voice/.venv-tts/bin/pytest tests/`. Pro v1 pokryjeme:

**`tests/test_sentence_chunker.py`** (kritické — zdroj budoucích bugů):
- Abbrev: `"Napiš p. Nováka."` → nesekat před `p.`.
- EN abbrev: `"Meet Dr. Smith."` → nesekat před `Dr.`.
- Čísla: `"Máme 3.14 km."` → nesekat před `3.14`.
- Ellipsis: `"Hmm… dobrá."` → respektovat `…`.
- Zavírací uvozovky: `'Řekl: "Ahoj."'` → sentence končí za uvozovkou.
- Code block: `'Text. ```py\nx=1\n```\nDalší.'` → code vyfiltrovaný,
  speakable chunks: "Text.", "Další." (code emit jako non-speakable).
- Inline code: `"Spusť `ls -la`."` → `ls -la` neskákat, ale ve výstupu.
- `<think>...</think>`: strippovat celé.
- Mixed CZ/EN: správně sekat bez ohledu na jazyk.
- První vs následný chunk: `first=True` → první věta (40+ znaků),
  `first=False` → 2–3 věty nebo 200 znaků.
- Hard cap: 400 znaků bez terminátoru → force flush.
- Streaming simulace: postupně pushovat substringy, kontrolovat že
  `(emit, remainder)` se drží invariant.

**`tests/test_normalize.py`** (reuse existing tts_cs.py funkcí):
- `normalize_cs` CZ zkratky (`atd.` → "a tak dále").
- `normalize_en` nezasahuje do CZ textu a naopak.

**`tests/test_detect_lang.py`**:
- CZ text s diakritikou → "cs".
- EN text "I love programming" → "en".
- Krátký input → respektuje `prev`.
- Mixed → fallback na `prev`.

**Integration test** (optional, manual):
- `tests/integration/test_turn.sh` — curl skript proti běžícímu
  serveru, ověří `/api/turn` vrací text+audio eventy, cancel funguje.

**Client-side testy**: zatím manuální — browser DevTools, 3 scénáře
z Fáze C. Vanilla JS bez build toolchainu by vyžadoval přidat jest,
to teď neřešíme.

Doporučuji implementovat Fázi A jako první PR (mergeable samostatně,
hned dává hodnotu), Fáze B jako druhý PR. Fáze A + B dohromady by byly
těžko review-able.
