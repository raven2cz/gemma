# Bilingual TTS s auto-detekcí jazyka (CZ/EN)

## Cíl

Umožnit voice chatu odpovídat buď česky nebo anglicky, s vhodným TTS modelem pro každý jazyk. Detekce automaticky podle LLM odpovědi; UI override `Auto / CZ / EN`. **Max 1 TTS model v VRAM** — druhý se nahraje při přepnutí (hot-swap).

## Architektura

### VRAM politika (nemění se proti současnosti)
- Chatterbox TTS singleton (2 GB). Whisper (3 GB) + TTS + whisper encoder + VAD + CUDA context ~= 8 GB → LLM musí ven před STT/TTS. To už je hotové.
- **Nové:** jen jeden z `CZ` / `EN` modelů je current. Swap = `_unload_tts_blocking()` + `_load_tts_blocking(lang)`.

### Detekce jazyka
- **Místo:** server-side v `/api/chat` **po dokončení streamu** (znám celou odpověď).
- **User input (whisper output):** detekci na něm **taky** spustíme, **ale** whisper běží `-l cs`, takže anglická věta se přepíše přes CS fonémy. User-side detekce je tedy nespolehlivá → používáme ji **jen pro volbu LLM system promptu**, nikoli pro TTS model.
- **LLM output:** spolehlivý zdroj pravdy pro TTS model.

### Heuristika

```python
CS_DIAC = set("ěščřžýáíéůúňďťĚŠČŘŽÝÁÍÉŮÚŇĎŤ")
CS_STOPWORDS = {"je","se","to","na","si","ve","ze","nebo","ale","takze","prosim","dekuji","jak","co","kdy","kde","jsem","jsi","jsme","delas","muzes","ktery","proc","ano","ne","nebo","neni"}
EN_STOPWORDS = {"the","is","are","and","or","but","you","this","that","with","how","what","when","where","have","has","was","were","will","would","can","could","should"}

def detect_lang(text: str, prev: str = "cs") -> str:
    t = text.strip()
    if len(t) < 10 or len(t.split()) < 3:
        return prev  # příliš krátké → pokračuj v předchozím jazyce
    if any(c in CS_DIAC for c in t):
        return "cs"  # silný signál
    words = set(t.lower().split())
    cs_hits = len(words & CS_STOPWORDS)
    en_hits = len(words & EN_STOPWORDS)
    if cs_hits == en_hits:
        return prev  # tie → pokračuj
    return "cs" if cs_hits > en_hits else "en"
```

### Override dropdown
- UI `Auto / 🇨🇿 CZ / 🇬🇧 EN` (localStorage key `langOverride`, default `auto`).
- Override force-uje **oba**: LLM system prompt i TTS model. `Auto` = detekce.

### System prompt per jazyk
- `cs`: "Odpovídej česky, stručně, v jednom odstavci bez markdown formátování, emoji a bez odrážek."
- `en`: "Respond in English, concise, one paragraph, no markdown, no emojis, no bullet points."

### Ref voice naming konvence
- `ref_*_cs.wav` → dostupný v CZ
- `ref_*_en.wav` → dostupný v EN
- `ref_*.wav` (bez suffixu) → dostupný vždy (univerzální nebo CZ default pro zpětnou kompatibilitu)
- `/api/refs?lang=cs|en` vrátí filtrovaný seznam.
- UI při změně jazyka refetchne `/api/refs` a nastaví default.
- **Fallback:** pokud pro daný jazyk neexistuje žádný ref, TTS běží bez cloning (Chatterbox default voice).

## Implementace

### 1. `voice/webapp/server.py`

#### Globální stav
```python
_TTS_MODEL = None
_TTS_MODULE = None
_TTS_CURRENT_LANG: str | None = None   # "cs" | "en" | None
_TTS_ERROR: str | None = None
_TTS_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
_TTS_READY = asyncio.Event()

LANG_TO_ID = {"cs": "pl", "en": "en"}  # Chatterbox language_id mapping
```

#### `_load_tts_blocking(lang: str)`
- Pokud `_TTS_CURRENT_LANG == lang`, no-op return.
- Pokud je naload jiný jazyk, zavolá `_unload_tts_blocking()` nejdřív.
- Pro `cs`: `from_pretrained` + `load_state_dict(t3_cs.safetensors)` → `t3.to("cuda").eval()`.
- Pro `en`: jen `from_pretrained` bez patche.
- Po úspěchu setne `_TTS_CURRENT_LANG = lang`, loguje `"tts loaded lang=%s in %.1fs, vram=%d MB"`.

#### `_unload_tts_blocking()`
```python
import gc, torch
global _TTS_MODEL, _TTS_CURRENT_LANG
if _TTS_MODEL is None:
    return
before = torch.cuda.memory_allocated()
# Explicit cleanup referencí co Chatterbox drží
try:
    _TTS_MODEL.t3.patched_model = None
    _TTS_MODEL.t3.compiled = False
except Exception:
    pass
del _TTS_MODEL
_TTS_MODEL = None
_TTS_CURRENT_LANG = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()
after = torch.cuda.memory_allocated()
log.info("tts unload: freed %.1f MB", (before - after) / 1024**2)
```

#### Swap s fallbackem
```python
def _switch_tts_blocking(target_lang: str) -> None:
    prev = _TTS_CURRENT_LANG
    if prev == target_lang:
        return
    _unload_tts_blocking()
    try:
        _load_tts_blocking(target_lang)
    except Exception as e:
        log.exception("tts swap to %s failed, restoring %s", target_lang, prev)
        if prev:
            try:
                _load_tts_blocking(prev)
            except Exception:
                log.error("restore %s failed too", prev)
        raise
```

#### `/api/chat` — detekce + system prompt + preload hint
- Přidá param z body `lang_override: "auto" | "cs" | "en"`.
- **Před streamem:** detekuje z posledního user message (fallback `cs`), spočítá `user_lang`. Pokud override ≠ auto, použije ho. Injektuje odpovídající system prompt.
- **Během streamu:** sbírá tokens. Po prvních ~50 znacích a pak po `done` spočítá `out_lang` na dosavadním buffered textu. **Early detekce:** pošle klientovi `{"lang_hint": "en"}` NDJSON event → klient může spustit swap paralelně přes `/api/tts/preload`. Po `done` pošle `{"lang": "en", "done": true}`.

#### `/api/tts/preload` (nový endpoint)
- Body: `{"lang": "cs"|"en"}`.
- Ne-blokující: vrátí hned 200. V executoru zařadí `_switch_tts_blocking(lang)`.
- Umožňuje UI zavolat po `lang_hint` a swap proběhne zatímco LLM dostreamuje.

#### `/api/tts` — přijme `lang`
- Body: `lang: "cs"|"en"` (required, default `cs`).
- Před syntézou: `await loop.run_in_executor(_TTS_EXECUTOR, _switch_tts_blocking, lang)`.
- `_tts_synth_blocking(text, ref_path, fast, lang)`:
  - `language_id = LANG_TO_ID[lang]`
  - `normalize_text(text, lang)` (viz níže).

#### `normalize_text(text, lang)` v `tts_cs.py`
- Pro `cs` — stávající `normalize` (ABBREV, číslovky CS).
- Pro `en` — minimální: `text.strip()`, vypuštění markdown `*_`, žádný ABBREV (jinak "GPU" → "gépéúčko" je špatně).

#### `/api/refs` — filtrování
- Query param `?lang=cs|en`. Bez paramu vrátí vše (zpětná kompat).
- Filtruje: `ref_*_cs.wav` ∪ `ref_*.wav` pro CZ; `ref_*_en.wav` ∪ `ref_*.wav` pro EN.

#### `/api/health` — přidat `tts_loaded_lang`
```python
"tts_ready": _TTS_READY.is_set() and _TTS_ERROR is None,
"tts_loaded_lang": _TTS_CURRENT_LANG,
"tts_error": _TTS_ERROR,
```

#### Logging
- `log.info("lang-detect source=%s text=%r → %s (diac=%d cs=%d en=%d override=%s)")`.

### 2. `voice/tts_cs.py`

- Rozštěp `normalize` → `normalize_cs` (stávající implementace) + `normalize_en` (minimal).
- Alias `normalize = normalize_cs` pro zpětnou kompat.
- Expose `detect_lang(text, prev)` — tak aby server neměl vlastní kopii (single source of truth).

### 3. `voice/webapp/static/app.js`

- Nový state `state.lang = localStorage.getItem('langOverride') || 'auto'`.
- Nový state `state.lastLang = 'cs'` (pro hysterezi mezi turny — ne přímo pro detekci, jen pro UX).
- Nový UI dropdown `langSelect` (HTML), v topbaru.
- `loadRefs()` → volá `/api/refs?lang=${currentLang}` a refetchne při změně jazyka.
- `runChatAndTTS()`:
  - Posílá `lang_override: state.lang` v `/api/chat` body.
  - Parser NDJSON: na `lang_hint` → `fetch('/api/tts/preload', { lang })` fire-and-forget.
  - Na finální `lang` uloží do proměnné → posílá do `/api/tts` body.
- Status text: když dojde `lang_hint` a je jiný než současný loaded, zobraz `"Načítám {EN|CZ} model..."`.

### 4. `voice/webapp/static/index.html`

- Topbar: přidat `<select id="langSelect">` s options `auto`, `cs`, `en` + emoji vlajky.

### 5. `voice/webapp/README.md`

- Sekce "Bilingual" s popisem auto-detekce, override, ref naming konvence, omezení (whisper je CS-only).

## Edge cases (ošetřené)

| Případ | Chování |
|---|---|
| Krátký user input („ano", „yes") | prev lang (start `cs`) |
| CZ bez diakritiky („delas dobry soucet") | stopwords CZ hit → `cs` |
| EN s CZ termínem („I love knedlíky") | diakritika → `cs`, čte CZ modelem (knedlíky OK, I love s CZ akcentem) |
| CZ s EN termínem („nainstaluj Docker") | diakritika nebo CZ stopwords → `cs` |
| Čistě kód/čísla | prev lang |
| LLM odpoví EN navzdory CZ prompt | detect → EN → swap; user ví proč to trvá (status text) |
| Override `CZ`, LLM napíše EN | force CZ prompt + CZ TTS (reads EN with CZ accent) — záměr overridu |
| První turn po F5 | prev = `cs` default |
| Swap OOM | fallback na předchozí model, 503 do UI |
| Stop během swap | executor doběhne, flag `_swap_canceled` přeskočí synth |
| Whisper transkribuje EN → CS | user-side detekce nespolehlivá → jen pro LLM prompt, TTS rozhoduje z outputu |
| Ref voice pro EN neexistuje | fallback: TTS bez cloning (Chatterbox default) |
| Preload endpoint bude volán bezprostředně po lang_hint, ale LLM ještě dostreamovává | swap proběhne v executoru, /api/tts čeká na dokončení |

## Latence odhad

| Scénář | Latence user end → první zvuk |
|---|---|
| Stejný jazyk jako minule, 12B LLM | ~3–5 s (LLM reload) + 2–3 s TTS = 5–8 s |
| Swap jazyka, preload běží paralelně se streamem | ~5–8 s (LLM reload + dominantní swap 5–10 s v parallelu se streamem) = 8–12 s |
| Swap jazyka, preload nefunguje (fallback sekvenční) | ~15 s worst case |

## Co NEDĚLÁME (záměrně odložené)

- Per-sentence streaming TTS (během LLM streamu přehrávat věty po jejich dokončení). Odloženo kvůli složitosti a konfliktu s VRAM budget (LLM + TTS současně).
- Cache detekce. User volá jednou per turn, overkill.
- Swap cancellation přes preempt signal. `_swap_canceled` flag pro po-swap cancel stačí.
- Více jazyků (DE, FR…). Stačí přidat do `LANG_TO_ID` a stopwords, ale UI a ref konvence by potřebovaly refaktor.

## Pořadí implementace

1. `tts_cs.py` — split normalize, expose `detect_lang`.
2. `server.py` — globální stav s `_TTS_CURRENT_LANG`, `_switch_tts_blocking`, nové endpointy/parametry.
3. `app.js` — dropdown, langOverride state, lang_hint handling, preload.
4. `index.html` — dropdown.
5. README update.
6. Smoke test: CZ turn, EN turn, CZ turn (swap × 2), override CZ při EN outputu (force).
