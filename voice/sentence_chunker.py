"""Streaming sentence chunker pro TTS pipeline.

Bere buffer LLM streamu a postupně emituje "speakable" chunky vhodné pro
syntézu TTS. Nespeakable části (code bloky, <think>) projde jako
non-speakable chunky — klient je zobrazí ve transkriptu, ale nepošle do
audio queue.

Klíčová pravidla (kvalita > rychlost):
- Nechceme agresivní krátký první chunk — radši počkat na pořádnou větu
  s přirozenou prozódií.
- Následné chunky cílí na ~220 znaků (dost na dobré znění, ne moc na
  pomalost).
- Konektorové věty (začínající "A ", "Ale ", "And ", "But "…) se
  nesekají — TTS je přednese dohromady s předchozí větou.
- Nesekat před zkratkami (CZ + EN), desetinnými čísly, ellipsí.
- Zachovat zavírací uvozovky/závorky u předchozí věty.
- Markdown fenced code bloky → non-speakable chunk (emit pro UI, ne pro TTS).
- `<think>…</think>` → zahodit úplně.

Viz tests/test_sentence_chunker.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# ─── Konfigurace ────────────────────────────────────────────────────────

# První chunk: moderátní práh — radši počkat na plnou větu než vyplivnout
# "Dobře.". Dřív bylo 40 (agresivní TTFB), teď 60.
FIRST_CHUNK_MIN = 60
# Nad tuhle velikost ignorujeme konektor-peek a emitujeme první chunk.
FIRST_CHUNK_MAX = 260
# Cílová velikost pro následné chunky.
CHUNK_TARGET = 220
# Nad tuhle velikost ignorujeme konektor-peek a emitujeme i přes konektor.
MAX_CHUNK_SIZE = 380
# Force flush text bez terminátoru (LLM halucinuje dlouhý běh bez tečky).
HARD_CAP = 500

# Konektory = slova na začátku věty, která tak těsně navazují na
# předchozí, že TTS zní přirozeněji, když se přednesou dohromady.
# Drž se konzervativního seznamu — false-positive (splnění se drobně
# odlišných vět) je horší než false-negative (nerozšklíbnout spojení).
CS_CONNECTORS = {
    "a", "ale", "avšak", "však", "jenže", "přesto", "nýbrž",
}
EN_CONNECTORS = {
    "and", "but", "yet", "nor",
}
CONNECTORS = CS_CONNECTORS | EN_CONNECTORS

# CZ zkratky (převzato z tts_cs.ABBREV jmenných klíčů + běžné CZ tituly).
CS_ABBREV = {
    "p.", "pí.", "pan.", "mj.", "tj.", "tzv.", "tzn.", "apod.", "atd.",
    "např.", "resp.", "cca.", "kpt.", "gen.", "prof.", "doc.", "ing.",
    "mgr.", "mudr.", "ph.d.", "č.", "čís.", "str.", "obr.", "tab.",
    "kap.", "r.", "stol.", "sv.", "nar.", "zemř.",
}
# EN zkratky — titly, časté obchodní/akademické.
EN_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "jr.", "sr.", "st.", "etc.",
    "e.g.", "i.e.", "vs.", "inc.", "ltd.", "co.", "corp.", "p.s.",
    "a.m.", "p.m.", "u.s.", "u.k.", "u.s.a.",
}
ABBREV = CS_ABBREV | EN_ABBREV

# Terminátory věty (včetně unicode ellipsis).
TERMINATORS = ".!?…"
CLOSERS = '"\')]»"«”’'  # uvozovky/závorky přilepit k předchozí větě

# ─── Regex pro pre-processing ─────────────────────────────────────────

# Fenced code block: ```...``` (s volitelným jazykem). DOTALL aby ``` neskončilo na \n.
FENCED_CODE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)
# Inline code: `...` (single backtick, non-greedy, ne-přes-newline).
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
# <think> tag.
THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
# Markdown odkaz [text](url) → "text".
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# Markdown formátování (*, _, #, ~, >) — smazat značky, ponechat text.
FORMAT_CHARS_RE = re.compile(r"[*_~#>]+")
# HTML tagy (light — pro robustnost, LLM občas vyplivne <br> apod.).
HTML_TAG_RE = re.compile(r"<[^>]+>")


# ─── Datové typy ──────────────────────────────────────────────────────

ChunkKind = Literal["sentence", "code", "skipped"]


@dataclass
class Chunk:
    """Vrácený kus textu. `speakable=True` = pošli do TTS; jinak jen UI."""
    text: str
    speakable: bool
    kind: ChunkKind


# ─── Speakable text extrakce ──────────────────────────────────────────

def _extract_code_blocks(buf: str) -> tuple[list[tuple[int, int, str]], str]:
    """Najde dokončené fenced code bloky v bufferu. Vrátí (spans, buffer_bez_bloků).

    Každý span je (start, end, content_without_fences). Nedokončený code block
    (otevřené ``` bez uzavření) se ponechá v bufferu — počkáme, až dojde.
    """
    spans: list[tuple[int, int, str]] = []
    # Collect matches, then rebuild string without them.
    matches = list(FENCED_CODE_RE.finditer(buf))
    if not matches:
        return spans, buf
    out_parts: list[str] = []
    cursor = 0
    for m in matches:
        out_parts.append(buf[cursor:m.start()])
        content = m.group(0)
        # Strip leading ```lang\n and trailing ```
        inner = re.sub(r"^```[^\n]*\n", "", content)
        inner = re.sub(r"```\s*$", "", inner)
        spans.append((m.start(), m.end(), inner.strip()))
        cursor = m.end()
    out_parts.append(buf[cursor:])
    return spans, "".join(out_parts)


def _has_unclosed_fence(buf: str) -> bool:
    """True pokud v bufferu je neuzavřený ```. Chunker pak nebude emitovat
    text za poslední nezavřenou fencí (počká na víc tokenů)."""
    count = buf.count("```")
    return count % 2 == 1


def _clean_for_tts(text: str) -> str:
    """Length-preserving čištění: markdown / HTML / `<think>` / inline code se
    nahradí mezerami tak, aby `len(cleaned) == len(text)` a pozice v `cleaned`
    odpovídaly pozicím ve vstupu 1:1. Volající pak může vrátit `stripped_buf`
    slice jako remainder bez mapování pozic. Whitespace collapse/strip dělá
    volající per emitted chunk."""
    def _blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())

    # <think>…</think> — zahoď celý blok (nahraď mezerami).
    text = THINK_RE.sub(_blank, text)
    # Inline code `xxx` → mezera + content + mezera (length-preserving: 2 backticks → 2 spaces).
    text = INLINE_CODE_RE.sub(lambda m: " " + m.group(1) + " ", text)
    # Markdown link [label](url) → label + padding. Délka stejná.
    def _link(m: re.Match) -> str:
        label = m.group(1)
        return label + " " * ((m.end() - m.start()) - len(label))
    text = LINK_RE.sub(_link, text)
    # HTML tagy → mezery stejné délky.
    text = HTML_TAG_RE.sub(_blank, text)
    # Markdown formátování *_~#> → mezera každý znak (ne collapse).
    text = FORMAT_CHARS_RE.sub(_blank, text)
    return text


def _normalize_chunk_text(text: str) -> str:
    """Collapse whitespace + strip — volá se per emitted chunk, ne na celé
    `cleaned` (to musí zachovat pozice)."""
    return re.sub(r"\s+", " ", text).strip()


# ─── Sentence splitter ────────────────────────────────────────────────

def _is_abbrev_before(text: str, pos: int) -> bool:
    """Je tečka na `pos` součástí zkratky? Pokrývá i multi-dot zkratky
    (Ph.D., e.g., i.e., U.S.) — tečka uvnitř zkratky nesmí být sentence
    boundary. Porovnáváme celou zkratku na pozici od začátku tokenu, ne jen
    token končící touhle tečkou."""
    if pos < 0 or pos >= len(text) or text[pos] != ".":
        return False
    # Start of token (zpět až k whitespace nebo začátku).
    start = pos
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    # Je některá zkratka z ABBREV umístěna na `start` a obsahuje `pos`?
    # (multi-dot zkratky jako "ph.d." mají `pos` buď na prvním, nebo druhém dotu).
    for abbrev in ABBREV:
        end = start + len(abbrev)
        if end > len(text):
            continue
        if start <= pos < end and text[start:end].lower() == abbrev:
            return True
    return False


def _is_decimal(text: str, pos: int) -> bool:
    """Je tečka na `pos` desetinná (3.14)?"""
    if pos < 0 or pos >= len(text) or text[pos] != ".":
        return False
    if pos == 0 or pos + 1 >= len(text):
        return False
    return text[pos - 1].isdigit() and text[pos + 1].isdigit()


def _find_sentence_boundaries(text: str) -> list[int]:
    """Vrátí indexy (exclusive) kde končí věta. Poslední index může být
    len(text) pokud text končí terminátorem."""
    boundaries: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in TERMINATORS:
            # Je to decimal tečka?
            if c == "." and _is_decimal(text, i):
                i += 1
                continue
            # Je to zkratka?
            if c == "." and _is_abbrev_before(text, i):
                i += 1
                continue
            # Sbalit případné následné terminátory (!!!, …, ?!) a zavírky.
            j = i + 1
            while j < n and text[j] in TERMINATORS:
                j += 1
            while j < n and text[j] in CLOSERS:
                j += 1
            # Za terminátorem musí být whitespace (nebo EOS) aby se to
            # počítalo jako konec věty. Jinak je to třeba URL nebo 3.14.5.
            if j < n and not text[j].isspace():
                i = j
                continue
            boundaries.append(j)
            i = j
        else:
            i += 1
    return boundaries


# ─── Peek: co následuje po hranici věty ──────────────────────────────

_PEEK_WORD_SKIP = set(" \t\n\r,;:'\"«»„“()[]–—-")


def _peek_next_word(text: str, start: int) -> str:
    """Vrátí první slovo po pozici `start` (přeskočí whitespace a úvodní
    interpunkci). Prázdný string = za `start` není žádné slovo."""
    i = start
    n = len(text)
    while i < n and text[i] in _PEEK_WORD_SKIP:
        i += 1
    if i >= n:
        return ""
    j = i
    while j < n and text[j].isalpha():
        j += 1
    return text[i:j].lower()


def _has_peek_content(text: str, start: int) -> bool:
    """True pokud za pozicí `start` je alespoň nějaký non-whitespace obsah.
    Používá se ve streamingu: když stream ještě nedodal další slovo po
    boundaries, nechceme emitovat (další slovo by mohl být konektor)."""
    i = start
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i < n


def _starts_with_connector(text: str, start: int) -> bool:
    """True pokud první slovo za `start` je konektor (a, ale, and, but, …).
    False i v případě prázdna — volající musí separátně testovat peek-content."""
    word = _peek_next_word(text, start)
    return word in CONNECTORS


# ─── Hlavní API ────────────────────────────────────────────────────────

def chunk_for_tts(
    buf: str,
    *,
    first: bool,
) -> tuple[list[Chunk], str]:
    """Z bufferu vyextrahuj kompletní chunky, vrať zbytek k dalšímu volání.

    Volá se opakovaně jak LLM streamuje tokeny: `chunks, buf = chunk_for_tts(buf+delta, first=not_emitted_yet)`.
    Po skončení streamu pošli zbytek přes `finalize()` — force flush.

    Strategy (quality > speed):
      1. Emit jenom na SENTENCE BOUNDARY (tečka, otazník, …). Nikdy mid-sentence.
      2. Pokud další věta začíná KONEKTOREM ("A", "Ale", "And", "But" …),
         POČKEJ a spoj ji s předchozí — TTS tak dostane přirozenou prozódii.
      3. Ve streamu, pokud ještě nevidíme obsah za boundary, POČKEJ — další
         token by mohl být konektor.
      4. HARD_CAP = emergency flush (LLM dlouhý běh bez tečky).

    Parametry:
        buf: aktuální akumulovaný LLM output (speakable + non-speakable dohromady).
        first: True pokud ještě nebyl emitován žádný speakable chunk v tomto turnu.
               První chunk má mírně menší target (FIRST_CHUNK_MIN), aby audio
               uživatel neslyšel po dlouhé pauze.

    Vrací:
        (chunks_k_emitu, remaining_buffer).
    """
    out: list[Chunk] = []

    # 1. Vyseknu dokončené code bloky — ty jdou ven jako non-speakable,
    # nezávisle na sentence chunku.
    code_spans, stripped_buf = _extract_code_blocks(buf)
    for _start, _end, code_content in code_spans:
        if code_content:
            out.append(Chunk(text=code_content, speakable=False, kind="code"))

    # 2. Pokud je neuzavřený fence v `stripped_buf` (teoreticky nemůže být,
    # protože _extract_code_blocks pracuje jen na párech), konzervativně
    # počkáme. Pro jistotu:
    if _has_unclosed_fence(stripped_buf):
        return out, stripped_buf

    # 3. Length-preserving cleaning: pozice v `cleaned` se 1:1 mapují do
    # `stripped_buf`. Remainder vracíme vždy jako slice z `stripped_buf` —
    # zachovává originální whitespace, nezavřené `**` / `<think>` /
    # inline code pro příští volání (kde se už můžou zavřít).
    cleaned = _clean_for_tts(stripped_buf)
    assert len(cleaned) == len(stripped_buf), "cleaning musí být length-preserving"
    if not cleaned.strip():
        # Jen whitespace / tagy — drop leading ws, zbytek vrátit pro další round.
        return out, stripped_buf.lstrip()

    # 4. Najdi sentence boundaries v cleaned textu.
    boundaries = _find_sentence_boundaries(cleaned)

    # 5. Sbírej věty do chunku.
    target = FIRST_CHUNK_MIN if first else CHUNK_TARGET
    max_size = FIRST_CHUNK_MAX if first else MAX_CHUNK_SIZE

    last_emit_end = 0  # pozice v cleaned (= v stripped_buf díky length-preserving)
    for b in boundaries:
        # POZOR: `b - last_emit_end` inflates kvůli length-preserving cleaningu
        # (strippované markdown/think jsou teď mezery). Sizing rozhodujeme podle
        # normalized content length — jinak by krátká věta se stripnutým prefixem
        # vypadala velká.
        text = _normalize_chunk_text(cleaned[last_emit_end:b])
        if not text:
            continue
        chunk_len = len(text)

        # Ještě jsme nepřekročili target — čekej na další větu.
        if chunk_len < target:
            continue

        # Nad max_size vždy emit (zastavíme nekonečný růst chunku).
        if chunk_len < max_size:
            # Ještě nevidíme obsah za boundary — další token by mohl být
            # konektor. Ve streamu čekáme.
            if not _has_peek_content(cleaned, b):
                continue
            # Další věta začíná konektorem — spoj je dohromady.
            if _starts_with_connector(cleaned, b):
                continue

        out.append(Chunk(text=text, speakable=True, kind="sentence"))
        last_emit_end = b
        if first:
            first = False
            target = CHUNK_TARGET
            max_size = MAX_CHUNK_SIZE

    # 6. Remainder handling — hard cap pro text bez terminátoru:
    cleaned_tail = cleaned[last_emit_end:]
    if len(cleaned_tail) >= HARD_CAP:
        # Force flush na poslední mezeře uvnitř prvních ~HARD_CAP znaků.
        cut = cleaned_tail.rfind(" ", 0, HARD_CAP)
        if cut > HARD_CAP // 2:  # Mezera musí být v rozumné pozici.
            text = _normalize_chunk_text(cleaned_tail[:cut])
            if text:
                out.append(Chunk(text=text, speakable=True, kind="sentence"))
            # Zbytek vrátíme z stripped_buf (zachová orig. whitespace
            # + nezavřené markdown/think pro další volání).
            return out, stripped_buf[last_emit_end + cut:].lstrip()

    # 7. Remainder = stripped_buf[last_emit_end:] — to je KRITICKÉ.
    # Kdybychom vrátili `cleaned[last_emit_end:]`, přišli bychom o:
    #   - whitespace (collapsed/stripped by volala _normalize_chunk_text),
    #   - nezavřené markdown markery (např. `**` čekající na uzavření v
    #     dalším tokenu),
    #   - nezavřené `<think>` nebo inline code, kterým LLM teprve dopíše
    #     closing tag.
    # Length-preserving cleanup garantuje, že last_emit_end je ve stripped_buf
    # stejná pozice jako v cleaned.
    return out, stripped_buf[last_emit_end:]


def finalize(buf: str) -> list[Chunk]:
    """Při ukončení LLM streamu vyprázdni zbylý buffer. Force flush i bez
    terminátoru (LLM třeba skončil slovem bez tečky). Konektor-navazující
    věty spojí (stejně jako chunk_for_tts) — tail se ale vždy emituje."""
    out: list[Chunk] = []

    code_spans, stripped_buf = _extract_code_blocks(buf)
    for _start, _end, code_content in code_spans:
        if code_content:
            out.append(Chunk(text=code_content, speakable=False, kind="code"))

    # Codex audit HIGH: neuzavřený fence (model dotekl max_tokens / EOS uprostřed
    # ```code... bez ukončení). `_extract_code_blocks` ho nevyzobne (hledá jen
    # uzavřené ```), takže by tail propadl do speakable cleanu → TTS by četl
    # kód. Tady ho zachytíme: tail od posledního ``` jde jako code chunk,
    # cokoli PŘED ním zůstane pro normální speakable processing.
    if _has_unclosed_fence(stripped_buf):
        last_fence = stripped_buf.rfind("```")
        if last_fence != -1:
            code_tail = stripped_buf[last_fence:].lstrip("`").lstrip("\n")
            if code_tail.strip():
                out.append(Chunk(text=code_tail, speakable=False, kind="code"))
            stripped_buf = stripped_buf[:last_fence]

    cleaned = _clean_for_tts(stripped_buf)
    if not cleaned.strip():
        return out

    # Rozsekat po větách. Pokud další věta začíná konektorem a aktuální
    # cumul je pod MAX, drž to dohromady s další.
    boundaries = _find_sentence_boundaries(cleaned)
    cursor = 0
    for b in boundaries:
        text = _normalize_chunk_text(cleaned[cursor:b])
        if not text:
            continue
        # Konektor-aware spojování (jen pokud máme kam růst).
        if len(text) < MAX_CHUNK_SIZE and _starts_with_connector(cleaned, b):
            continue
        out.append(Chunk(text=text, speakable=True, kind="sentence"))
        cursor = b
    # Zbytek (bez terminátoru nebo poslední věta) jako poslední chunk.
    tail = _normalize_chunk_text(cleaned[cursor:])
    if tail:
        out.append(Chunk(text=tail, speakable=True, kind="sentence"))
    return out
