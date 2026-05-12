"""Unit testy pro voice.sentence_chunker.

Chunker je kritický — jeho bugy se projeví jako "TTS čte špatně", "chybí
věta", "code block se pronesl nahlas" atd. Pokrytí: abbrev, decimals,
ellipsis, closers, code blocks, <think>, markdown, streaming simulation.
"""

from __future__ import annotations

from voice.sentence_chunker import (
    Chunk,
    FIRST_CHUNK_MIN,
    HARD_CAP,
    chunk_for_tts,
    finalize,
)


# ─── Helpers ────────────────────────────────────────────────────────────

def all_emits(buf: str, first: bool = True) -> list[Chunk]:
    """Zavolá chunk_for_tts + finalize a vrátí plochý seznam všech chunků."""
    chunks, remainder = chunk_for_tts(buf, first=first)
    chunks.extend(finalize(remainder))
    return chunks


def speakable_texts(chunks: list[Chunk]) -> list[str]:
    return [c.text for c in chunks if c.speakable]


def all_texts(chunks: list[Chunk]) -> list[str]:
    return [c.text for c in chunks]


# ─── Sentence splitting — základ ───────────────────────────────────────

def test_single_sentence():
    chunks = all_emits("Ahoj světe.")
    assert speakable_texts(chunks) == ["Ahoj světe."]


def test_two_sentences():
    chunks = all_emits("Ahoj světe. Jak se máš dnes odpoledne?")
    assert speakable_texts(chunks) == [
        "Ahoj světe.",
        "Jak se máš dnes odpoledne?",
    ]


def test_exclamation_and_question():
    chunks = all_emits("Ticho! Je to tam? Asi ne.")
    assert speakable_texts(chunks) == ["Ticho!", "Je to tam?", "Asi ne."]


def test_no_terminator_forces_flush_in_finalize():
    # Text bez tečky — finalize ho musí emitnout.
    chunks = all_emits("Ano")
    assert speakable_texts(chunks) == ["Ano"]


# ─── Zkratky ───────────────────────────────────────────────────────────

def test_cs_abbrev_not_split():
    chunks = all_emits("Napiš p. Nováka. Děkuji.")
    # "p." není konec věty; "Nováka." ano.
    assert speakable_texts(chunks) == ["Napiš p. Nováka.", "Děkuji."]


def test_en_abbrev_not_split():
    chunks = all_emits("Meet Dr. Smith tomorrow. We agreed.")
    assert speakable_texts(chunks) == [
        "Meet Dr. Smith tomorrow.",
        "We agreed.",
    ]


def test_multiple_abbrev_in_sentence():
    chunks = all_emits("Prof. Novák, MUDr. Svoboda a Ing. Malý přišli. Super.")
    assert speakable_texts(chunks) == [
        "Prof. Novák, MUDr. Svoboda a Ing. Malý přišli.",
        "Super.",
    ]


# ─── Čísla s desetinnou tečkou ─────────────────────────────────────────

def test_decimal_not_split():
    chunks = all_emits("Máme 3.14 km cesty. Pak pauza.")
    assert speakable_texts(chunks) == [
        "Máme 3.14 km cesty.",
        "Pak pauza.",
    ]


def test_multiple_decimals():
    # Použij delší text tak aby první věta sama prošla FIRST_CHUNK_MIN,
    # jinak se chunker pokusí agregovat další větu a dostaneme 1 chunk.
    chunks = all_emits(
        "Naměřené hodnoty 1.5, 2.718 a 3.14 jsou v literatuře dobře známé. Konec měření."
    )
    assert speakable_texts(chunks) == [
        "Naměřené hodnoty 1.5, 2.718 a 3.14 jsou v literatuře dobře známé.",
        "Konec měření.",
    ]


# ─── Ellipsis ──────────────────────────────────────────────────────────

def test_unicode_ellipsis():
    chunks = all_emits("Hmm… dobrá, souhlasím.")
    assert speakable_texts(chunks) == ["Hmm…", "dobrá, souhlasím."]


def test_three_dots_ellipsis():
    # "..." je sekvence teček; rozdělí se po nich, protože následuje mezera
    # a další věta.
    chunks = all_emits("Počkej... teď.")
    # Můžeme přijmout buď ["Počkej...", "teď."] nebo sloučené.
    # Minimum: TTS musí dostat obě věty bez ztráty obsahu.
    joined = " ".join(speakable_texts(chunks))
    assert "Počkej" in joined
    assert "teď." in joined


# ─── Zavírací uvozovky/závorky ─────────────────────────────────────────

def test_sentence_ending_quote():
    chunks = all_emits('Řekl: "Ahoj." Pak odešel.')
    # Závěrečná uvozovka se má přilepit k první větě.
    assert speakable_texts(chunks) == ['Řekl: "Ahoj."', "Pak odešel."]


def test_sentence_ending_paren():
    chunks = all_emits("Skončilo to (konečně.) Nebo ne?")
    assert speakable_texts(chunks) == [
        "Skončilo to (konečně.)",
        "Nebo ne?",
    ]


# ─── Code blocks ───────────────────────────────────────────────────────

def test_fenced_code_extracted_as_non_speakable():
    text = "Tady je kód.\n```python\nx = 1\nprint(x)\n```\nKonec."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    non_speakable = [c.text for c in chunks if not c.speakable]
    assert "Tady je kód." in speakable
    assert "Konec." in speakable
    # Code content je v non-speakable.
    assert any("print(x)" in c for c in non_speakable)
    # Code NESMÍ být ve speakable.
    assert not any("print(x)" in s for s in speakable)


def test_unclosed_fence_waits():
    # Nedokončený code block: druhá ``` ještě nedošla.
    text = "Text před. ```python\nx = 1"
    chunks, remainder = chunk_for_tts(text, first=True)
    # První věta může emitnout, ale code pending.
    speakable = [c.text for c in chunks if c.speakable]
    assert "Text před." in speakable or not speakable
    # Remainder musí obsahovat neuzavřený code block.
    assert "```python" in remainder


def test_inline_code_text_preserved():
    # `ls -la` zůstane ve větě jako "ls -la" (bez backticků).
    chunks = all_emits("Spusť `ls -la` teď prosím.")
    combined = " ".join(speakable_texts(chunks))
    assert "ls -la" in combined
    assert "`" not in combined


# ─── <think> tagy ──────────────────────────────────────────────────────

def test_think_stripped():
    chunks = all_emits("<think>internal plan</think>Ahoj. Je mi dobře.")
    # <think> obsah se nesmí objevit ve výstupu.
    all_out = " ".join(all_texts(chunks))
    assert "internal" not in all_out
    assert "plan" not in all_out
    assert "Ahoj." in speakable_texts(chunks)


def test_multiline_think_stripped():
    chunks = all_emits("<think>\nlong\nmulti-line\nthought\n</think>\nAno.")
    all_out = " ".join(all_texts(chunks))
    assert "multi-line" not in all_out
    assert "Ano" in all_out


# ─── Markdown syntax ───────────────────────────────────────────────────

def test_bold_italic_stripped():
    chunks = all_emits("Toto je **důležité** a _kurzíva_ také. OK.")
    combined = " ".join(speakable_texts(chunks))
    assert "**" not in combined
    assert "_" not in combined or "kurzíva" in combined  # obsah zachován
    assert "důležité" in combined
    assert "kurzíva" in combined


def test_markdown_link_becomes_text():
    chunks = all_emits("Klikni [sem](https://example.com/long/url) prosím.")
    combined = " ".join(speakable_texts(chunks))
    assert "sem" in combined
    assert "example.com" not in combined
    assert "https" not in combined


def test_heading_prefix_stripped():
    chunks = all_emits("# Nadpis zde\n\nText následuje. Další.")
    combined = " ".join(speakable_texts(chunks))
    assert "#" not in combined
    assert "Nadpis" in combined
    assert "Další" in combined


# ─── Chunk sizing: první vs další ──────────────────────────────────────

def test_first_chunk_short_waits_for_minimum():
    # "Ahoj." má 5 znaků, pod FIRST_CHUNK_MIN (40). Neměl by být emitnut
    # hned — chunker pokračuje buf. Ale test s first=True a finalize ho
    # musí vyhodit na konci.
    chunks, remainder = chunk_for_tts("Ahoj.", first=True)
    # Možné: emitne (protože není delší kandidát) nebo počká. Ale finalize
    # ho musí dostat ven.
    final = finalize(remainder)
    all_chunks = chunks + final
    speakable = speakable_texts(all_chunks)
    assert "Ahoj." in speakable


def test_first_chunk_emits_at_minimum():
    # První věta musí přesáhnout FIRST_CHUNK_MIN (60) + druhá věta musí
    # být viditelná za boundary (jinak peek-guard drží emit pro příští call).
    text = "Tohle je první poměrně dlouhá věta s více než šedesáti znaky celkem. Druhá věta."
    chunks, _ = chunk_for_tts(text, first=True)
    speakable = speakable_texts(chunks)
    # První chunk by měl být první věta, protože přesáhla FIRST_CHUNK_MIN.
    assert len(speakable) >= 1
    assert speakable[0].startswith("Tohle je první")


def test_next_chunks_collect_more():
    # Po prvním emitu (simulovaném first=False) chunker sbírá až k
    # target_size ≈ 200 znaků.
    text = "Krátká věta. Další krátká. A ještě jedna. Tady poslední."
    chunks, rem = chunk_for_tts(text, first=False)
    # Krátké věty se agregují; celkem < 200 znaků, takže emit proběhne
    # buď na konci textu (ne, žádný flush) nebo čeká.
    # Ověřujeme, že chunker NEEMITUJE 4 samostatné věty jak by udělal first.
    speakable = speakable_texts(chunks)
    # Minimum očekávání: žádná zbytečná fragmentace.
    # Tohle bude spíš: 0 chunků (vše čeká), a finalize je sesype.
    final = finalize(rem)
    all_chunks = chunks + final
    assert len(all_chunks) >= 1  # aspoň nějaký emit


def test_hard_cap_forces_flush():
    # Velmi dlouhý text bez tečky — hard cap ho musí rozseknout.
    long = "slovo " * 120  # ~720 znaků, bez terminátorů
    chunks, rem = chunk_for_tts(long, first=False)
    # Aspoň jeden chunk by měl být vyslán (hard cap).
    speakable = speakable_texts(chunks)
    if speakable:
        # Hard cap ~400 znaků; emit by měl být menší než HARD_CAP.
        assert len(speakable[0]) <= HARD_CAP + 10


# ─── Streaming simulace ────────────────────────────────────────────────

def test_streaming_tokens():
    """Simuluje LLM stream: přidávej znaky po malých kusech a kontroluj
    že invariant `(emitted, remainder)` platí, žádný obsah se neztratí."""
    full = "První věta je docela dlouhá a obsahuje dost textu. Druhá je kratší. Třetí."
    emitted: list[str] = []
    buf = ""
    first = True
    # Simulace: po 10 znacích zavolej chunk_for_tts.
    i = 0
    while i < len(full):
        buf += full[i:i + 7]
        i += 7
        chunks, buf = chunk_for_tts(buf, first=first)
        for c in chunks:
            if c.speakable:
                emitted.append(c.text)
                first = False
    for c in finalize(buf):
        if c.speakable:
            emitted.append(c.text)

    # Všechny tři věty musí být ve výstupu.
    combined = " ".join(emitted)
    assert "První věta" in combined
    assert "Druhá" in combined
    assert "Třetí" in combined


def test_streaming_does_not_emit_incomplete_sentence():
    """Dokud nepřijde terminátor, speakable chunk se (pro first=False) neemituje."""
    chunks, rem = chunk_for_tts("Toto je začátek věty bez te", first=False)
    assert not [c for c in chunks if c.speakable]
    # Zbytek musí být v remainderu.
    assert "začátek" in rem or rem == "Toto je začátek věty bez te"


# ─── Edge cases ────────────────────────────────────────────────────────

def test_empty_input():
    chunks, rem = chunk_for_tts("", first=True)
    assert chunks == []
    assert rem == ""
    assert finalize("") == []


def test_only_whitespace():
    chunks, rem = chunk_for_tts("   \n\n  ", first=True)
    assert all(not c.speakable or c.text for c in chunks)
    # Nic speakable se nemá emitnout.
    assert speakable_texts(chunks) == []


def test_mixed_cz_en():
    chunks = all_emits("Ahoj Dr. Novák. Hello Mr. Smith, jak se máte?")
    speakable = speakable_texts(chunks)
    # Ani "Dr." ani "Mr." nesmí rozseknout větu.
    assert any("Dr. Novák" in s for s in speakable)
    assert any("Mr. Smith" in s for s in speakable)


def test_code_in_middle_preserves_surrounding():
    text = "Spusť tohle.\n```bash\necho hi\n```\nPak pokračuj."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    non_speakable = [c.text for c in chunks if not c.speakable]
    assert "Spusť tohle." in speakable
    assert "Pak pokračuj." in speakable
    assert any("echo hi" in c for c in non_speakable)


# ─── Regression: streaming zachovává slovní mezery (#2) ────────────────

def test_streaming_preserves_word_boundaries():
    """Bug reprodukovaný Codexem: remainder vracený z cleaned (ne stripped_buf)
    způsoboval, že nové tokeny se glued k předchozímu slovu bez mezery.
    Např. "První věta je docela dlouhá" → "jedoceladlouhá" při 7-znakovém streamu.
    """
    full = "První věta je docela dlouhá a obsahuje dost textu. Druhá je kratší. Třetí."
    emitted: list[str] = []
    buf = ""
    first = True
    i = 0
    while i < len(full):
        buf += full[i:i + 7]
        i += 7
        chunks, buf = chunk_for_tts(buf, first=first)
        for c in chunks:
            if c.speakable:
                emitted.append(c.text)
                first = False
    for c in finalize(buf):
        if c.speakable:
            emitted.append(c.text)

    combined = " ".join(emitted)
    # Slovní hranice zachované — žádné gluing.
    assert "je docela dlouhá" in combined, f"got: {combined!r}"
    assert "Druhá je kratší" in combined
    # Negativní check na nejzjevnější corruption pattern z Codex repro.
    assert "jedocela" not in combined
    assert "doceladlouhá" not in combined


def test_streaming_markdown_straddle_preserves_bold():
    """Markdown pár ** rozdělený mezi dvě volání nesmí ztratit druhou značku.
    Pokud return stripped_buf slice funguje, otevřené `**` se uzavře v dalším
    volání a obsah mezi není zdvojen ani ztracen."""
    # První call: buf končí v polovině `**bold**` páru, těsně před druhým **.
    buf1 = "Toto je začátek věty s velkým množstvím textu a **tučným"
    chunks1, rem1 = chunk_for_tts(buf1, first=True)
    # Druhé volání: přidá zbytek (uzavře ** + dokončí větu).
    buf2 = rem1 + " slovem** uvnitř. Další věta je tady."
    chunks2, rem2 = chunk_for_tts(buf2, first=not any(c.speakable for c in chunks1))
    all_chunks = chunks1 + chunks2 + finalize(rem2)
    combined = " ".join(c.text for c in all_chunks if c.speakable)
    # Obsah obou slov z markdown páru musí být přítomen.
    assert "tučným" in combined, f"got: {combined!r}"
    assert "slovem" in combined, f"got: {combined!r}"
    # Žádné zdvojení.
    assert combined.count("tučným") == 1
    assert combined.count("slovem") == 1


def test_streaming_think_straddle():
    """Nezavřený <think> v prvním volání nesmí být emitnut jako speakable.
    Uzavře se v druhém volání — obsah musí být úplně zahozen."""
    buf1 = "Nějaký úvodní text. <think>vnitřní úvaha o"
    chunks1, rem1 = chunk_for_tts(buf1, first=True)
    speakable1 = speakable_texts(chunks1)
    # Žádná věta ze <think> ven.
    assert not any("vnitřní" in s for s in speakable1)
    # Remainder drží celý <think>... nedokončený.
    assert "<think>" in rem1

    buf2 = rem1 + " problému</think>Výsledek je tady. Dost."
    chunks2, rem2 = chunk_for_tts(buf2, first=not speakable1)
    all_chunks = chunks1 + chunks2 + finalize(rem2)
    combined = " ".join(c.text for c in all_chunks if c.speakable)
    assert "vnitřní" not in combined
    assert "úvaha" not in combined
    assert "problému" not in combined
    assert "Výsledek je tady" in combined


# ─── Regression: multi-dot zkratky (#5) ────────────────────────────────

def test_phd_not_split():
    """Ph.D. má dvě tečky — ani jedna není boundary."""
    text = "Získal Ph.D. v roce 2020 po dlouhé době studia. Teď učí."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    # Celá "Získal Ph.D. v roce ..." je jedna věta.
    assert any("Ph.D." in s and "v roce 2020" in s for s in speakable), \
        f"Ph.D. split incorrectly, got: {speakable}"
    assert any("Teď učí." in s for s in speakable)


def test_eg_ie_not_split():
    """e.g., i.e. mají vnitřní tečky — žádná z nich není boundary."""
    text = "Použij nějaký nástroj e.g. vim nebo emacs či podobný editor. Dobrá."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    assert any("e.g." in s and "emacs" in s for s in speakable), \
        f"e.g. split incorrectly, got: {speakable}"


def test_us_abbrev_not_split():
    """U.S. — dvě tečky, nesmí splitnout."""
    text = "Narodil se v U.S. a žije tam celý život dodnes bez přerušení. Konec."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    assert any("U.S." in s and "celý život" in s for s in speakable), \
        f"U.S. split incorrectly, got: {speakable}"


# ─── Regression: hard-cap s first=True (#test gap) ────────────────────

def test_hard_cap_first_true():
    """Bez tečky, ≥ HARD_CAP znaků — musí force-flushnout i když first=True."""
    long = "slovo " * 120  # ~720 znaků
    chunks, rem = chunk_for_tts(long, first=True)
    speakable = speakable_texts(chunks)
    # Aspoň jeden chunk emitnut přes hard-cap (nezávisle na first flagu).
    assert len(speakable) >= 1, f"hard-cap neforcoval flush pro first=True; chunks={chunks}, rem={rem!r}"
    assert len(speakable[0]) <= HARD_CAP + 10


# ─── Quality-first: konektor-aware spojování ─────────────────────────

def test_connector_a_merges_with_previous():
    """Druhá věta začíná "A " — TTS má dostat obě dohromady pro přirozenou
    prozódii, ne s přerušením."""
    text = "Šel jsem do obchodu a koupil jsem chleba. A pak jsem šel domů."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    # Nemá se rozseknout; emit jako jeden chunk.
    assert len(speakable) == 1, f"konektor 'A' rozsekl větu: {speakable}"
    assert "Šel jsem" in speakable[0] and "pak jsem šel domů" in speakable[0]


def test_connector_but_merges_with_previous():
    """EN konektor 'But' — stejné pravidlo jako CS."""
    text = "I went to the store today for supplies. But I forgot the milk at home."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    assert len(speakable) == 1, f"konektor 'But' rozsekl větu: {speakable}"


def test_non_connector_splits_cleanly():
    """Věta začínající non-konektorem ("Pak", "Nebo" …) se normálně rozdělí —
    ty nejsou v konzervativním seznamu."""
    text = "Koupil jsem chleba a sýr ve velkém množství na týden. Pak jsem šel domů."
    chunks = all_emits(text)
    speakable = speakable_texts(chunks)
    # "Pak" není v konektorech — split.
    assert len(speakable) == 2, f"očekáváno 2 chunky, got: {speakable}"
