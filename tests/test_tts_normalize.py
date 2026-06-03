"""TTS text normalizace — čísla (num2words), verze, anglické termíny, chunk merge.

Regrese pro bug: čísla se četla po jednotlivých číslicích ("42" → "čtyři dva")
místo kardinálně ("čtyřicet dva"). num2words(lang=cs) to opravuje.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "voice"))
import tts_cs  # noqa: E402


# ──────────────── Čísla CS (hlavní bug) ────────────────

@pytest.mark.parametrize("raw, expected_sub", [
    ("Mám 42 jablek.", "čtyřicet dva"),
    ("Stojí 9875 korun.", "devět tisíc osmset sedmdesát pět"),
    ("Rok 2026.", "dva tisíce dvacet šest"),
    ("Mám 5 aut.", "pět"),
    ("Bylo 100 lidí.", "sto"),
    ("Cena 1000000.", "milion"),
])
def test_cs_integers_cardinal(raw, expected_sub):
    out = tts_cs.normalize_cs(raw)
    assert expected_sub in out, f"{out!r} neobsahuje {expected_sub!r}"
    # Žádné číslice nesmí zůstat
    assert not any(ch.isdigit() for ch in out), f"zůstala číslice: {out!r}"


def test_cs_no_digit_by_digit():
    """Regrese: '42' NESMÍ být 'čtyři dva' (starý bug)."""
    out = tts_cs.normalize_cs("Mám 42 korun.")
    assert "čtyři dva" not in out
    assert "čtyřicet dva" in out


def test_cs_decimal():
    out = tts_cs.normalize_cs("Teplota 3.14 stupňů.")
    assert "tři celá čtrnáct" in out
    assert not any(ch.isdigit() for ch in out)


def test_cs_decimal_comma():
    out = tts_cs.normalize_cs("Mám 3,5 kila.")
    assert "tři celá pět" in out


def test_cs_percent():
    out = tts_cs.normalize_cs("Hlasitost 70%.")
    assert "sedmdesát procent" in out


def test_cs_version_dots():
    """0.1.7 = verze, číst 'tečka', ne 'celá' (regrese: rozbíjelo se)."""
    out = tts_cs.normalize_cs("Verze 0.1.7 je nová.")
    assert "nula tečka jedna tečka sedm" in out
    assert "celá" not in out


def test_cs_ip_address():
    out = tts_cs.normalize_cs("Server 192.168.1.1 doma.")
    assert "tečka" in out
    assert not any(ch.isdigit() for ch in out)


def test_cs_comma_list_not_decimal():
    """'3, 4 a 5' = výčet (mezera za čárkou) → NE desetinné."""
    out = tts_cs.normalize_cs("Mám 3, 4 a 5 jablek.")
    assert "tři" in out and "čtyři" in out and "pět" in out
    assert "celá" not in out


def test_cs_thousands_with_space():
    out = tts_cs.normalize_cs("Cena 1 000 korun.")
    assert "tisíc" in out
    assert not any(ch.isdigit() for ch in out)


def test_cs_long_digit_string_fallback():
    """Velmi dlouhý řetězec číslic (ID/hash) → po číslicích, ne obří číslovka."""
    out = tts_cs.normalize_cs("ID 1234567890123456.")
    # >12 číslic → digit fallback
    assert "jedna" in out and "dva" in out
    assert not any(ch.isdigit() for ch in out)


# ──────────────── Anglické termíny → fonetika ────────────────

@pytest.mark.parametrize("raw, expected_sub", [
    ("Udělej commit.", "komit"),
    ("Pošli email.", "ímejl"),
    ("Otevři pull request.", "pul rikvest"),
    ("Je to open source.", "oupn sors"),
    ("Default nastavení.", "defolt"),
])
def test_cs_english_phonetic(raw, expected_sub):
    out = tts_cs.normalize_cs(raw)
    assert expected_sub in out, f"{out!r} neobsahuje {expected_sub!r}"


def test_cs_english_case_insensitive():
    out = tts_cs.normalize_cs("Udělej COMMIT a Deploy.")
    assert "komit" in out and "deploj" in out


# ──────────────── EN čísla ────────────────

def test_en_integers():
    out = tts_cs.normalize_en("I have 42 apples.")
    assert "forty-two" in out
    assert not any(ch.isdigit() for ch in out)


def test_en_no_cz_abbrev_leak():
    """normalize_en NESMÍ aplikovat CZ fonetiku (GPU → gépéúčko je špatně v EN)."""
    out = tts_cs.normalize_en("My GPU is fast.")
    assert "gépéúčko" not in out


# ──────────────── Chunk merge (short-segment halucinace) ────────────────

def test_chunk_merges_short():
    """Krátké chunky (< MIN_CHARS) se slijí — Chatterbox je jinak halucinuje."""
    chunks = tts_cs.chunk("Ano. Ne. Možná. Tohle je dostatečně dlouhá věta na samostatný chunk.")
    # "Ano." "Ne." "Možná." jsou krátké → slité dohromady
    assert all(len(c) >= tts_cs.MIN_CHARS or len(chunks) == 1 for c in chunks)
    assert len(chunks) < 4  # nesmí být 4 samostatné mini-chunky


def test_chunk_long_text_split():
    """Dlouhý text se dělí pod MAX_CHARS."""
    long = "Toto je věta. " * 40
    chunks = tts_cs.chunk(tts_cs.normalize_cs(long))
    assert all(len(c) <= tts_cs.MAX_CHARS for c in chunks)


def test_chunk_empty():
    assert tts_cs.chunk("") == []


# ──────────────── Robustnost (nikdy nespadnout) ────────────────

def test_normalize_never_crashes():
    weird = ["", "   ", "123", "...", "@@@ 99999999999999999999 @@@",
             "Cena: 1,234,567.89 USD", "🎉 5 emoji 🎉", "0.0.0.0"]
    for w in weird:
        tts_cs.normalize_cs(w)   # nesmí vyhodit
        tts_cs.normalize_en(w)
