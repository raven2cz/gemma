#!/usr/bin/env python3
"""OpenAI TTS backend — alternativa k lokálnímu Chatterboxu.

Volá OpenAI /v1/audio/speech. Na rozdíl od Chatterboxu zvládá češtinu
s anglickými slovy nativně (code-switching) i čísla — proto se pro tenhle
backend NEPOUŽÍVÁ tts_cs.normalize (žádné num2words/fonetické hacky).

Lokální Chatterbox zůstává default (zdarma, offline). OpenAI je opt-in
(přepínatelný v UI) pro kvalitu. Klíč z OPENAI_API_KEY.

Audio: žádáme WAV (24 kHz), zapíšeme přímo na out_path — zbytek pipeline
(servírování /api/turn/{tid}/audio/{seq}.wav) funguje beze změny.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import httpx

# Validní OpenAI TTS hlasy (2026). Default nova (ženský, přívětivý).
OPENAI_VOICES = (
    "alloy", "ash", "ballad", "coral", "echo", "fable",
    "nova", "onyx", "sage", "shimmer", "verse",
)
DEFAULT_VOICE = "nova"
DEFAULT_MODEL = "gpt-4o-mini-tts"
_ENDPOINT = "https://api.openai.com/v1/audio/speech"
_MAX_INPUT_CHARS = 4096  # OpenAI limit na jeden request

_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class OpenAITTSError(RuntimeError):
    pass


def get_api_key() -> str:
    """API klíč pro gemma OpenAI TTS. Priorita:
      1. ~/.gemma-openai-key (gemma-specific, 0600) — záměrně PŘED env, ať lze
         použít dedikovaný test klíč i když je v env globální OPENAI_API_KEY.
      2. env OPENAI_API_KEY
      3. ~/.openai-api (legacy fallback)
    Override cesty přes OPENAI_API_KEY_FILE. World/group readable soubor → ignore.
    """
    override = os.environ.get("OPENAI_API_KEY_FILE")
    file_candidates = [Path(override)] if override else [
        Path.home() / ".gemma-openai-key",
        Path.home() / ".openai-api",
    ]
    # Gemma-specific soubor má přednost (dedikovaný klíč pro tenhle projekt).
    primary = file_candidates[0]
    if not override and primary.name == ".gemma-openai-key":
        k = _read_key_file(primary)
        if k:
            return k
    env = os.environ.get("OPENAI_API_KEY", "").strip()
    if env:
        return env
    for p in file_candidates:
        k = _read_key_file(p)
        if k:
            return k
    return ""


def _read_key_file(p: Path) -> str:
    """Přečte klíč ze souboru, jen pokud existuje a NENÍ world/group readable."""
    if not p.is_file():
        return ""
    try:
        st = p.stat()
    except OSError:
        return ""
    if st.st_mode & 0o077:
        return ""  # world/group readable → ignore (hardening)
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _clean_text(text: str) -> str:
    """Minimální očista — OpenAI si poradí s čísly i angličtinou, takže jen
    strip control sekvencí + markdown symbolů. ŽÁDNÝ num2words/fonetika."""
    text = _ANSI.sub("", text)
    text = _THINK.sub("", text)
    text = re.sub(r"[*_`#~]+", " ", text)        # markdown markery
    text = re.sub(r"\s+", " ", text).strip()
    return text[:_MAX_INPUT_CHARS]


def is_available() -> bool:
    return bool(get_api_key())


def synth_openai_blocking(
    text: str,
    out_path: Path,
    *,
    voice: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    lang: str = "cs",
    instructions: str | None = None,
    timeout_sec: float = 30.0,
    cancel_event=None,
) -> Path | None:
    """Synth jeden text → WAV na out_path. Blocking (volá se z executoru jako
    Chatterbox cesta). Vrátí out_path, None při prázdném textu, raise při chybě.

    cancel_event (threading.Event | None): pre-flight check. Jednorázový HTTP
    call nelze přerušit uprostřed, ale to je OK — krátké requesty.
    """
    if cancel_event is not None and cancel_event.is_set():
        return None
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    key = get_api_key()
    if not key:
        raise OpenAITTSError(
            "OPENAI_API_KEY není nastavený (export OPENAI_API_KEY=... nebo ~/.openai-api)"
        )
    if voice not in OPENAI_VOICES:
        voice = DEFAULT_VOICE

    payload = {
        "model": model,
        "voice": voice,
        "input": cleaned,
        "response_format": "wav",
    }
    # gpt-4o-mini-tts umí instructions (styl/jazyk). Pomáhá držet češtinu.
    if instructions and model == "gpt-4o-mini-tts":
        payload["instructions"] = instructions
    elif model == "gpt-4o-mini-tts":
        payload["instructions"] = (
            "Mluv plynně a přirozeně. Anglická slova vyslov anglicky."
            if lang == "cs" else "Speak naturally and fluently."
        )

    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(
                _ENDPOINT,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json=payload,
            )
    except httpx.TimeoutException as e:
        raise OpenAITTSError(f"OpenAI TTS timeout po {timeout_sec}s") from e
    except httpx.HTTPError as e:
        raise OpenAITTSError(f"OpenAI TTS HTTP chyba: {type(e).__name__}: {e}") from e

    if resp.status_code != 200:
        # Chybové tělo je JSON {"error": {"message": ...}}
        msg = f"HTTP {resp.status_code}"
        try:
            err = resp.json().get("error", {})
            msg = err.get("message") or msg
        except Exception:
            pass
        raise OpenAITTSError(f"OpenAI TTS: {msg}")

    out_path.write_bytes(resp.content)
    return out_path


def main():
    import sys
    import time
    if len(sys.argv) < 3:
        print("usage: openai_tts.py <text> <out.wav> [voice] [model]", file=sys.stderr)
        sys.exit(1)
    text, out = sys.argv[1], sys.argv[2]
    voice = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_VOICE
    model = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_MODEL
    t0 = time.monotonic()
    synth_openai_blocking(text, Path(out), voice=voice, model=model)
    dt = (time.monotonic() - t0) * 1000
    import soundfile as sf
    d, sr = sf.read(out)
    print(f"saved {out} ({len(d)/sr:.2f}s audio @ {sr}Hz) in {dt:.0f} ms (voice={voice}, model={model})")


if __name__ == "__main__":
    main()
