# Tests

Spuštění:

```bash
./voice/.venv-tts/bin/pytest tests/ -v
```

## Struktura

- `test_sentence_chunker.py` — streaming sentence splitter pro TTS (Phase B).
- `test_detect_lang.py` — CZ/EN heuristika z `voice/tts_cs.py`.
- `test_normalize.py` — normalizace zkratek a čísel pro TTS.
- `integration/` — manuální curl skripty proti běžícímu serveru.

Testy nemají být závislé na GPU / Chatterbox / Ollama. Pro moduly co
importují heavy deps (`voice.tts_cs` loading Chatterbox při import
time) použij fixtures s monkey-patch.
