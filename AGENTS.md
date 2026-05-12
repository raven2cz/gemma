# Projekt: Gemma 4 Local Stack

Lokální AI stack: Ollama (Gemma 4 ×3) + whisper.cpp (STT CS) + Chatterbox-TTS-Czech + opencode. Hardware: RTX 5070 Ti 16 GB, 64 GB RAM, Arch Linux.

## Jazyk

- **Odpovídej česky**, pokud uživatel nepožádá jinak. Pro odborné termíny používej anglické originály (např. „context window", ne „kontextové okno").
- Buď **stručný a přesný**. Žádné úvody typu „Jasně!", žádné shrnutí po každé změně. Vysvětluj jen tehdy, když je to nutné.

## Konvence

- **Shell**: bash na Arch Linuxu. Pro delší skripty preferuj `set -euo pipefail` a kontrolu existence souborů před použitím.
- **Python**: venv je `voice/.venv-tts` (Python 3.11). Nikdy neinstaluj do systémového Pythonu.
- **Cesty**: projekt je `/home/box/git/github/gemma`. Modely Ollamy: `/home/box/git/github/gemma/models/ollama`. GGUF: `gguf/`. Whisper modely: `whisper.cpp/models/`.
- **Modelfiles**: zdroj pravdy je `modelfiles/*.Modelfile`. Po úpravě vždy `ollama create <tag> -f ...`.
- **Systemd**: override pro Ollamu je v `/etc/systemd/system/ollama.service.d/override.conf`. Respektuj `OLLAMA_KEEP_ALIVE=4m`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`.

## Modely

- `gemma4-e4b-32k` — malý, celý ve VRAM, 32K ctx. Pro rychlé věci.
- `gemma4-31b-8k` — Ollama stock Q4_K_M, 32K ctx, auto GPU/CPU split. Kvalitní default.
- `gemma4-31b-gguf` — Unsloth UD-Q4_K_XL, 32K ctx, `num_gpu=30`. Pro laděné tool-calling workflow.

## Workflow

- Před změnou konfiguračních souborů (systemd, Modelfile, opencode.json) zkontroluj aktuální stav.
- Downloady jsou velké (modely 10–20 GB) a síť kolísá — při resumování používej `curl -C -` / `hf download` / `ollama pull` (všechny umí resume).
- 300 Mbit linka reálně jede ~50 Mbit; nestahuj paralelně víc než jeden velký model naráz.

## Ne-dělat

- Neinstaluj nové systémové balíčky přes `sudo pacman` bez dotazu.
- Neupravuj `~/.config/opencode/opencode.json` přímo — edituj `/home/box/git/github/gemma/opencode.json` a kopíruj.
- Nezkoušej `num_gpu 99` na 31B GGUF — přetéká 16GB VRAM (použij 30).
- Nepřidávej k odpovědím prázdné shrnutí „to je vše" / „úspěšně dokončeno".
