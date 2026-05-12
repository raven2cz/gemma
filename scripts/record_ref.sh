#!/usr/bin/env bash
# Nahraje system audio z default sinku a uloží jako chatterbox-ready referenci.
#
# Použití:
#   ./scripts/record_ref.sh <family> [duration=15] [skip=0] [lang=cs]
#
# Příklady:
#   ./scripts/record_ref.sh eva                 # 15s, od začátku, CS
#   ./scripts/record_ref.sh eva 12 30           # 12s, přeskoč prvních 30s (jingle)
#   ./scripts/record_ref.sh mary 15 0 en        # 15s, EN voice family
#
# Výstup: voice/ref_<family>_<lang>.wav  (mono 24kHz PCM16, loudness-normalized,
#                                         trim tichých konců, highpass 80 Hz)
# + voice/raw/<family>.wav (nezpracovaný záznam, pro případ úprav)
set -euo pipefail

FAMILY="${1:?usage: record_ref.sh <family> [duration=15] [skip=0] [lang=cs]}"
DUR="${2:-15}"
SKIP="${3:-0}"
LANG="${4:-cs}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOICE_DIR="$ROOT/voice"
RAW_DIR="$VOICE_DIR/raw"
mkdir -p "$RAW_DIR"

RAW="$RAW_DIR/${FAMILY}.wav"
REF="$VOICE_DIR/ref_${FAMILY}_${LANG}.wav"

SINK="$(pactl get-default-sink)"
MONITOR="${SINK}.monitor"

echo "── nahrávání system audio ──────────────────────────────"
echo "  source   : $MONITOR"
echo "  skip     : ${SKIP}s"
echo "  duration : ${DUR}s"
echo "  raw      : $RAW"
echo
echo "Spusť přehrávání teď. Nahrávání startuje za 2s…"
sleep 2

# Krok 1: raw záznam (stereo 48kHz float — beze ztráty, pro případné pozdější úpravy)
ffmpeg -hide_banner -loglevel error -y \
  -f pulse -i "$MONITOR" \
  -ss "$SKIP" -t "$DUR" \
  -c:a pcm_f32le -ar 48000 -ac 2 \
  "$RAW"

# Kontrola peaku — jestli není prázdné
PEAK=$(ffmpeg -hide_banner -i "$RAW" -af "volumedetect" -f null - 2>&1 \
       | awk '/max_volume:/{print $5+0}')
echo "  raw peak : ${PEAK} dB"
if [[ $(awk "BEGIN{print ($PEAK < -60)}") == "1" ]]; then
  echo "  ⚠ raw je téměř ticho — zkontroluj, že něco hraje na default sinku"
  exit 2
fi

# Krok 2: zpracování na chatterbox-ready ref
#  - mono downmix (-ac 1)
#  - 24kHz (Chatterbox native sr pro ref)
#  - PCM16 LE (subtype co používají stávající ref_*.wav)
#  - highpass 80Hz (odfiltruj rumble/DC)
#  - silenceremove: odstraň ticho na začátku/konci (< -40dB)
#  - loudnorm: normalizuj na -18 LUFS / -2 dBTP (broadcast-safe, nerozjeté)
ffmpeg -hide_banner -loglevel error -y \
  -i "$RAW" \
  -af "highpass=f=80,silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB:stop_periods=-1:stop_silence=0.3:stop_threshold=-40dB,loudnorm=I=-18:TP=-2:LRA=7" \
  -ar 24000 -ac 1 -sample_fmt s16 \
  "$REF"

# Výpis výsledku
DUR_OUT=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$REF")
SIZE=$(stat -c %s "$REF")
echo
echo "── hotovo ──────────────────────────────────────────────"
echo "  ref      : $REF"
echo "  duration : ${DUR_OUT}s"
echo "  size     : $((SIZE/1024)) KB"
echo "  formát   : mono 24kHz PCM16 (chatterbox-ready)"
echo
echo "Poslech:  mpv $REF"
echo "V UI se objeví ve Voice dropdownu jako '${FAMILY}' (po refreshi stránky)."
