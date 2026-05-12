#!/usr/bin/env bash
# End-to-end hlasový chat: mic → whisper CS → LLM → chatterbox CS → reproduktor
# Použití: bash scripts/chat_voice.sh [sekund_nahrávání=5] [model=gemma4-e4b-32k] [fast]
#   fast = TTS bez CFG (cfg_weight=0) přes monkey-patch → ~2× rychlejší, menší expresivita
set -e

SEC="${1:-5}"
MODEL="${2:-gemma4-e4b-32k}"
FAST="${3:-}"
TTS_EXTRA=""
[ "$FAST" = "fast" ] && TTS_EXTRA="--fast"

ROOT="/home/box/git/github/gemma"
WHISPER="$ROOT/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL="$ROOT/whisper.cpp/models/ggml-large-v3.bin"
VAD_MODEL="$ROOT/whisper.cpp/models/ggml-silero-v6.2.0.bin"
REF="$ROOT/voice/ref_female.wav"
TMP=$(mktemp -d)
IN="$TMP/in.wav"
OUT="$TMP/out.wav"

trap "rm -rf $TMP" EXIT

echo ">>> Uvolňuji LLM z VRAM (kvůli whisperu)..."
for m in gemma4-e4b-32k gemma4-31b-8k gemma4-31b-gguf; do
  ollama stop "$m" 2>/dev/null || true
done

echo ">>> Mluv česky ($SEC s)..."
# arecord 16kHz mono (whisper.cpp chce 16kHz)
arecord -q -f S16_LE -r 16000 -c 1 -d "$SEC" "$IN"

echo ">>> Whisper (CS)..."
TEXT=$("$WHISPER" -m "$WHISPER_MODEL" -vm "$VAD_MODEL" --vad -l cs -f "$IN" -nt 2>/dev/null | tr -d '\n' | sed 's/^ *//;s/ *$//')
echo "USER: $TEXT"

if [ -z "$TEXT" ]; then
  echo "(nic nerozpoznáno)"
  exit 1
fi

echo ">>> Gemma ($MODEL)..."
if ! curl -sf http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama neběží na :11434. Spusť: sudo systemctl start ollama" >&2
  exit 1
fi
SYS="Odpovídej česky, stručně, v jednom odstavci bez markdown formátování, emoji a bez odrážek."
RAW=$(curl -sS http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d "$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','think':False,'stream':False,'messages':[{'role':'system','content':'$SYS'},{'role':'user','content':sys.argv[1]}]}))" "$TEXT")")
REPLY=$(printf '%s' "$RAW" | python3 -c "import json,sys
try:
    d=json.loads(sys.stdin.read())
except Exception as e:
    print('', end=''); sys.exit(0)
if 'error' in d:
    import sys as s; s.stderr.write('Ollama error: '+str(d['error'])+'\n'); sys.exit(2)
print((d.get('message') or {}).get('content','').strip())") || { echo "Chyba LLM: $RAW" >&2; exit 1; }
if [ -z "$REPLY" ]; then
  echo "Prázdná odpověď z LLM. Raw: $RAW" >&2
  exit 1
fi
echo "ASSISTANT: $REPLY"

# Omez na ~800 znaků (TTS to chunkuje po větách, ale moc dlouhé = moc dlouhé přehrávání)
SPEAK=$(printf '%s' "$REPLY" | cut -c1-800)

echo ">>> Uvolňuji model z VRAM..."
ollama stop "$MODEL" 2>/dev/null || true

echo ">>> Chatterbox TTS..."
cd "$ROOT/voice"
./.venv-tts/bin/python tts_cs.py "$SPEAK" "$OUT" cuda "$REF" $TTS_EXTRA 2>&1 | tail -3

echo ">>> Přehrávám..."
aplay -q "$OUT"
