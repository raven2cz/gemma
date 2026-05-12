#!/usr/bin/env bash
# Smoke test — prověří že všechny 3 Gemma 4 modely odpovídají česky.
# Použití: bash scripts/smoke_test.sh
set -e

PROMPT="Pozdrav v jedné krátké větě česky a napiš, jaký jsi model."

for tag in gemma4-e4b-32k gemma4-31b-8k gemma4-31b-gguf; do
  if ollama show "$tag" >/dev/null 2>&1; then
    echo "=== $tag ==="
    time ollama run "$tag" "$PROMPT"
    echo
  else
    echo "=== $tag SKIPPED (tag neexistuje) ==="
  fi
done

echo "=== whisper CUDA test ==="
WHISPER=/home/box/git/github/gemma/whisper.cpp
SAMPLE=$WHISPER/samples/jfk.wav
if [ -f "$SAMPLE" ]; then
  "$WHISPER/build/bin/whisper-cli" \
    -m "$WHISPER/models/ggml-large-v3.bin" \
    -l en -f "$SAMPLE" 2>&1 | tail -8
else
  echo "sample audio chybí, přeskakuji"
fi
