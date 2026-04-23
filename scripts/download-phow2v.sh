#!/usr/bin/env bash
# Download a PhoW2V variant from VinAI's public mirror into ./models/.
# Usage: ./scripts/download-phow2v.sh [word|syllable] [100|300]
# Defaults: word, 300.
set -euo pipefail

VARIANT="${1:-word}"
DIMS="${2:-300}"

case "$VARIANT" in
  word)     SUFFIX="words" ;;
  syllable) SUFFIX="syllables" ;;
  *) echo "variant must be 'word' or 'syllable'" >&2; exit 2 ;;
esac

case "$DIMS" in
  100|300) ;;
  *) echo "dims must be 100 or 300" >&2; exit 2 ;;
esac

URL="https://public.vinai.io/word2vec_vi_${SUFFIX}_${DIMS}dims.zip"
OUT_DIR="models"
ZIP_PATH="${OUT_DIR}/word2vec_vi_${SUFFIX}_${DIMS}dims.zip"
TXT_PATH="${OUT_DIR}/word2vec_vi_${SUFFIX}_${DIMS}dims.txt"

mkdir -p "$OUT_DIR"

if [[ -f "$TXT_PATH" ]]; then
  echo "already present: $TXT_PATH"
  exit 0
fi

echo "downloading $URL"
curl -fL --progress-bar -o "$ZIP_PATH" "$URL"

echo "extracting to $OUT_DIR"
unzip -o -j "$ZIP_PATH" -d "$OUT_DIR"
rm -f "$ZIP_PATH"

# Unzip may produce a differently-named .txt depending on archive contents.
# Rename to the expected path if a single .txt was extracted.
if [[ ! -f "$TXT_PATH" ]]; then
  shopt -s nullglob
  candidates=("$OUT_DIR"/*.txt)
  if [[ ${#candidates[@]} -eq 1 ]]; then
    mv "${candidates[0]}" "$TXT_PATH"
  else
    echo "warning: could not resolve extracted .txt path; check $OUT_DIR" >&2
  fi
fi

echo "ready: $TXT_PATH"
