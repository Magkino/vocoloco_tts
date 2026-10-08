#!/bin/sh
# Mirror the model files into ./models (git-ignored) for local development, so
# reloading the dev server doesn't re-download ~3 GB from Hugging Face.
# Only needs curl. Resumable: re-run to continue an interrupted download.
#   docker run --rm -v "$PWD":/src -w /src curlimages/curl sh scripts/fetch-models.sh
#   (or just: sh scripts/fetch-models.sh)
set -e
BASE=https://huggingface.co/Gigsu/vocoloco-onnx/resolve/main
DIR="$(dirname "$0")/../models"
mkdir -p "$DIR"
cd "$DIR"

# Files land as .part and are renamed when complete.
fetch() {
  if [ -f "$1" ]; then echo "have $1"; return 0; fi
  curl -fsSL --retry 30 --retry-delay 5 --retry-all-errors -C - -o "$1.part" "$BASE/$1"
  mv "$1.part" "$1"
  echo "done $1"
}

# Hugging Face caps each connection at ~2.5 MB/s, so fetch all files in parallel.
pids=""
for f in omnivoice-main.onnx_data_00 omnivoice-main.onnx_data_01 omnivoice-main.onnx_data_02 \
         omnivoice-main.onnx_data_03 omnivoice-main.onnx_data_04 omnivoice-main-split.onnx \
         omnivoice-decoder.onnx omnivoice-encoder-fixed.onnx omnivoice-config.json; do
  fetch "$f" &
  pids="$pids $!"
done
failed=0
for p in $pids; do wait "$p" || failed=1; done
if [ "$failed" -ne 0 ]; then echo "Some downloads failed; re-run to resume." >&2; exit 1; fi

# The manifest goes last: the app only switches to the mirror once it exists.
fetch omnivoice-main-manifest.json
echo "Done. Restart the dev server: docker compose up --build --watch web"
