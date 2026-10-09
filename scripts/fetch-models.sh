#!/bin/sh
# Mirror the model files into ./models (git-ignored) for local development, so
# reloading the dev server doesn't re-download ~3.5 GB from Hugging Face.
# Only needs curl. Resumable: re-run to continue an interrupted download.
#   docker run --rm -v "$PWD":/src -w /src curlimages/curl sh scripts/fetch-models.sh
#   (or just: sh scripts/fetch-models.sh)
set -e
HF=https://huggingface.co
DIR="$(dirname "$0")/../models"
mkdir -p "$DIR"
cd "$DIR"

# fetch <url> <file>: Hugging Face caps each connection at ~2.5 MB/s, so a
# file is fetched as up to 8 byte ranges in parallel. Ranges land as
# <file>.partN; the file appears once all of them are complete.
fetch() {
  url=$1; out=$2
  if [ -f "$out" ]; then echo "have $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  size=$(curl -fsIL --retry 10 --retry-all-errors "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }')
  n=8
  [ "$size" -lt 67108864 ] && n=1
  chunk=$(( (size + n - 1) / n ))
  pids=""; parts=""; i=0
  while [ $i -lt $n ]; do
    a=$((i * chunk)); b=$((a + chunk - 1))
    [ $b -ge "$size" ] && b=$((size - 1))
    part="$out.part$i"; parts="$parts $part"
    if [ ! -f "$part" ] || [ "$(wc -c < "$part")" -ne $((b - a + 1)) ]; then
      curl -fsSL --retry 30 --retry-delay 5 --retry-all-errors -r "$a-$b" -o "$part" "$url" &
      pids="$pids $!"
    fi
    i=$((i + 1))
  done
  for p in $pids; do wait "$p" || return 1; done
  # shellcheck disable=SC2086
  cat $parts > "$out.tmp" && mv "$out.tmp" "$out" && rm -f $parts
  echo "done $out"
}

run_all() { # fetch "url file" pairs in parallel
  pids=""
  while read -r url out; do
    [ -n "$url" ] || continue
    fetch "$url" "$out" &
    pids="$pids $!"
  done
  failed=0
  for p in $pids; do wait "$p" || failed=1; done
  if [ "$failed" -ne 0 ]; then echo "Some downloads failed; re-run to resume." >&2; exit 1; fi
}

# TTS models
TTS=$HF/Gigsu/vocoloco-onnx/resolve/main
run_all <<EOF
$TTS/omnivoice-main.onnx_data_00 omnivoice-main.onnx_data_00
$TTS/omnivoice-main.onnx_data_01 omnivoice-main.onnx_data_01
$TTS/omnivoice-main.onnx_data_02 omnivoice-main.onnx_data_02
$TTS/omnivoice-main.onnx_data_03 omnivoice-main.onnx_data_03
$TTS/omnivoice-main.onnx_data_04 omnivoice-main.onnx_data_04
$TTS/omnivoice-main-split.onnx omnivoice-main-split.onnx
$TTS/omnivoice-decoder.onnx omnivoice-decoder.onnx
$TTS/omnivoice-encoder-fixed.onnx omnivoice-encoder-fixed.onnx
$TTS/omnivoice-config.json omnivoice-config.json
EOF
# The manifest goes last: the app only switches to the mirror once it exists.
fetch "$TTS/omnivoice-main-manifest.json" omnivoice-main-manifest.json

# Speech recognition for the voice wizard (Whisper large-v3-turbo, the files
# the app loads on WebGPU). config.json goes last, like the manifest above.
W=onnx-community/whisper-large-v3-turbo
WR=$HF/$W/resolve/main
run_all <<EOF
$WR/onnx/encoder_model_q4f16.onnx $W/onnx/encoder_model_q4f16.onnx
$WR/onnx/decoder_model_merged_q4f16.onnx $W/onnx/decoder_model_merged_q4f16.onnx
$WR/generation_config.json $W/generation_config.json
$WR/preprocessor_config.json $W/preprocessor_config.json
$WR/tokenizer.json $W/tokenizer.json
$WR/tokenizer_config.json $W/tokenizer_config.json
$WR/special_tokens_map.json $W/special_tokens_map.json
$WR/added_tokens.json $W/added_tokens.json
$WR/normalizer.json $W/normalizer.json
$WR/vocab.json $W/vocab.json
$WR/merges.txt $W/merges.txt
EOF
fetch "$WR/config.json" "$W/config.json"

echo "Done. Restart the dev server: docker compose up --build --watch web"
