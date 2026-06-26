#!/usr/bin/env bash
# Fetch the default Piper voice into packaging/models/ so build_deb.sh can bundle
# it. Run on the *build* machine, not the user's. Keeps the .deb self-contained.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/models"
mkdir -p "$DEST"

VOICE="${1:-en_US-amy-medium}"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium"

echo "==> Fetching Piper voice: $VOICE"
curl -fL "$BASE/${VOICE}.onnx"      -o "$DEST/${VOICE}.onnx"
curl -fL "$BASE/${VOICE}.onnx.json" -o "$DEST/${VOICE}.onnx.json"
echo "==> Saved to $DEST"
