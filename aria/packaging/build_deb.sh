#!/usr/bin/env bash
# Build a self-contained .deb for Aria.
#
# Strategy: bundle a venv under /opt/aria/venv with the full voice stack, the
# Piper voice model (/opt/aria/models), and the openWakeWord models (inside the
# venv). /usr/bin/aria runs the bundled venv's Python directly, so Aria never
# relies on the user's system Python or PATH. postinst downloads NOTHING heavy.
#
# Usage:  ./build_deb.sh [version]   (version defaults to aria.__version__)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

VERSION="${1:-$(grep -oP '__version__\s*=\s*"\K[^"]+' "$REPO/aria/__init__.py")}"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo amd64)"
VOICE="en_US-amy-medium"
WAKEWORD="hey_jarvis"

# Stage on disk (not tmpfs) — the bundled venv is hundreds of MB.
STAGE="$(mktemp -d -p "${ARIA_BUILD_TMP:-/var/tmp}")"
PKG="$STAGE/aria_${VERSION}_${ARCH}"
trap 'rm -rf "$STAGE"' EXIT

# Pick a system Python to back the venv (the venv's bin/python symlinks to it).
PYBIN="$(command -v python3.11 || command -v python3.12 || command -v python3)"
echo "==> Aria ${VERSION} (${ARCH}); base python: ${PYBIN}"

echo "==> Staging in $PKG"
mkdir -p "$PKG/opt/aria/models" "$PKG/usr/bin" "$PKG/DEBIAN" \
         "$PKG/usr/lib/systemd/user" "$PKG/usr/share/applications" \
         "$PKG/usr/share/icons/hicolor/scalable/apps"

echo "==> Building bundled venv with the voice stack"
uv venv "$PKG/opt/aria/venv" --python "$PYBIN"
VENV_PY="$PKG/opt/aria/venv/bin/python"
uv pip install --python "$VENV_PY" "$REPO"

echo "==> Bundling Piper voice ($VOICE)"
if [ ! -f "$HERE/models/$VOICE.onnx" ]; then
  # Reuse a copy the wizard already downloaded, else fetch it.
  USER_MODEL="${XDG_DATA_HOME:-$HOME/.local/share}/aria/models/$VOICE.onnx"
  if [ -f "$USER_MODEL" ]; then
    mkdir -p "$HERE/models"
    cp "$USER_MODEL" "$USER_MODEL.json" "$HERE/models/"
  else
    "$HERE/fetch_models.sh" "$VOICE"
  fi
fi
cp "$HERE/models/$VOICE.onnx" "$HERE/models/$VOICE.onnx.json" "$PKG/opt/aria/models/"

echo "==> Bundling openWakeWord models into the venv"
"$VENV_PY" - "$WAKEWORD" <<'PY'
import sys
from openwakeword.utils import download_models
download_models([sys.argv[1]])
print("  wakeword + feature models cached in the bundled venv")
PY

echo "==> Launcher + systemd user unit + desktop entry/icon"
install -m 755 "$HERE/aria-launcher.sh" "$PKG/usr/bin/aria"
install -m 644 "$HERE/aria.service" "$PKG/usr/lib/systemd/user/aria.service"
install -m 644 "$HERE/aria.desktop" "$PKG/usr/share/applications/aria.desktop"
install -m 644 "$HERE/aria.svg" "$PKG/usr/share/icons/hicolor/scalable/apps/aria.svg"

echo "==> Debian control + maintainer scripts"
INSTALLED_KB="$(du -sk "$PKG/opt" | cut -f1)"
cat > "$PKG/DEBIAN/control" <<EOF
Package: aria
Version: ${VERSION}
Section: utils
Priority: optional
Architecture: ${ARCH}
Depends: libportaudio2, libsecret-1-0, python3 (>= 3.11)
Recommends: brightnessctl, playerctl, wl-clipboard, grim
Installed-Size: ${INSTALLED_KB}
Maintainer: Aria contributors <aria@localhost>
Description: Aria - fast, agentic, voice-first AI assistant for Linux.
 A terminal-native voice assistant. Talk to it; it talks back. No GUI.
 Bundles a self-contained Python runtime, a local Piper voice, and an
 on-device wake word. Run 'aria setup', then 'aria', then 'aria enable'.
EOF

for script in postinst prerm postrm; do
  install -m 755 "$HERE/$script" "$PKG/DEBIAN/$script"
done

echo "==> dpkg-deb build"
OUT="$REPO/aria_${VERSION}_${ARCH}.deb"
dpkg-deb --build --root-owner-group "$PKG" "$OUT"
echo "==> Built $OUT ($(du -h "$OUT" | cut -f1))"
