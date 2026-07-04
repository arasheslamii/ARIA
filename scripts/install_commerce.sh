#!/usr/bin/env bash
# Install Aria's agentic-ordering browser engine into the bundled venv.
# Run with:  sudo bash scripts/install_commerce.sh
# One-time setup. Does NOT touch your system Python — everything lands in
# /opt/aria/venv, the exact environment `aria daemon` runs from.
set -euo pipefail

VENV=/opt/aria/venv
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "!! $PY not found — is the .deb installed?" >&2
  exit 1
fi

echo "==> [1/5] Bootstrapping pip into the bundled venv (it ships without one)"
"$PY" -m ensurepip --upgrade
"$PY" -m pip install --upgrade pip

echo "==> [2/5] Installing browser-use + playwright + langchain-openai"
"$PY" -m pip install 'browser-use>=0.1' 'playwright>=1.40' 'langchain-openai>=0.1'

echo "==> [3/5] Downloading the Chromium browser Playwright drives"
"$VENV/bin/playwright" install chromium

echo "==> [4/5] Installing the system libraries Chromium needs"
"$VENV/bin/playwright" install-deps

echo "==> [5/5] Verifying the imports resolve inside the venv"
"$PY" - <<'PYEOF'
import browser_use, playwright, langchain_openai
print("browser_use   OK")
print("playwright    OK")
print("langchain_openai OK")
PYEOF

echo
echo "==> Done. Now (as your normal user, NOT root) run:"
echo "      systemctl --user restart aria"
echo "    then ask Aria to order the pizza."
