#!/usr/bin/env bash
# Install the local (offline) brain: Ollama + a model sized to this machine.
# Usage: install_local.sh [model-tag]   (default: qwen2.5:3b)
# Idempotent: skips anything already present. Run by `aria install-local`.
set -euo pipefail

MODEL="${1:-qwen2.5:3b}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "==> Installing Ollama (the official installer; may ask for your password)"
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "==> Ollama is already installed."
fi

# Make sure the server is up (the installer usually starts a systemd service;
# cover the case where it didn't).
if ! curl -fsS --max-time 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "==> Starting the Ollama server"
    (ollama serve >/dev/null 2>&1 &)
    for _ in $(seq 1 15); do
        sleep 1
        curl -fsS --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
    done
fi

echo "==> Downloading ${MODEL} (one-time; a few GB — grab a coffee)"
ollama pull "${MODEL}"

echo "==> Done. ${MODEL} is ready."
echo "    Aria will use it automatically whenever the cloud is rate-limited or offline."
echo "    To make it the MAIN brain instead, run: aria setup  ->  Change AI provider -> Local"
