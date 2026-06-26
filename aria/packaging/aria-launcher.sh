#!/usr/bin/env bash
# /usr/bin/aria — runs the bundled, self-contained venv's Python directly, so
# Aria never depends on the user's system Python or PATH. Points the model
# resolver at the bundled Piper voice under /opt/aria/models.
export ARIA_MODELS_DIR="${ARIA_MODELS_DIR:-/opt/aria/models}"
exec /opt/aria/venv/bin/python -m aria "$@"
