"""Lightweight per-turn debug tracing, gated behind ARIA_DEBUG=1.

Prints route, exposed tool names, and each tool call + result to stderr so we can
see selection/dispatch during a live `aria` / validate_brain run without adding a
logging dependency or noise to normal runs.
"""

from __future__ import annotations

import os
import sys


def debug_enabled() -> bool:
    return os.environ.get("ARIA_DEBUG", "") not in ("", "0", "false", "False")


def dlog(message: str) -> None:
    if debug_enabled():
        print(f"\033[2m[aria:debug] {message}\033[0m", file=sys.stderr, flush=True)


def _truncate(text: str, limit: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"
