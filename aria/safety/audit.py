"""Append-only audit trail of every action Aria takes.

Written as JSON lines to the state dir. Never records secrets/PII beyond the
tool name + arguments the user themselves requested; callers should redact
sensitive args before logging if needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from aria.config.loader import state_dir


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (state_dir() / "audit.log")

    def record(
        self,
        *,
        tool: str,
        arguments: dict[str, Any],
        risk: str,
        outcome: str,
        confirmed: bool,
    ) -> None:
        entry = {
            "ts": time.time(),
            "tool": tool,
            "arguments": arguments,
            "risk": risk,
            "confirmed": confirmed,
            "outcome": outcome,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
