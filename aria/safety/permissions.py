"""Safety/permissions layer.

Every tool call is classified safe / confirm / blocked. ``confirm`` actions
(send email, spend money, book, delete files, system changes) require an explicit
yes before they run. Classification starts from the tool's declared ``risk`` and
can be tightened (never loosened) by per-call heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aria.tools.base import Risk, Tool

# Tool names that are always blocked regardless of declaration.
_BLOCKED: set[str] = set()

# Argument heuristics that escalate a "safe" tool to "confirm".
_DANGEROUS_HINTS = ("rm -rf", "sudo", "mkfs", "dd if=", ":(){", "shutdown", "reboot")


@dataclass
class PermissionDecision:
    risk: Risk
    reason: str

    @property
    def allowed_without_confirmation(self) -> bool:
        return self.risk == "safe"


def classify(tool: Tool, arguments: dict[str, Any] | None) -> PermissionDecision:
    if tool.name in _BLOCKED:
        return PermissionDecision("blocked", f"{tool.name} is blocked by policy")

    # No-arg tool calls may arrive with arguments=None — never crash on .values().
    arguments = arguments or {}
    flat = " ".join(str(v) for v in arguments.values()).lower()
    if any(hint in flat for hint in _DANGEROUS_HINTS):
        return PermissionDecision("confirm", "arguments look potentially destructive")

    return PermissionDecision(tool.risk, f"declared risk={tool.risk}")


def needs_confirmation(decision: PermissionDecision, require_confirmation: bool) -> bool:
    if decision.risk == "blocked":
        return True  # caller must refuse
    if decision.risk == "confirm":
        return require_confirmation
    return False
