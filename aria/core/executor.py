"""Resilient tool execution.

Wraps every tool call with: permission classification + confirmation gate,
timeout, bounded retry with backoff, audit logging, and a graceful spoken
fallback. A single tool failure must never crash the conversation loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aria.safety.audit import AuditLog
from aria.safety.permissions import classify, needs_confirmation
from aria.tools.base import Tool, ToolError, ToolResult

# A confirm callback returns True if the user approved the action.
ConfirmFn = Callable[[str], Awaitable[bool]]


@dataclass
class ExecConfig:
    timeout_s: float = 15.0
    retries: int = 2
    require_confirmation: bool = True


class ToolExecutor:
    def __init__(self, audit: AuditLog, config: ExecConfig) -> None:
        self._audit = audit
        self._config = config

    async def execute(
        self,
        tool: Tool,
        arguments: dict | None,
        *,
        confirm: ConfirmFn | None = None,
    ) -> ToolResult:
        # Defensive: no-arg calls may arrive as None; classify + run need a dict.
        arguments = arguments or {}
        decision = classify(tool, arguments)

        if decision.risk == "blocked":
            self._log(tool, arguments, decision.risk, "blocked", confirmed=False)
            return ToolResult(content="blocked by policy", spoken="I can't do that one.")

        if needs_confirmation(decision, self._config.require_confirmation):
            approved = await confirm(tool.name) if confirm else False
            if not approved:
                self._log(tool, arguments, decision.risk, "declined", confirmed=False)
                return ToolResult(content="user declined", spoken="Okay, I won't do that.")

        result = await self._run_with_resilience(tool, arguments)
        outcome = "ok" if not result.content.startswith("error") else "error"
        self._log(tool, arguments, decision.risk, outcome, confirmed=decision.risk != "safe")
        return result

    async def _run_with_resilience(self, tool: Tool, arguments: dict) -> ToolResult:
        last_err: Exception | None = None
        for attempt in range(self._config.retries + 1):
            try:
                return await asyncio.wait_for(
                    tool.run(**arguments), timeout=self._config.timeout_s
                )
            except ToolError as exc:
                # Expected failure — don't retry, surface it.
                return ToolResult(content=f"error: {exc}", spoken=f"That didn't work: {exc}")
            except TimeoutError as exc:
                last_err = exc
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            if attempt < self._config.retries:
                await asyncio.sleep(0.4 * (attempt + 1))  # linear backoff
        return ToolResult(
            content=f"error: {last_err}",
            spoken="I couldn't complete that — want me to try a different way?",
        )

    def _log(self, tool: Tool, args: dict, risk: str, outcome: str, *, confirmed: bool) -> None:
        try:
            self._audit.record(
                tool=tool.name, arguments=args, risk=risk, outcome=outcome, confirmed=confirmed
            )
        except Exception:  # auditing must never break execution
            pass
