"""Fallback LLM provider: keep working past the primary's rate limit.

Wraps a primary (Groq) and a fallback (another free provider). On a transient
primary failure (rate limit / connection), it retries on the fallback — so Groq's
daily cap never takes Aria offline. The fallback uses its own model name (the
primary's per-call model names don't apply to it).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from aria.llm.base import (
    ChatResult,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    Message,
    ToolSpec,
)

_TRANSIENT = (LLMRateLimitError, LLMConnectionError)


class FallbackProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, fallback: LLMProvider, fallback_model: str) -> None:
        self._primary = primary
        self._fallback = fallback
        self._fallback_model = fallback_model

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        try:
            return await self._primary.chat(
                messages, model=model, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )
        except _TRANSIENT:
            return await self._fallback.chat(
                messages, model=self._fallback_model, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        started = False
        try:
            async for delta in self._primary.stream(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            ):
                started = True
                yield delta
            return
        except _TRANSIENT:
            if started:
                raise  # already spoke partial output — caller handles the rest
        # Primary failed before any token: re-stream from the fallback.
        async for delta in self._fallback.stream(
            messages, model=self._fallback_model, temperature=temperature, max_tokens=max_tokens
        ):
            yield delta

    async def aclose(self) -> None:  # pragma: no cover
        for p in (self._primary, self._fallback):
            try:
                await p.aclose()
            except Exception:  # noqa: BLE001
                pass
