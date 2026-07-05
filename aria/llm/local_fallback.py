"""Last-resort LOCAL fallback: when the cloud brain is rate-limited or offline,
the turn transparently runs on the user's own Ollama models instead.

A local 3-8B model is less brilliant than the cloud 70B — but it is free,
unlimited, private, and always up, and a slightly dumber answer beats "I've hit
my usage limit, try later". This wraps ANY provider chain (Groq alone, or Groq +
a configured free fallback) as the outermost layer.

Detection is lazy and cached: nothing is probed until the cloud actually fails,
so users without Ollama pay zero cost, and an `ollama serve` started mid-session
is still picked up. Model choice reuses the adaptive ranking in
:mod:`aria.llm.ollama` (biggest tool-capable model for reasoning, smallest snappy
one for routing).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable

from aria.llm.base import (
    ChatResult,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    Message,
    ToolSpec,
)
from aria.llm.ollama import detect_ollama, pick_models, rank_models
from aria.llm.openai_compat import OpenAICompatProvider

log = logging.getLogger("aria.llm")

_TRANSIENT = (LLMRateLimitError, LLMConnectionError)
_LOCAL_BASE = "http://localhost:11434/v1"
# A cold Ollama model load can take a minute-plus on CPU; don't mistake it for
# a dead server.
_LOCAL_TIMEOUT_S = 180.0


class LocalFallbackProvider(LLMProvider):
    def __init__(
        self,
        inner: LLMProvider,
        *,
        fast_model: str,
        base: str = _LOCAL_BASE,
        local: LLMProvider | None = None,
    ) -> None:
        self._inner = inner
        # The chain's per-call model names are the CLOUD's; anything that isn't
        # the configured fast model maps to the local reasoning pick.
        self._fast = fast_model
        self._base = base
        self._local = local  # injectable for tests; usually built on first use
        self._picks: dict | None = None
        # Set by the session: called ONCE per switch so Aria can TELL the user
        # she's on the slower local brain — silent multi-minute turns read as
        # "broken", not "degraded".
        self.on_switch: Callable[[], None] | None = None

    async def _local_ready(self) -> bool:
        if self._picks and self._picks.get("reasoning_model"):
            return True
        models = await asyncio.to_thread(detect_ollama, self._base)
        picks = pick_models(rank_models(models))
        if not picks.get("reasoning_model"):
            return False
        self._picks = picks
        if self._local is None:
            self._local = OpenAICompatProvider("ollama", self._base, timeout=_LOCAL_TIMEOUT_S)
        log.info(
            "Cloud LLM unavailable — switching to local Ollama (%s, router %s) "
            "until it recovers.",
            picks["reasoning_model"], picks["fast_model"],
        )
        if self.on_switch is not None:
            try:
                self.on_switch()
            except Exception:  # noqa: BLE001 - the announcement is cosmetic
                pass
        return True

    def _map(self, model: str) -> str:
        """EVERY fallback call runs on the small fast pick. The fallback is a
        stopgap, and on a CPU the big local model turns each turn into minutes
        of prompt-crunching — a quick decent answer beats a slow better one.
        (Users who WANT the big local model pick provider=ollama as their main.)
        """
        assert self._picks is not None
        del model
        return self._picks["fast_model"]

    def _forget_local(self) -> None:
        """A local call failed (Ollama stopped?): re-detect next time."""
        self._picks = None

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
            return await self._inner.chat(
                messages, model=model, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )
        except _TRANSIENT as cloud_err:
            if not await self._local_ready():
                raise
            try:
                return await self._local.chat(  # type: ignore[union-attr]
                    messages, model=self._map(model), tools=tools,
                    temperature=temperature, max_tokens=max_tokens,
                )
            except _TRANSIENT:
                # Report the CLOUD's failure (the accurate story for the user —
                # "rate limited" beats a misleading "offline" from a dead Ollama).
                self._forget_local()
                raise cloud_err from None

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
            async for delta in self._inner.stream(
                messages, model=model, temperature=temperature, max_tokens=max_tokens
            ):
                started = True
                yield delta
            return
        except _TRANSIENT as cloud_err:
            if started:
                raise  # mid-sentence: partial cloud output can't be re-streamed
            if not await self._local_ready():
                raise
            saved_err = cloud_err
        local_started = False
        try:
            async for delta in self._local.stream(  # type: ignore[union-attr]
                messages, model=self._map(model),
                temperature=temperature, max_tokens=max_tokens,
            ):
                local_started = True
                yield delta
        except _TRANSIENT:
            self._forget_local()
            if local_started:
                raise
            raise saved_err from None

    async def aclose(self) -> None:  # pragma: no cover - shutdown best-effort
        for p in (self._inner, self._local):
            if p is None:
                continue
            try:
                await p.aclose()
            except Exception:  # noqa: BLE001
                pass
