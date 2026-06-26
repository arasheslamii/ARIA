"""Groq implementation of :class:`LLMProvider`.

Groq exposes an OpenAI-compatible chat API, so this class is also a near-drop-in
template for the future OpenAI-compatible/Ollama providers (swap base_url +
client). Connections are pre-warmed by the orchestrator to shave first-token
latency.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import AsyncIterator
from typing import Any

from groq import (
    APIConnectionError,
    AsyncGroq,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)

from aria.llm.base import (
    ChatResult,
    LLMAuthError,
    LLMConnectionError,
    LLMProvider,
    Message,
    ToolCall,
    ToolSpec,
)

# Llama models on Groq occasionally emit a malformed tool call that Groq rejects
# with a 400 `tool_use_failed`. We salvage it, then retry, then degrade — see
# GroqProvider.chat. This is how many extra round-trips we allow before degrading.
_MAX_TUF_RETRIES = 2

# Matches the model's malformed wrapper: <function=NAME(... or <function=NAME>...
_FUNC_NAME_RE = re.compile(r"<\s*function\s*=\s*([A-Za-z0-9_.\-]+)")


def _translate(exc: Exception) -> Exception:
    """Map Groq SDK errors to provider-agnostic ones so the runtime can show a
    friendly message instead of leaking a raw traceback."""
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return LLMAuthError(str(exc))
    if isinstance(exc, APIConnectionError):
        return LLMConnectionError(str(exc))
    return exc


def _tool_use_failed_text(exc: Exception) -> str | None:
    """If ``exc`` is a Groq ``tool_use_failed`` 400, return its ``failed_generation``
    string (possibly empty); otherwise None. Tolerates the error body being either
    ``{"error": {...}}`` or the error dict itself."""
    if not isinstance(exc, BadRequestError):
        return None
    body = getattr(exc, "body", None)
    err: dict[str, Any] = {}
    if isinstance(body, dict):
        inner = body.get("error", body)
        err = inner if isinstance(inner, dict) else {}
    code = err.get("code") or getattr(exc, "code", None)
    if code != "tool_use_failed":
        return None
    return err.get("failed_generation") or ""


def _extract_args(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a malformed tool call, tolerating
    truncation (unbalanced/​unterminated) the way router._parse tolerates noise."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    end: int | None = None
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is not None:
        candidate = text[start:end]
    else:  # truncated — repair by closing the string and braces
        candidate = text[start:]
        if in_str:
            candidate += '"'
        candidate += "}" * max(depth, 1)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        # Last-ditch: strip trailing wrappers and progressively close braces.
        base = text[start:].split("</function>")[0].rstrip(") \n\t")
        for extra in range(1, 6):
            try:
                parsed = json.loads(base + "}" * extra)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                continue
    return None


def salvage_tool_call(failed_generation: str) -> ToolCall | None:
    """Reconstruct a ToolCall from Groq's ``failed_generation`` text, e.g.
    '<function=web_search({"query":"x","max_results":5}</function>'."""
    if not failed_generation:
        return None
    m = _FUNC_NAME_RE.search(failed_generation)
    if not m:  # fall back to an OpenAI-style {"name": ...} if present
        m = re.search(r'"name"\s*:\s*"([A-Za-z0-9_.\-]+)"', failed_generation)
    if not m:
        return None
    name = m.group(1)
    args = _extract_args(failed_generation) or {}
    return ToolCall(id=f"salvaged_{uuid.uuid4().hex[:8]}", name=name, arguments=args)


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, *, timeout: float = 30.0) -> None:
        self._client = AsyncGroq(api_key=api_key, timeout=timeout)

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        def build(include_tools: bool) -> dict:
            kwargs: dict = {"model": model, "messages": [m.to_api() for m in messages]}
            if include_tools and tools:
                kwargs["tools"] = [t.to_api() for t in tools]
                kwargs["tool_choice"] = "auto"
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            return kwargs

        include_tools = bool(tools)
        retries = 0
        while True:
            try:
                resp = await self._client.chat.completions.create(**build(include_tools))
                return self._to_result(resp)
            except Exception as exc:  # noqa: BLE001
                failed = _tool_use_failed_text(exc)
                if failed is None:
                    # Not a malformed-tool-call error — surface cleanly.
                    raise _translate(exc) from exc
                # (a) Salvage the intended call from the failed generation.
                salvaged = salvage_tool_call(failed)
                if salvaged is not None:
                    return ChatResult(
                        content="", tool_calls=[salvaged], model=model, finish_reason="tool_calls"
                    )
                # (b) Retry the same request — the next generation is often valid.
                if include_tools and retries < _MAX_TUF_RETRIES:
                    retries += 1
                    continue
                # (c) Degrade: drop tools so the user still gets a text answer.
                if include_tools:
                    include_tools = False
                    continue
                raise _translate(exc) from exc

    @staticmethod
    def _to_result(resp) -> ChatResult:
        choice = resp.choices[0]
        raw_calls = choice.message.tool_calls or []
        return ChatResult(
            content=choice.message.content or "",
            tool_calls=[ToolCall.from_raw(c) for c in raw_calls],
            model=resp.model,
            finish_reason=choice.finish_reason or "",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict = {
            "model": model,
            "messages": [m.to_api() for m in messages],
            "stream": True,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:  # noqa: BLE001 - re-raised as a clean LLMError
            raise _translate(exc) from exc

    async def aclose(self) -> None:
        await self._client.close()
