"""Generic OpenAI-compatible chat provider (httpx, no extra SDK).

Works with any OpenAI-style /chat/completions endpoint — Cerebras, Google Gemini's
OpenAI-compat endpoint, OpenAI itself, local servers. Used as the FREE fallback
provider when Groq is rate-limited, behind the same LLMProvider interface.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from aria.llm.base import (
    ChatResult,
    LLMAuthError,
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    Message,
    ToolCall,
    ToolSpec,
)


# Some local chat templates (e.g. certain Ollama llama3.x builds) leak the role
# header as the first streamed tokens: "assistant\n\nHi!". Spoken aloud, that's
# jarring — strip it once at the start of a stream.
_ROLE_ECHO = re.compile(r"^\s*assistant\s*[:\n]\s*", re.IGNORECASE)
# Once this much text has arrived without matching, it's real content.
_ROLE_ECHO_MAX = 24


async def strip_role_echo(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    """Pass deltas through, holding back only the first few characters until a
    leaked role header can be ruled out (then everything flows untouched)."""
    head = ""
    deciding = True
    async for delta in deltas:
        if not deciding:
            yield delta
            continue
        head += delta
        if _ROLE_ECHO.match(head):
            head = _ROLE_ECHO.sub("", head, count=1)
            deciding = False
        else:
            probe = head.lstrip().lower()
            still_possible = "assistant".startswith(probe[: len("assistant")]) or (
                probe.startswith("assistant") and len(probe) <= len("assistant") + 2
            )
            if not still_possible or len(head) >= _ROLE_ECHO_MAX:
                deciding = False
        if not deciding and head:
            yield head
            head = ""
    if head:  # stream ended while still deciding
        tail = _ROLE_ECHO.sub("", head, count=1)
        if tail:
            yield tail


class OpenAICompatProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: str, *, timeout: float = 30.0) -> None:
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    @staticmethod
    def _status_error(status: int, text: str) -> Exception:
        if status == 429:
            return LLMRateLimitError(text)
        if status in (401, 403):
            return LLMAuthError(text)
        return RuntimeError(f"HTTP {status}: {text[:200]}")

    def _body(self, messages, model, temperature, max_tokens, *, tools=None, stream=False):
        body: dict[str, Any] = {"model": model, "messages": [m.to_api() for m in messages]}
        if tools:
            body["tools"] = [t.to_api() for t in tools]
            body["tool_choice"] = "auto"
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if stream:
            body["stream"] = True
        return body

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        body = self._body(messages, model, temperature, max_tokens, tools=tools)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base}/chat/completions", headers=self._headers(), json=body
                )
        except httpx.HTTPError as exc:
            raise LLMConnectionError(str(exc)) from exc
        if resp.status_code >= 400:
            raise self._status_error(resp.status_code, resp.text)
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {}
            calls.append(
                ToolCall(id=c.get("id", ""), name=fn.get("name", ""),
                         arguments=args if isinstance(args, dict) else {})
            )
        return ChatResult(
            content=_ROLE_ECHO.sub("", msg.get("content") or "", count=1),
            tool_calls=calls,
            model=data.get("model", model),
            finish_reason=choice.get("finish_reason") or "",
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async for delta in strip_role_echo(self._raw_stream(messages, model,
                                                            temperature, max_tokens)):
            yield delta

    async def _raw_stream(
        self,
        messages: list[Message],
        model: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[str]:
        body = self._body(messages, model, temperature, max_tokens, stream=True)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST", f"{self._base}/chat/completions", headers=self._headers(), json=body
                ) as resp:
                    if resp.status_code >= 400:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        raise self._status_error(resp.status_code, text)
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
        except httpx.HTTPError as exc:
            raise LLMConnectionError(str(exc)) from exc
