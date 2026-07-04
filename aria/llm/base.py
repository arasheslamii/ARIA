"""Provider-agnostic LLM interface.

Anything swappable (Groq, OpenAI-compatible, Anthropic, Ollama) implements
:class:`LLMProvider`. The orchestrator only ever talks to this interface.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


class LLMError(Exception):
    """Base class for provider-agnostic LLM failures."""


class LLMAuthError(LLMError):
    """The API key is invalid, expired, or missing."""


class LLMConnectionError(LLMError):
    """The provider could not be reached (network/DNS/timeout)."""


class LLMRateLimitError(LLMError):
    """The provider rate-limited the request (e.g. Groq's free daily cap)."""


@dataclass
class ToolCall:
    """A model's request to invoke a tool. ``arguments`` is ALWAYS a dict."""

    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        # Invariant: no-arg / malformed calls must never carry None — downstream
        # code does arguments.values() and tool.run(**arguments).
        if not isinstance(self.arguments, dict):
            self.arguments = {}

    @classmethod
    def from_raw(cls, raw: Any) -> ToolCall:
        args = raw.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):  # None, "", or non-object JSON (list/scalar)
            args = {}
        return cls(id=raw.id, name=raw.function.name, arguments=args)


@dataclass
class Message:
    role: Role
    content: str = ""
    # assistant messages may carry tool calls; tool messages carry a call id.
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_api(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        if self.name:
            msg["name"] = self.name
        return msg


# Convenience constructors -------------------------------------------------
def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls or [])


def tool_result(call_id: str, content: str, name: str | None = None) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id, name=name)


@dataclass
class ToolSpec:
    """JSON-schema description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_api(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ChatResult:
    """Non-streaming result: text and/or tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""


class LLMProvider(ABC):
    """The seam every model backend implements."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """One-shot completion, optionally with tool calls."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield text deltas as they arrive (for streaming TTS)."""

    async def aclose(self) -> None:  # pragma: no cover - optional cleanup
        """Release any pooled connections."""
