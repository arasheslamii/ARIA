"""Shared test fakes: in-memory LLM, STT, TTS, mic/speaker — no network, no audio."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from aria.llm.base import ChatResult, LLMProvider, Message, ToolCall
from aria.voice.audio import Microphone, Speaker
from aria.voice.base import STT, TTS, AudioChunk


class FakeLLM(LLMProvider):
    """Scripted provider. ``chat_queue`` drives tool-call turns; ``stream_text``
    is emitted token-by-token by :meth:`stream`."""

    def __init__(self, *, stream_text: str = "Sure thing.", chat_queue=None) -> None:
        self.stream_text = stream_text
        self.chat_queue = list(chat_queue or [])
        self.calls: list[str] = []

    async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
        self.calls.append(model)
        if self.chat_queue:
            return self.chat_queue.pop(0)
        return ChatResult(content=self.stream_text, model=model, finish_reason="stop")

    async def stream(self, messages, *, model, temperature=None, max_tokens=None):
        for word in self.stream_text.split(" "):
            yield word + " "


class FakeSTT(STT):
    def __init__(self, text: str = "what time is it") -> None:
        self.text = text

    async def transcribe(self, audio: AudioChunk, *, language=None) -> str:
        return self.text


class FakeTTS(TTS):
    sample_rate = 22050

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        self.spoken.append(text)
        yield np.zeros(256, dtype="float32")


class FakeMic(Microphone):
    """Yields a scripted list of frames, then stops."""

    def __init__(self, frames: list[np.ndarray], sample_rate: int = 16000) -> None:
        super().__init__(sample_rate=sample_rate, block_ms=30)
        self._frames = frames

    async def frames(self):  # type: ignore[override]
        for f in self._frames:
            # Yield control like the real mic (which awaits queue.get()) so the
            # background speak task can make progress between frames.
            await asyncio.sleep(0)
            yield f


class FakeSpeaker(Speaker):
    def __init__(self) -> None:
        super().__init__()
        self.played: list[np.ndarray] = []

    async def play(self, frames):  # type: ignore[override]
        # Mirror the real Speaker: play() does NOT clear the stop flag; the
        # pipeline resets it once per turn.
        async for f in frames:
            if self._stop.is_set():
                break
            self.played.append(f)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
