"""Swappable voice interfaces.

Every concrete backend (Groq Whisper, faster-whisper, Piper, Silero, openWakeWord)
implements one of these so the pipeline never depends on a concrete vendor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

# Audio is passed around as float32 mono PCM in [-1, 1] at a known sample rate.
AudioArray = np.ndarray


@dataclass
class AudioChunk:
    pcm: AudioArray  # float32 mono
    sample_rate: int


class STT(ABC):
    """Speech-to-text."""

    @abstractmethod
    async def transcribe(self, audio: AudioChunk, *, language: str | None = None) -> str:
        """Transcribe a complete utterance."""


class TTS(ABC):
    """Text-to-speech. Streams audio so we can speak sentence-by-sentence."""

    sample_rate: int

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[AudioArray]:
        """Yield audio frames for ``text`` as they are produced."""


class VAD(ABC):
    """Voice activity detection — both endpointing and barge-in use this."""

    @abstractmethod
    def is_speech(self, frame: AudioArray, sample_rate: int) -> bool:
        """Return True if the frame contains speech."""

    def reset(self) -> None:  # pragma: no cover - optional stateful reset
        """Reset any internal state between utterances."""


class WakeWord(ABC):
    """Always-on wake-word spotter."""

    @abstractmethod
    def process(self, frame: AudioArray, sample_rate: int) -> float:
        """Return the wake-word probability for this frame [0, 1]."""

    def reset(self) -> None:  # pragma: no cover
        ...
