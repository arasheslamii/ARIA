"""Kokoro TTS (local, CPU, ONNX) behind the :class:`TTS` interface.

Kokoro-82M is a dramatic naturalness upgrade over Piper — near-human prosody —
while staying fully local and real-time on CPU. It needs two files (fetched by
the wizard, never at import time): the model (`kokoro-v1.0.onnx`, ~310 MB) and
the voice pack (`voices-v1.0.bin`, ~27 MB).

The `kokoro-onnx` package is an optional dependency: it's imported lazily so a
Piper-only install never pays for it, and a missing package/model raises a clear
FileNotFoundError that `build_tts` turns into a graceful Piper fallback.

Synthesis is synchronous ONNX inference, so it runs in a worker thread (same
pattern as PiperTTS) and the output is sliced into small frames so barge-in can
stop playback mid-sentence.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from aria.voice.base import TTS

SAMPLE_RATE = 24_000
# Slice synthesized audio into ~170ms frames so Speaker.stop() (barge-in) can
# interrupt between frames instead of waiting out a whole sentence.
_FRAME_SAMPLES = 4096

MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


class KokoroTTS(TTS):
    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> None:
        try:
            from kokoro_onnx import Kokoro  # optional dependency
        except ImportError as exc:
            raise FileNotFoundError(
                "The Kokoro voice engine isn't installed — run "
                "`pip install kokoro-onnx` (or re-run `aria setup` and pick a "
                "Piper voice instead)."
            ) from exc

        model, voices = Path(model_path), Path(voices_path)
        for p in (model, voices):
            if not p.exists():
                raise FileNotFoundError(
                    f"Kokoro voice model not found: {p}. Run `aria setup` and "
                    "pick a Kokoro voice to download it."
                )
        self._kokoro = Kokoro(str(model), str(voices))
        self._voice = voice
        self._speed = max(0.5, min(speed, 2.0))
        self.sample_rate = SAMPLE_RATE

    def _create(self, text: str) -> np.ndarray:
        samples, _sr = self._kokoro.create(
            text, voice=self._voice, speed=self._speed, lang="en-us"
        )
        return np.asarray(samples, dtype="float32").reshape(-1)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        # One blocking ONNX inference per sentence chunk, off the event loop.
        samples = await asyncio.to_thread(self._create, text)
        for start in range(0, samples.size, _FRAME_SAMPLES):
            yield samples[start : start + _FRAME_SAMPLES]
