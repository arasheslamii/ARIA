"""Piper TTS (local, CPU, no GPU) behind the :class:`TTS` interface.

Synthesis runs in a worker thread and yields audio frames as they are produced
so the pipeline can start playback on the first sentence. If the Piper voice
model is missing, construction raises and the caller can fall back / prompt the
wizard to fetch a voice.

Piper's Python API changed across releases, so we detect the available surface
at load time and normalise every chunk to a float32 mono array:
  * older ``piper-tts``: ``voice.synthesize_stream_raw(text, length_scale=…)``
    yields raw little-endian int16 PCM bytes;
  * newer ``piper-tts`` (1.3+): ``voice.synthesize(text, syn_config=…)`` yields
    ``AudioChunk`` objects exposing a float array / int16 bytes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import numpy as np

from aria.voice.base import TTS

_INT16_MAX = 32767.0


def _to_float32(chunk: object) -> np.ndarray:
    """Normalise one Piper output chunk (bytes or AudioChunk) to float32 mono."""
    if isinstance(chunk, (bytes, bytearray)):
        return np.frombuffer(bytes(chunk), dtype="<i2").astype("float32") / _INT16_MAX
    # Newer AudioChunk: prefer a ready-made float array, else int16 bytes.
    arr = getattr(chunk, "audio_float_array", None)
    if arr is not None:
        return np.asarray(arr, dtype="float32").reshape(-1)
    raw = getattr(chunk, "audio_int16_bytes", None)
    if raw is not None:
        return np.frombuffer(raw, dtype="<i2").astype("float32") / _INT16_MAX
    raise TypeError(f"Unrecognised Piper chunk type: {type(chunk)!r}")


class PiperTTS(TTS):
    def __init__(self, model_path: str | Path, speed: float = 1.0) -> None:
        from piper.voice import PiperVoice  # optional dependency

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Piper voice model not found: {path}")
        self._voice = PiperVoice.load(str(path))
        self.sample_rate = self._voice.config.sample_rate
        self._length_scale = 1.0 / max(speed, 0.1)
        # Pick the synthesis path once, up front.
        self._legacy = hasattr(self._voice, "synthesize_stream_raw")

    def _raw_chunks(self, text: str) -> Iterator[object]:
        if self._legacy:
            yield from self._voice.synthesize_stream_raw(
                text, length_scale=self._length_scale
            )
            return
        # Newer API: build a SynthesisConfig if the package ships one.
        syn_config = None
        try:
            from piper import SynthesisConfig  # type: ignore

            syn_config = SynthesisConfig(length_scale=self._length_scale)
        except Exception:  # pragma: no cover - version without SynthesisConfig
            pass
        if syn_config is not None:
            yield from self._voice.synthesize(text, syn_config=syn_config)
        else:
            yield from self._voice.synthesize(text)

    async def synthesize(self, text: str) -> AsyncIterator[np.ndarray]:
        # Piper is synchronous; pull chunks off a thread and convert to float32
        # frames for the Speaker so the event loop stays responsive.
        queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce() -> None:
            try:
                for chunk in self._raw_chunks(text):
                    loop.call_soon_threadsafe(queue.put_nowait, _to_float32(chunk))
            except Exception as exc:  # surface synthesis errors to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, produce)
        while True:
            frame = await queue.get()
            if frame is None:
                break
            if isinstance(frame, Exception):
                raise frame
            yield frame
