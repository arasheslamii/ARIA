"""Audio I/O over PortAudio (sounddevice).

Microphone capture is an async generator of fixed-size frames. Playback is
cancellable for barge-in: calling :meth:`Speaker.stop` halts the current
utterance immediately.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

try:  # sounddevice import fails without PortAudio; keep import-time safe.
    import sounddevice as sd
except (OSError, ImportError):  # pragma: no cover - environment dependent
    sd = None  # type: ignore[assignment]


class AudioError(RuntimeError):
    """Raised when the mic/speaker is missing or locked."""


def ensure_audio() -> None:
    if sd is None:
        raise AudioError(
            "PortAudio/sounddevice is unavailable. Install libportaudio2 and "
            "check that an audio device is present."
        )


class Microphone:
    """Async frame source. Yields float32 mono frames of ``block_ms``."""

    def __init__(
        self,
        sample_rate: int = 16000,
        block_ms: int = 30,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.block_size = int(sample_rate * block_ms / 1000)
        self.device = device

    async def frames(self) -> AsyncIterator[np.ndarray]:
        ensure_audio()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            # Runs on PortAudio's thread; hand frames to the loop thread-safely.
            if status:  # overflow/underflow — drop, don't crash the loop.
                pass
            loop.call_soon_threadsafe(self._offer, queue, indata[:, 0].copy())

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=callback,
            ):
                while True:
                    yield await queue.get()
        except sd.PortAudioError as exc:  # pragma: no cover
            raise AudioError(f"Could not open microphone: {exc}") from exc

    @staticmethod
    def _offer(queue: asyncio.Queue, frame: np.ndarray) -> None:
        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass  # prefer fresh audio over backpressure


class Speaker:
    """Streaming, interruptible playback for sentence-by-sentence TTS."""

    def __init__(self, sample_rate: int = 22050, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """Barge-in: stop speaking right now."""
        self._stop.set()

    def reset(self) -> None:
        """Clear a stale barge-in stop. Must be called at the START of each
        utterance — otherwise a stop set on one turn would mute every later turn
        (``_speak`` checks ``interrupted`` before ``play`` runs)."""
        self._stop.clear()

    @property
    def interrupted(self) -> bool:
        return self._stop.is_set()

    async def play(self, frames: AsyncIterator[np.ndarray]) -> None:
        """Play an async stream of float32 frames until interrupted or done.

        Does NOT clear the stop flag — the pipeline resets it once per turn via
        :meth:`reset` so a barge-in mid-utterance persists across the remaining
        sentences of that utterance."""
        ensure_audio()
        loop = asyncio.get_running_loop()
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
            )
        except sd.PortAudioError as exc:  # pragma: no cover
            raise AudioError(f"Could not open speaker: {exc}") from exc

        with stream:
            async for frame in frames:
                if self._stop.is_set():
                    break
                # Offload the blocking write so the event loop stays responsive
                # (and can detect barge-in concurrently).
                await loop.run_in_executor(None, stream.write, frame)
