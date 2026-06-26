"""Local STT fallback via faster-whisper (CTranslate2).

Behind the same :class:`STT` interface as the cloud default. Pulled in only when
the ``local-stt`` extra is installed; transcription runs in a thread so it never
blocks the event loop.
"""

from __future__ import annotations

import asyncio

from aria.voice.base import STT, AudioChunk


class FasterWhisperSTT(STT):
    def __init__(self, model_size: str = "base.en", compute_type: str = "int8") -> None:
        from faster_whisper import WhisperModel  # optional dependency

        self._model = WhisperModel(model_size, compute_type=compute_type)

    async def transcribe(self, audio: AudioChunk, *, language: str | None = None) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio, language)

    def _transcribe_sync(self, audio: AudioChunk, language: str | None) -> str:
        segments, _ = self._model.transcribe(
            audio.pcm, language=language, vad_filter=True
        )
        return " ".join(seg.text for seg in segments).strip()
