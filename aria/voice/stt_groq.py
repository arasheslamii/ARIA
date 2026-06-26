"""Groq Whisper STT (default).

Uses ``whisper-large-v3-turbo`` over Groq's audio endpoint. Audio is encoded to
an in-memory WAV (no temp files) before upload.
"""

from __future__ import annotations

import io
import wave

import numpy as np
from groq import AsyncGroq

from aria.voice.base import STT, AudioChunk


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


class GroqSTT(STT):
    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self.model = model

    async def transcribe(self, audio: AudioChunk, *, language: str | None = None) -> str:
        wav = _to_wav_bytes(audio.pcm, audio.sample_rate)
        resp = await self._client.audio.transcriptions.create(
            file=("speech.wav", wav, "audio/wav"),
            model=self.model,
            language=language,
            response_format="text",
        )
        # SDK returns a str for response_format="text", else an object.
        return resp if isinstance(resp, str) else getattr(resp, "text", "")
