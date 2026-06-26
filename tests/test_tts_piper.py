"""Piper chunk-normalisation + dual-API selection (no piper/model needed)."""

from __future__ import annotations

import numpy as np

from aria.voice.tts_piper import PiperTTS, _to_float32


def test_to_float32_from_int16_bytes():
    pcm = np.array([0, 32767, -32768], dtype="<i2").tobytes()
    out = _to_float32(pcm)
    assert out.dtype == np.float32
    assert out[0] == 0.0
    assert abs(out[1] - 1.0) < 1e-3


def test_to_float32_from_float_array_chunk():
    class Chunk:
        audio_float_array = np.array([0.1, -0.2], dtype="float32")

    out = _to_float32(Chunk())
    assert out.tolist() == [float(np.float32(0.1)), float(np.float32(-0.2))]


def test_to_float32_from_int16_chunk():
    class Chunk:
        audio_int16_bytes = np.array([16383], dtype="<i2").tobytes()

    out = _to_float32(Chunk())
    assert abs(out[0] - 0.5) < 1e-3


def _bare_piper() -> PiperTTS:
    # Bypass __init__ (which needs piper + a model) to exercise path selection.
    return PiperTTS.__new__(PiperTTS)


def test_raw_chunks_uses_legacy_stream_raw():
    tts = _bare_piper()
    tts._length_scale = 1.0
    tts._legacy = True

    class Voice:
        def synthesize_stream_raw(self, text, length_scale):
            assert length_scale == 1.0
            yield b"\x00\x00"

    tts._voice = Voice()
    chunks = list(tts._raw_chunks("hi"))
    assert chunks == [b"\x00\x00"]


def test_raw_chunks_uses_new_synthesize():
    tts = _bare_piper()
    tts._length_scale = 1.0
    tts._legacy = False

    sentinel = object()

    class Voice:
        def synthesize(self, text, **kwargs):
            yield sentinel

    tts._voice = Voice()
    assert list(tts._raw_chunks("hi")) == [sentinel]
