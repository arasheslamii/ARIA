"""More natural local TTS: voice catalog, Piper TTS-interface conformance, and
config-driven voice selection (FIX 4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aria.config.loader import state_dir
from aria.tui.voices import _PATHS, SAMPLE_TEXT, VOICES


def test_voice_catalog_is_consistent():
    assert set(VOICES) <= set(_PATHS)  # every offered voice is downloadable
    assert "en_US-amy-medium" in VOICES  # fast default present
    # Includes more natural / higher-quality options than before.
    assert any("very natural" in d or "natural" in d for d in VOICES.values())
    assert any(v.endswith("-high") for v in VOICES)
    assert SAMPLE_TEXT  # an audition sample exists


def _local_voice() -> Path | None:
    p = state_dir() / "models" / "en_US-amy-medium.onnx"
    return p if p.exists() else None


async def test_piper_conforms_to_tts_interface():
    voice = _local_voice()
    if voice is None:
        pytest.skip("no local Piper voice available")
    from aria.voice.base import TTS
    from aria.voice.tts_piper import PiperTTS

    tts = PiperTTS(voice)
    assert isinstance(tts, TTS)
    assert isinstance(tts.sample_rate, int) and tts.sample_rate > 0
    frames = [f async for f in tts.synthesize("Hello there, this is a test.")]
    assert frames, "synthesize yielded no audio"
    assert all(isinstance(f, np.ndarray) and f.dtype == np.float32 for f in frames)


def test_voice_selectable_via_config(tmp_path, monkeypatch):
    from aria.app import resolve_piper_model
    from aria.config.schema import AriaConfig

    models = tmp_path / "models"
    models.mkdir()
    (models / "en_GB-alba-medium.onnx").write_bytes(b"fake")
    monkeypatch.setenv("ARIA_MODELS_DIR", str(models))

    cfg = AriaConfig()
    cfg.tts.voice = "en_GB-alba-medium"
    assert resolve_piper_model(cfg).name == "en_GB-alba-medium.onnx"
