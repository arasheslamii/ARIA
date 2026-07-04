"""The activation earcon: a short rising two-tone "I'm listening" chime.

Played the moment a capture starts (wake word heard, or the push-to-talk key
pressed) so the user always KNOWS Aria is live — no more talking into the void
wondering whether she woke up. Deliberately short (~180ms), quiet, and a pure
tone pair, so it can't be mistaken for speech by the VAD while the mic is open.
"""

from __future__ import annotations

import numpy as np

# A5 then D6: a quick rising pair reads as "go ahead" (falling would read "done").
_TONES: tuple[tuple[float, float], ...] = ((880.0, 0.07), (1174.66, 0.09))
_GAP_S = 0.02
_LEVEL = 0.18  # quiet: audible confirmation, not a doorbell
_RAMP_S = 0.008  # attack/release fade so the tone edges don't click


def make_chime(sample_rate: int) -> np.ndarray:
    """One float32 PCM buffer with the full chime at ``sample_rate``."""
    parts: list[np.ndarray] = []
    for freq, dur in _TONES:
        n = max(1, int(sample_rate * dur))
        t = np.arange(n) / sample_rate
        tone = np.sin(2 * np.pi * freq * t)
        ramp = max(1, int(sample_rate * _RAMP_S))
        env = np.ones(n)
        env[:ramp] = np.linspace(0.0, 1.0, ramp)
        env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        parts.append((tone * env * _LEVEL).astype("float32"))
        parts.append(np.zeros(int(sample_rate * _GAP_S), dtype="float32"))
    return np.concatenate(parts)
