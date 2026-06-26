"""Voice activity detection.

Silero VAD is the default (accurate, tiny). If torch/silero isn't installed we
fall back to a cheap energy gate so the pipeline still endpoints and barges in.
"""

from __future__ import annotations

import numpy as np

from aria.voice.base import VAD


class EnergyVAD(VAD):
    """Zero-dependency RMS energy gate. Good enough for barge-in fallback."""

    def __init__(self, threshold: float = 0.012) -> None:
        self.threshold = threshold

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        if frame.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return rms >= self.threshold


class SileroVAD(VAD):
    """Silero VAD via torch hub. Lazy-loaded; raises if torch is absent."""

    def __init__(self, threshold: float = 0.5) -> None:
        import torch  # local import: optional dependency

        self.threshold = threshold
        self._torch = torch
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self._model.eval()

    def is_speech(self, frame: np.ndarray, sample_rate: int) -> bool:
        # Silero expects 16k mono; chunk sizes of 512 work best.
        tensor = self._torch.from_numpy(frame.astype("float32"))
        with self._torch.no_grad():
            prob = float(self._model(tensor, sample_rate).item())
        return prob >= self.threshold

    def reset(self) -> None:
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()


def make_vad(backend: str, threshold: float) -> VAD:
    """Factory honouring config, with graceful fallback to energy VAD."""
    if backend == "silero":
        try:
            return SileroVAD(threshold=threshold)
        except Exception:  # torch missing or hub offline -> degrade, don't fail
            return EnergyVAD()
    return EnergyVAD()
