"""Wake-word spotting via openWakeWord (local, optional, default on).

If the model can't be loaded we degrade to a NullWakeWord that never triggers,
and the pipeline falls back to push-to-talk / always-listen mode.
"""

from __future__ import annotations

import numpy as np

from aria.voice.base import WakeWord


class NullWakeWord(WakeWord):
    """No-op spotter used when openWakeWord is unavailable."""

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        return 0.0


class OpenWakeWord(WakeWord):
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5) -> None:
        from pathlib import Path

        from openwakeword.model import Model  # optional dependency
        from openwakeword.utils import download_models

        self.threshold = threshold
        # ``model`` is either a stock openWakeWord name ("hey_jarvis") or a PATH
        # to a custom-trained .onnx/.tflite (e.g. a "hey topol" model). Paths
        # load as-is; their scores are keyed by the file's stem.
        is_path = model.endswith((".onnx", ".tflite")) or "/" in model
        self.model_name = Path(model).stem if is_path else model
        if is_path and not Path(model).expanduser().exists():
            raise FileNotFoundError(
                f"Custom wake-word model not found: {model} — check "
                "[wakeword] model in the config."
            )
        if not is_path:
            # Ensure the stock model + feature models are cached (no-op after).
            try:
                download_models([model])
            except Exception:  # offline + already cached is fine; load will tell
                pass
        # Force the ONNX backend: the bundled tflite_runtime is built against
        # numpy<2 and crashes under numpy 2.x, while onnxruntime supports it.
        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        # openWakeWord keys scores by the model basename (e.g. "hey_jarvis_v0.1").
        self._score_keys = list(self._model.models.keys())

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        # openWakeWord expects int16 PCM.
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype("int16")
        scores = self._model.predict(pcm16)
        # Match by exact key or prefix so "hey_jarvis" finds "hey_jarvis_v0.1".
        best = 0.0
        for key, val in scores.items():
            if key == self.model_name or key.startswith(self.model_name):
                best = max(best, float(val))
        return best

    def reset(self) -> None:
        if hasattr(self._model, "reset"):
            self._model.reset()


def make_wakeword(enabled: bool, model: str, threshold: float) -> WakeWord:
    if not enabled:
        return NullWakeWord()
    try:
        return OpenWakeWord(model=model, threshold=threshold)
    except Exception:  # model/asset missing -> degrade gracefully
        return NullWakeWord()
