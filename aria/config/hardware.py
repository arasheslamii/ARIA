"""Machine capability probe → an honest local-model recommendation.

Used by `aria install-local` (and the wizard) to size the local brain to the
machine instead of guessing: enough RAM to hold the model without swapping, and
enough free disk for the download. Qwen 2.5 is the recommended family — strong
tool-calling at small sizes, which is the capability Aria actually needs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MachineProfile:
    ram_gb: float
    cpu_cores: int
    gpu: str | None  # GPU name if one is usable, else None
    free_disk_gb: float

    def describe(self) -> str:
        gpu = self.gpu or "none (CPU only)"
        return (
            f"{self.ram_gb:.0f} GB RAM, {self.cpu_cores} CPU cores, GPU: {gpu}, "
            f"{self.free_disk_gb:.0f} GB free disk"
        )


def _system_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return 0.0


def _detect_gpu() -> str | None:
    """Best-effort NVIDIA/AMD GPU detection (a GPU makes local models several
    times faster, but changes the recommendation, not the eligibility)."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:  # noqa: BLE001 - detection is cosmetic
            pass
    if Path("/dev/kfd").exists():  # AMD ROCm compute device
        return "AMD (ROCm)"
    return None


def probe_machine() -> MachineProfile:
    return MachineProfile(
        ram_gb=_system_ram_gb(),
        cpu_cores=os.cpu_count() or 1,
        gpu=_detect_gpu(),
        free_disk_gb=shutil.disk_usage(Path.home()).free / 1e9,
    )


# (min RAM GB, model tag, approx download GB). RAM thresholds leave headroom for
# the OS + Aria + the TTS/wake models; a model that swaps is worse than a smaller
# one that doesn't.
_TIERS: list[tuple[float, str, float]] = [
    (28.0, "qwen2.5:14b", 9.0),
    (14.0, "qwen2.5:7b", 4.7),
    (7.0, "qwen2.5:3b", 1.9),
    (3.5, "qwen2.5:1.5b", 1.0),
]


def recommend_local_model(profile: MachineProfile) -> tuple[str | None, str]:
    """(model_tag, human note). model_tag is None when the machine is too small
    for a local model that's actually worth talking to."""
    for min_ram, model, dl_gb in _TIERS:
        if profile.ram_gb >= min_ram:
            if profile.free_disk_gb < dl_gb + 2.0:
                return None, (
                    f"Your machine could run {model}, but there's only "
                    f"{profile.free_disk_gb:.0f} GB of free disk (needs ~{dl_gb:.0f} GB). "
                    "Free some space and try again."
                )
            speed = (
                f"With your {profile.gpu}, it will feel quick."
                if profile.gpu
                else "It runs on CPU — noticeably slower than the cloud brain, "
                     "but free, unlimited, and fully offline."
            )
            return model, (
                f"Recommended local model for this machine: {model} "
                f"(~{dl_gb:.0f} GB download). {speed}"
            )
    return None, (
        "This machine has too little RAM for a local model that's worth using — "
        "stick with the cloud brain."
    )
