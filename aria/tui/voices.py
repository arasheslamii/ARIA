"""Piper voice catalog + downloader.

The .deb bundles one default voice. The wizard can fetch additional voices into
the state dir on demand (never in postinst — installs stay fast and offline-safe).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from aria.config.loader import state_dir

# A curated set of natural English Piper voices. Medium voices stream in
# real-time on CPU (first word < ~1.2s); the *-high voices sound a touch more
# natural but synthesize a bit slower.
VOICES: dict[str, str] = {
    "en_US-amy-medium": "American English, warm female — fast (default)",
    "en_US-hfc_female-medium": "American English, very natural female — fast",
    "en_US-kristin-medium": "American English, friendly female — fast",
    "en_GB-jenny_dioco-medium": "British English, natural female — fast",
    "en_US-ryan-high": "American English, clear male — natural, slightly slower",
    "en_US-lessac-high": "American English, neutral — natural, slightly slower",
    "en_GB-alba-medium": "British English, female — fast",
}

_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# Path fragments per voice (lang/region/name/quality).
_PATHS = {
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-hfc_female-medium": "en/en_US/hfc_female/medium/en_US-hfc_female-medium",
    "en_US-kristin-medium": "en/en_US/kristin/medium/en_US-kristin-medium",
    "en_GB-jenny_dioco-medium": "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high",
    "en_US-lessac-high": "en/en_US/lessac/high/en_US-lessac-high",
    "en_GB-alba-medium": "en/en_GB/alba/medium/en_GB-alba-medium",
}

# A short, natural sentence the wizard plays to audition a voice.
SAMPLE_TEXT = "Hi, I'm Aria. This is what I'll sound like."


def voice_dir() -> Path:
    d = state_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def installed(voice: str) -> bool:
    return (voice_dir() / f"{voice}.onnx").exists() or (
        Path(__file__).parents[1] / "packaging" / "models" / f"{voice}.onnx"
    ).exists()


async def download_voice(voice: str) -> Path:
    """Fetch the .onnx + .json for a voice into the state dir."""
    if voice not in _PATHS:
        raise ValueError(f"unknown voice: {voice}")
    out = voice_dir() / f"{voice}.onnx"
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for suffix in ("", ".json"):
            url = f"{_BASE}/{_PATHS[voice]}.onnx{suffix}"
            dest = out if suffix == "" else Path(str(out) + ".json")
            if dest.exists():
                continue
            resp = await client.get(url)
            resp.raise_for_status()
            await asyncio.to_thread(dest.write_bytes, resp.content)
    return out
