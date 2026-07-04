"""Voice catalog + downloader (Kokoro + Piper).

The .deb bundles one default Piper voice. The wizard can fetch additional voices
into the state dir on demand (never in postinst — installs stay fast and
offline-safe). Kokoro voices share two model files (~340 MB total, one-time
download) and are by far the most natural — they're listed first.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from aria.config.loader import state_dir
from aria.voice.tts_kokoro import MODEL_FILE as _KOKORO_MODEL
from aria.voice.tts_kokoro import VOICES_FILE as _KOKORO_VOICES_BIN

# Kokoro-82M voices: near-human prosody, fully local, real-time on CPU. All of
# them share the same two model files, so switching between them is instant once
# the one-time download is done.
KOKORO_VOICES: dict[str, str] = {
    "af_heart": "American English, warm female — most natural (Kokoro)",
    "af_bella": "American English, bright female — very natural (Kokoro)",
    "af_sarah": "American English, calm female — very natural (Kokoro)",
    "am_michael": "American English, warm male — very natural (Kokoro)",
    "am_adam": "American English, deep male — very natural (Kokoro)",
    "bf_emma": "British English, natural female (Kokoro)",
    "bm_george": "British English, natural male (Kokoro)",
}

_KOKORO_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)

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


def voice_catalog() -> dict[str, str]:
    """Every offerable voice (Kokoro first — they sound best) for the wizard."""
    return {**KOKORO_VOICES, **VOICES}


def is_kokoro(voice: str) -> bool:
    return voice in KOKORO_VOICES


def installed(voice: str) -> bool:
    if is_kokoro(voice):
        d = voice_dir()
        return (d / _KOKORO_MODEL).exists() and (d / _KOKORO_VOICES_BIN).exists()
    return (voice_dir() / f"{voice}.onnx").exists() or (
        Path(__file__).parents[1] / "packaging" / "models" / f"{voice}.onnx"
    ).exists()


async def _fetch(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Stream a (possibly large) file to disk via a temp name, so an interrupted
    download never leaves a truncated file that looks installed."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            async for chunk in resp.aiter_bytes(1 << 20):
                await asyncio.to_thread(fh.write, chunk)
    tmp.rename(dest)


async def download_voice(voice: str) -> Path:
    """Fetch the files for a voice into the state dir (idempotent)."""
    if is_kokoro(voice):
        d = voice_dir()
        async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
            for name in (_KOKORO_MODEL, _KOKORO_VOICES_BIN):
                dest = d / name
                if not dest.exists():
                    await _fetch(client, f"{_KOKORO_BASE}/{name}", dest)
        return d / _KOKORO_MODEL
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
