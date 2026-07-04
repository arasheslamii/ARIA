"""Activation methods (0.7.0): hold-to-talk, the "I'm listening" chime, and the
shorter follow-up windows."""

from __future__ import annotations

import numpy as np
import pytest

from aria.config.schema import (
    ActivationConfig,
    AudioConfig,
    ConversationConfig,
    VADConfig,
    WakeWordConfig,
)
from aria.voice.chime import make_chime
from aria.voice.hotkey import KEY_CHOICES, keyboards_with_key, resolve_key
from aria.voice.pipeline import _FOLLOWUP_WINDOW_S, VoicePipeline
from aria.voice.vad import EnergyVAD
from tests.conftest import FakeMic, FakeSpeaker, FakeSTT, FakeTTS
from tests.test_voice_loop import _silence, _speech


# --- key resolution + device discovery (pure, no /dev/input needed) ---------
def test_resolve_key_maps_friendly_names():
    assert resolve_key("right ctrl") == 97
    assert resolve_key("  F8 ") == 66
    with pytest.raises(ValueError, match="Choices"):
        resolve_key("turbo button")
    assert set(KEY_CHOICES) >= {"right ctrl", "caps lock", "f8"}


_PROC_SAMPLE = """\
I: Bus=0019 Vendor=0000 Product=0005 Version=0000
N: Name="Lid Switch"
H: Handlers=event0
B: EV=21
B: SW=1

I: Bus=0011 Vendor=0001 Product=0001 Version=ab54
N: Name="AT Translated Set 2 keyboard"
H: Handlers=sysrq kbd event3 leds
B: EV=120013
B: KEY=402000000 3803078f800d001 feffffdfffefffff fffffffffffffffe

I: Bus=0003 Vendor=046d Product=c077 Version=0111
N: Name="Logitech USB Optical Mouse"
H: Handlers=mouse0 event4
B: EV=17
B: KEY=1f0000 0 0 0 0
"""


def test_keyboards_with_key_finds_only_real_keyboards():
    # RIGHTCTRL (97) is on the AT keyboard's bitmask, not the mouse or lid switch.
    assert keyboards_with_key(97, _PROC_SAMPLE) == ["/dev/input/event3"]
    assert keyboards_with_key(66, _PROC_SAMPLE) == ["/dev/input/event3"]  # F8
    # A code no device has (BTN_TOUCH region beyond both masks).
    assert keyboards_with_key(0x2FF, _PROC_SAMPLE) == []


# --- the chime ---------------------------------------------------------------
def test_chime_is_short_quiet_and_clickless():
    pcm = make_chime(22050)
    assert pcm.dtype == np.float32
    assert 0.1 < len(pcm) / 22050 < 0.3  # ~200ms: noticeable, not a doorbell
    assert np.max(np.abs(pcm)) <= 0.2  # quiet
    assert abs(pcm[0]) < 0.01 and abs(pcm[-1]) < 0.01  # faded edges, no click


# --- push-to-talk in the pipeline --------------------------------------------
class FakeKey:
    pressed = False


class BoomWake:
    """Proves wake-word spotting is OFF in pure hotkey mode."""

    def process(self, frame, sample_rate):
        raise AssertionError("wake word must not be consulted in hotkey mode")

    def reset(self):
        pass


class OnceWakeQuiet:
    def __init__(self) -> None:
        self._used = False

    def process(self, frame, sample_rate) -> float:
        if self._used:
            return 0.0
        if float(np.sqrt(np.mean(np.square(frame)))) > 0.05:
            self._used = True
            return 1.0
        return 0.0

    def reset(self):
        pass


class KeyedMic(FakeMic):
    """Holds `key` down for frames [press_at, release_at)."""

    def __init__(self, frames, key: FakeKey, press_at: int, release_at: int) -> None:
        super().__init__(frames)
        self._key = key
        self._press = press_at
        self._release = release_at

    async def frames(self):  # type: ignore[override]
        i = 0
        async for f in super().frames():
            self._key.pressed = self._press <= i < self._release
            i += 1
            yield f


def _ptt_pipeline(mic, key, *, mode="hotkey", chime=False, wakeword=None):
    return VoicePipeline(
        stt=FakeSTT("what's the weather"),
        tts=FakeTTS(),
        vad=EnergyVAD(threshold=0.01),
        wakeword=wakeword or BoomWake(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=False),
        wake_cfg=WakeWordConfig(enabled=True),
        conversation_cfg=ConversationConfig(enabled=False),
        activation_cfg=ActivationConfig(mode=mode, chime=chime),
        hotkey=key,
        mic=mic,
        speaker=FakeSpeaker(),
    )


async def test_hold_key_talk_release_answers_without_wake_word():
    key = FakeKey()
    # ALL speech, NO trailing silence: only the key RELEASE can end this capture.
    frames = [_speech() for _ in range(40)]
    mic = KeyedMic(frames, key, press_at=0, release_at=20)
    transcripts: list[str] = []

    async def respond(t):
        transcripts.append(t)
        yield "Sunny."

    await _ptt_pipeline(mic, key).run(respond)
    assert transcripts == ["what's the weather"]  # captured, wake never consulted


async def test_a_tap_too_short_to_be_speech_is_dropped():
    key = FakeKey()
    frames = [_silence() for _ in range(30)]
    mic = KeyedMic(frames, key, press_at=0, release_at=10)  # held over silence
    transcripts: list[str] = []

    async def respond(t):
        transcripts.append(t)
        yield "?"

    await _ptt_pipeline(mic, key).run(respond)
    assert transcripts == []  # no voiced audio -> never sent to STT


async def test_hybrid_mode_keeps_the_wake_word_working():
    key = FakeKey()  # never pressed
    frames = [_speech() for _ in range(12)] + [_silence() for _ in range(25)]
    mic = KeyedMic(frames, key, press_at=999, release_at=999)
    transcripts: list[str] = []

    async def respond(t):
        transcripts.append(t)
        yield "Hello!"

    await _ptt_pipeline(mic, key, mode="hybrid", wakeword=OnceWakeQuiet()).run(respond)
    assert len(transcripts) == 1


async def test_activation_plays_the_listening_chime():
    key = FakeKey()
    frames = [_speech() for _ in range(40)]
    mic = KeyedMic(frames, key, press_at=0, release_at=20)
    pipeline = _ptt_pipeline(mic, key, chime=True)

    async def respond(t):
        yield "Sunny."

    await pipeline.run(respond)
    played = pipeline.speaker.played
    assert played, "nothing was played at all"
    # The FIRST thing out of the speaker is the chime, before any TTS audio.
    assert len(played[0]) == len(make_chime(FakeTTS.sample_rate))


# --- the shorter windows ------------------------------------------------------
def test_followup_windows_are_four_seconds():
    assert ConversationConfig().window_s == 4.0
    assert _FOLLOWUP_WINDOW_S == 4.0


def test_activation_defaults():
    a = ActivationConfig()
    assert a.mode == "wake_word"  # opt-in: nothing changes until the user picks
    assert a.hotkey in KEY_CHOICES
    assert a.chime is True
