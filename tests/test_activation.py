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


# --- recovery: a mid-turn crash must never leave her deaf (the "dead until
# reboot" bug: the daemon re-enters run() but the state machine was stuck in
# THINKING, which the loop only ever leaves — so every frame was ignored).
class ExplodingOnceSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__("hello")
        self.calls = 0

    async def transcribe(self, audio, *, language=None):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("network died mid-transcription")
        return "hello again"


async def test_crashed_turn_recovers_on_rerun_like_the_daemon_does():
    key = FakeKey()
    stt = ExplodingOnceSTT()
    transcripts: list[str] = []

    async def respond(t):
        transcripts.append(t)
        yield "Hi!"

    def pipeline_with(frames):
        mic = KeyedMic(frames, key, press_at=0, release_at=20)
        p = _ptt_pipeline(mic, key)
        p.stt = stt
        return p

    frames = [_speech() for _ in range(40)]
    pipeline = pipeline_with(frames)
    with pytest.raises(RuntimeError, match="network died"):
        await pipeline.run(respond)
    # The crash left the machine mid-turn — the daemon now re-runs the SAME
    # pipeline object (run_with_mic_retry), like after any transient failure.
    key.pressed = False
    pipeline.mic = KeyedMic([_speech() for _ in range(40)], key, 0, 20)
    await pipeline.run(respond)
    assert transcripts == ["hello again"]  # she woke up again, no reboot needed


async def test_crash_while_speaking_also_recovers(monkeypatch):
    # Same idea, but the death happens in the reply stream (TTS/LLM path),
    # leaving state SPEAKING with a leftover _ptt_hold.
    key = FakeKey()
    calls = {"n": 0}

    async def respond(t):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider exploded mid-reply")
        yield "Recovered."

    frames = [_speech() for _ in range(40)]
    pipeline = _ptt_pipeline(KeyedMic(frames, key, 0, 20), key)
    with pytest.raises(RuntimeError, match="exploded"):
        await pipeline.run(respond)
    key.pressed = False
    pipeline.mic = KeyedMic([_speech() for _ in range(40)], key, 0, 20)
    await pipeline.run(respond)
    assert calls["n"] == 2  # the second turn was heard and answered


# --- 0.9.2: the chime must never be part of the capture ----------------------
class LengthSpySTT(FakeSTT):
    def __init__(self, text="what's the weather") -> None:
        super().__init__(text)
        self.sample_counts: list[int] = []

    async def transcribe(self, audio, *, language=None):
        self.sample_counts.append(audio.pcm.size)
        return self.text


async def test_chime_span_is_discarded_from_the_capture():
    """The live bug: the chime bled from the speakers into the mic, Whisper
    transcribed it as 'BEEP', and she told the user they had trouble speaking."""
    key = FakeKey()
    frames = [_speech() for _ in range(40)]
    mic = KeyedMic(frames, key, press_at=0, release_at=30)
    stt = LengthSpySTT()
    pipeline = _ptt_pipeline(mic, key, chime=True)
    pipeline.stt = stt
    transcripts: list[str] = []

    async def respond(t):
        transcripts.append(t)
        yield "Sunny."

    await pipeline.run(respond)
    assert transcripts == ["what's the weather"]
    # The chime span (~0.35s = ~11 frames of 480 samples) was NOT recorded.
    skipped = int((len(make_chime(FakeTTS.sample_rate)) / FakeTTS.sample_rate + 0.15)
                  * 1000 / 30)
    assert skipped >= 8
    # 30 pressed frames minus the chime span (trigger + release frames included).
    assert stt.sample_counts[0] <= (30 - skipped + 1) * 480


async def test_junk_transcripts_never_reach_the_brain():
    from aria.voice.pipeline import _is_junk

    for junk in ("BEEP", ".", "Oh", "Uh...", "hmm", "you", "Mm-hm..."[:3]):
        assert _is_junk(junk), junk
    for real in ("yes", "no", "yeah", "okay", "stop", "what time is it"):
        assert not _is_junk(real), real

    # End-to-end: a capture that transcribes to junk is dropped silently.
    key = FakeKey()
    frames = [_speech() for _ in range(30)]
    mic = KeyedMic(frames, key, press_at=0, release_at=20)
    pipeline = _ptt_pipeline(mic, key)
    pipeline.stt = FakeSTT("BEEP")
    heard: list[str] = []

    async def respond(t):
        heard.append(t)
        yield "?"

    await pipeline.run(respond)
    assert heard == []  # she stays quiet instead of "you seem to have trouble"
