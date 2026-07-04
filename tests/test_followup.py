"""Conversational follow-up: after a turn that expects a reply (a pending
confirmation), the pipeline re-opens the mic with NO wake word so the user can just
say "yes". A normal turn still returns to IDLE; a silent follow-up times out."""

from __future__ import annotations

import numpy as np

from aria.config.schema import AudioConfig, VADConfig, WakeWordConfig
from aria.voice.base import WakeWord
from aria.voice.pipeline import State, VoicePipeline
from aria.voice.vad import EnergyVAD
from tests.conftest import FakeSpeaker, FakeSTT, FakeTTS
from tests.test_voice_loop import _silence, _speech


class OnceWake(WakeWord):
    """Fires exactly once, on the first speech burst ever, then never again — so a
    SECOND turn can ONLY happen via follow-up listening, not the wake word. reset()
    deliberately does not re-arm it."""

    def __init__(self) -> None:
        self._used = False

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        if self._used:
            return 0.0
        if float(np.sqrt(np.mean(np.square(frame)))) > 0.05:
            self._used = True
            return 1.0
        return 0.0

    def reset(self) -> None:
        pass  # stay used-up


def _pipeline(mic_frames, *, awaiting, vad_cfg=None):
    from tests.conftest import FakeMic

    states: list[State] = []
    pipeline = VoicePipeline(
        stt=FakeSTT("yes"),
        tts=FakeTTS(),
        vad=EnergyVAD(threshold=0.01),
        wakeword=OnceWake(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=vad_cfg or VADConfig(silence_ms=300, barge_in=False),
        wake_cfg=WakeWordConfig(enabled=True),  # wake REQUIRED unless we follow up
        on_state=states.append,
        awaiting_reply=awaiting,
        mic=FakeMic(mic_frames),
        speaker=FakeSpeaker(),
    )
    return pipeline, states


async def test_pending_confirmation_reopens_mic_without_wake_word():
    # Turn 1 asks for confirmation (sets pending); turn 2 ("yes") must be captured
    # via follow-up listening, since the wake word only ever fires once.
    frames = ([_speech() for _ in range(5)] + [_silence() for _ in range(25)]) * 2
    state = {"turn": 0, "pending": False}
    transcripts: list[str] = []

    async def respond(transcript: str):
        state["turn"] += 1
        transcripts.append(transcript)
        if state["turn"] == 1:
            state["pending"] = True  # like the orchestrator stashing a _Pending
            yield "Should I go ahead? Say yes or no."
        else:
            state["pending"] = False  # confirmation resolved
            yield "Done."

    pipeline, states = _pipeline(frames, awaiting=lambda: state["pending"])
    await pipeline.run(respond)

    assert len(transcripts) == 2  # the reply was heard with NO second wake word
    # The post-question transition was SPEAKING -> LISTENING (follow-up), not IDLE.
    assert _has_subsequence(states, [State.SPEAKING, State.LISTENING])
    assert pipeline.state is State.IDLE  # resolved -> back to idle at the end


async def test_normal_turn_returns_to_idle_and_requires_wake_word():
    # No pending: after speaking, IDLE. The second utterance is ignored because the
    # wake word won't fire again.
    frames = ([_speech() for _ in range(5)] + [_silence() for _ in range(25)]) * 2
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "It's noon."

    pipeline, _states = _pipeline(frames, awaiting=lambda: False)
    await pipeline.run(respond)

    assert len(transcripts) == 1  # only the wake-triggered turn ran
    assert pipeline.state is State.IDLE


async def test_followup_times_out_to_idle_and_keeps_pending(monkeypatch):
    # Pending stays set, but no one answers -> after the window we fall back to
    # wake-word IDLE without dropping the pending state.
    monkeypatch.setattr("aria.voice.pipeline._FOLLOWUP_WINDOW_S", 0.0)  # expire at once
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(40)]
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "Should I go ahead? Say yes or no."  # leaves pending True forever

    pipeline, states = _pipeline(frames, awaiting=lambda: True)
    await pipeline.run(respond)

    assert len(transcripts) == 1  # the silent follow-up never captured a 2nd turn
    assert pipeline.state is State.IDLE  # timed out back to idle
    # We entered follow-up listening at least once before giving up.
    assert _has_subsequence(states, [State.SPEAKING, State.LISTENING, State.IDLE])


def _has_subsequence(seq: list, sub: list) -> bool:
    it = iter(seq)
    return all(any(x == s for x in it) for s in sub)
