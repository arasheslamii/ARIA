"""End-to-end voice-loop smoke test — fully mocked, no audio, no network.

Exercises the real pipeline state machine: wake → VAD-endpointed capture → STT →
respond() → sentencizer → TTS → speaker, plus the orchestrator turn path and the
sentencizer's "speak sentence 1 first" behaviour.
"""

from __future__ import annotations

import time

import numpy as np

from aria.config.schema import AudioConfig, VADConfig, WakeWordConfig
from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.tools.base import ToolRegistry
from aria.voice.pipeline import _BARGE_GRACE_S, State, VoicePipeline
from aria.voice.sentencizer import sentence_chunks
from aria.voice.vad import EnergyVAD
from aria.voice.wakeword import NullWakeWord
from tests.conftest import FakeLLM, FakeMic, FakeSpeaker, FakeSTT, FakeTTS


def _speech(n: int = 480) -> np.ndarray:
    return (np.random.randn(n) * 0.2).astype("float32")


def _silence(n: int = 480) -> np.ndarray:
    return np.zeros(n, dtype="float32")


async def _collect(aiter):
    return [x async for x in aiter]


async def test_sentencizer_flushes_first_sentence_early():
    async def deltas():
        for tok in ["Hello", " there.", " How", " are", " you", " today?"]:
            yield tok

    out = await _collect(sentence_chunks(deltas()))
    assert out[0].startswith("Hello there.")
    assert any("today?" in s for s in out)


async def test_orchestrator_chitchat_streams(tmp_path):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    llm = FakeLLM(
        stream_text="Hi! How can I help?",
        chat_queue=[__import__("aria.llm.base", fromlist=["ChatResult"]).ChatResult(
            content='{"route":"chitchat","needs_tools":[],"reason":"greeting"}'
        )],
    )
    orch = Orchestrator(
        llm=llm,
        registry=ToolRegistry(),
        memory=mem,
        reasoning_model="big",
        fast_model="small",
    )
    deltas = await _collect(orch.respond("hello"))
    assert "".join(deltas).strip() == "Hi! How can I help?"
    # The user + assistant turns were persisted.
    assert (await mem.recent_turns())[-1][0] == "assistant"
    await mem.close()


async def test_pipeline_runs_one_full_turn(tmp_path):
    # Build a frame script: speech then trailing silence to trigger endpointing.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(40)]
    mic = FakeMic(frames)
    speaker = FakeSpeaker()
    tts = FakeTTS()

    spoken_states = []
    latencies = []
    pipeline = VoicePipeline(
        stt=FakeSTT("what time is it"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),  # disabled wake word -> trip on first speech
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=False),
        wake_cfg=WakeWordConfig(enabled=False),
        on_state=lambda s: spoken_states.append(s),
        on_latency=lambda resp, wake: latencies.append((resp, wake)),
        mic=mic,
        speaker=speaker,
    )

    async def respond(transcript: str):
        assert transcript == "what time is it"
        for tok in ["It's", " almost", " noon."]:
            yield tok + " "

    await pipeline.run(respond)
    # Aria spoke at least one sentence through TTS.
    assert tts.spoken, "expected TTS to be invoked"
    assert "noon" in " ".join(tts.spoken)
    # Latency was reported exactly once, for the first spoken word, with sane
    # ordering (wake precedes end-of-speech, so wake delta >= response delta).
    assert len(latencies) == 1
    resp_dt, wake_dt = latencies[0]
    assert resp_dt >= 0 and wake_dt >= resp_dt


async def test_pipeline_multi_turn_with_bargein(tmp_path):
    # Regression for "async generator is already running": with barge_in=True the
    # old code opened a SECOND iterator on the mic generator, crashing on turn 2.
    # Two utterances, each: speech then enough trailing silence to endpoint.
    frames = ([_speech() for _ in range(5)] + [_silence() for _ in range(25)]) * 2
    mic = FakeMic(frames)
    speaker = FakeSpeaker()
    tts = FakeTTS()

    turns: list[str] = []
    pipeline = VoicePipeline(
        stt=FakeSTT("what time is it"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),  # always-listen: speech itself triggers a turn
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=True),  # the REAL default
        wake_cfg=WakeWordConfig(enabled=False),
        mic=mic,
        speaker=speaker,
    )

    async def respond(transcript: str):
        turns.append(transcript)
        yield "On it. "

    # Must complete both turns without raising RuntimeError from a 2nd consumer.
    await pipeline.run(respond)

    assert turns == ["what time is it", "what time is it"]  # both turns ran
    assert len(tts.spoken) == 2  # TTS invoked once per turn
    assert pipeline.state is State.IDLE  # cleanly back to idle
    assert not speaker.interrupted  # stop flag clear at end


def _barge_pipeline(*, barge_in=True, armed=True, in_grace=False, block_ms=30) -> VoicePipeline:
    p = VoicePipeline(
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=block_ms),
        vad_cfg=VADConfig(silence_ms=300, barge_in=barge_in),
        wake_cfg=WakeWordConfig(enabled=False),
        mic=FakeMic([]),
        speaker=FakeSpeaker(),
    )
    if armed:
        now = time.perf_counter()
        p._last_frame_ts = now  # audio currently playing
        # In-grace => burst just started (filler); else burst is past the grace.
        p._burst_start = now if in_grace else now - (_BARGE_GRACE_S + 1.0)
    return p


def test_speaker_reset_clears_interrupted():
    speaker = FakeSpeaker()
    speaker.stop()
    assert speaker.interrupted
    speaker.reset()
    assert not speaker.interrupted


def test_barge_disarmed_before_audio_starts():
    # The reported bug: barge counted during the silent THINKING window.
    p = _barge_pipeline(armed=False)
    count = 0
    for _ in range(50):  # 50 frames of "speech" before any audio plays
        count, stop = p._barge_check(_speech(), 16000, count)
        assert not stop
    assert count == 0  # never accumulates pre-audio


def test_barge_disarmed_during_grace_period():
    # FIX 1: a short self-spoken burst (the filler) is still inside the grace
    # window, so even sustained "speech" (her own voice) must NOT trip barge-in.
    p = _barge_pipeline(armed=True, in_grace=True)
    count = 0
    for _ in range(p._barge_frames + 5):
        count, stop = p._barge_check(_speech(), 16000, count)
        assert not stop
    assert count == 0


def test_barge_requires_sustained_speech_then_stops():
    p = _barge_pipeline(armed=True)  # past the grace period
    assert p._barge_frames == round(280 / 30)  # ~9 frames (~280ms)
    count = 0
    for _ in range(p._barge_frames - 1):  # one short of the threshold
        count, stop = p._barge_check(_speech(), 16000, count)
        assert not stop
    count, stop = p._barge_check(_speech(), 16000, count)  # crosses threshold
    assert stop  # genuine sustained over-speech AFTER grace DOES stop playback


def test_single_noisy_frame_does_not_barge():
    p = _barge_pipeline(armed=True)
    count, stop = p._barge_check(_speech(), 16000, 0)
    assert not stop and count == 1
    count, stop = p._barge_check(_silence(), 16000, count)  # any quiet frame resets
    assert not stop and count == 0


def test_barge_off_when_disabled():
    p = _barge_pipeline(barge_in=False, armed=True)
    count = 0
    for _ in range(50):
        count, stop = p._barge_check(_speech(), 16000, count)
        assert not stop and count == 0


def test_default_config_disables_bargein():
    # FIX A: out of the box, barge-in is OFF so Aria never self-cuts on a laptop.
    from aria.config.schema import AriaConfig

    assert AriaConfig().vad.barge_in is False


def test_bargein_energy_gate_ignores_echo():
    # FIX A: opt-in barge-in must not trip on Aria's own voice (echo at her output
    # level) but must still trip on a clearly-louder user.
    p = _barge_pipeline(armed=True)
    p._output_rms = 0.3  # Aria is currently outputting at ~0.3 RMS

    echo = np.full(480, 0.3, dtype="float32")  # mic hears her at output level
    count = 0
    for _ in range(p._barge_frames + 3):
        count, stop = p._barge_check(echo, 16000, count)
    assert not stop  # echo gated out — never self-barges

    loud = np.full(480, 0.95, dtype="float32")  # user clearly louder than echo
    count = 0
    for _ in range(p._barge_frames):
        count, stop = p._barge_check(loud, 16000, count)
    assert stop  # genuine over-talk still stops her


async def test_bargein_off_plays_full_multisentence_answer():
    # FIX A: with barge-in disabled, a multi-sentence (multi-second) answer plays
    # in full and is never stopped — even though the mic keeps hearing her.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(30)]
    tts = FakeTTS()
    speaker = FakeSpeaker()
    pipeline = VoicePipeline(
        stt=FakeSTT("tell me a story"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=False),  # default off
        wake_cfg=WakeWordConfig(enabled=False),
        mic=FakeMic(frames),
        speaker=speaker,
    )

    async def respond(_t):
        yield "Sentence one. "
        yield "Sentence two. "
        yield "Sentence three."

    await pipeline.run(respond)
    joined = " ".join(tts.spoken)
    assert "Sentence one." in joined
    assert "Sentence two." in joined
    assert "Sentence three." in joined  # answer never cut off
    assert not speaker.interrupted


async def test_filler_then_answer_both_play_with_bargein(tmp_path):
    # FIX 1b: a tool turn that speaks the filler then the real answer must play
    # BOTH — the filler must not self-barge and suppress the answer.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(25)]
    tts = FakeTTS()
    pipeline = VoicePipeline(
        stt=FakeSTT("what's the news"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=True),  # barge-in ON
        wake_cfg=WakeWordConfig(enabled=False),
        mic=FakeMic(frames),
        speaker=FakeSpeaker(),
    )

    async def respond(_t):
        yield "Let me check. "  # the filler
        yield "The headline is peace breaks out."  # the real answer

    await pipeline.run(respond)
    joined = " ".join(tts.spoken)
    assert "Let me check." in joined
    assert "peace breaks out" in joined  # answer NOT suppressed
    assert pipeline.state is State.IDLE


async def test_stale_stop_does_not_mute_next_turn():
    # The cross-turn mute bug: a stop set on a prior turn must not silence the
    # next utterance. _end_capture must reset the speaker before playback.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(25)]
    speaker = FakeSpeaker()
    speaker.stop()  # simulate a leftover barge-in stop from a previous turn
    assert speaker.interrupted
    tts = FakeTTS()
    pipeline = VoicePipeline(
        stt=FakeSTT("what time is it"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=True),
        wake_cfg=WakeWordConfig(enabled=False),
        mic=FakeMic(frames),
        speaker=speaker,
    )

    async def respond(transcript: str):
        yield "Here you go."

    await pipeline.run(respond)
    assert tts.spoken, "a stale stop flag must not mute the turn"


async def test_announcement_spoken_when_idle():
    # A proactive announcement queued while idle is spoken via TTS.
    import asyncio

    q: asyncio.Queue[str] = asyncio.Queue()
    q.put_nowait("Hey — your laundry timer's up.")
    tts = FakeTTS()
    # All-silence frames: never a user turn, so the pipeline stays idle and drains.
    frames = [_silence() for _ in range(10)]
    pipeline = VoicePipeline(
        stt=FakeSTT(),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=True),
        wake_cfg=WakeWordConfig(enabled=True),  # wake word on -> silence won't start a turn
        announcements=q,
        mic=FakeMic(frames),
        speaker=FakeSpeaker(),
    )

    async def respond(_t):  # should never be called — no user turn here
        raise AssertionError("respond must not run for an announcement")
        yield ""  # pragma: no cover

    await pipeline.run(respond)
    assert tts.spoken == ["Hey — your laundry timer's up."]
    assert pipeline.state is State.IDLE


async def test_announcement_waits_until_turn_finishes():
    # An announcement queued mid-turn must NOT interrupt it; it speaks afterwards.
    import asyncio

    q: asyncio.Queue[str] = asyncio.Queue()
    tts = FakeTTS()
    # One real utterance, then idle silence for the announcement to drain into.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(25)]
    pipeline = VoicePipeline(
        stt=FakeSTT("what time is it"),
        tts=tts,
        vad=EnergyVAD(threshold=0.01),
        wakeword=NullWakeWord(),
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=True),
        wake_cfg=WakeWordConfig(enabled=False),  # always-listen: speech is a turn
        announcements=q,
        mic=FakeMic(frames),
        speaker=FakeSpeaker(),
    )

    async def respond(_t):
        # Queue the announcement DURING the turn; it must wait for idle.
        q.put_nowait("Reminder: standup.")
        yield "It is noon."

    await pipeline.run(respond)
    # Turn answer first, announcement second — both spoken, in order.
    assert tts.spoken == ["It is noon.", "Reminder: standup."]
    assert pipeline.state is State.IDLE
