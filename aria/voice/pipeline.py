"""The streaming voice loop: wake -> listen -> think -> speak, with barge-in.

State machine per turn:
  IDLE      : wake word spotting (or always-listen if disabled)
  LISTENING : VAD-endpointed capture until trailing silence
  THINKING  : STT -> orchestrator (handled by the caller's `respond` coroutine)
  SPEAKING  : stream sentences to TTS; concurrent VAD watches for barge-in

The pipeline owns audio + endpointing only. It calls back into a `respond`
coroutine that yields text deltas, keeping the LLM/agent logic decoupled.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from enum import Enum, auto

import numpy as np

from aria.config.schema import AudioConfig, VADConfig, WakeWordConfig
from aria.voice.audio import Microphone, Speaker
from aria.voice.base import STT, TTS, VAD, AudioChunk, WakeWord
from aria.voice.sentencizer import sentence_chunks

# A `respond` takes the transcript and yields assistant text deltas.
RespondFn = Callable[[str], AsyncIterator[str]]

# Barge-in requires this much CONTINUOUS over-speech before stopping playback.
# Generous (vs ~90ms) because X11 has no acoustic echo cancellation, so Aria's
# own voice bleeds into the mic; we'd rather under-trigger than cut her off.
_BARGE_MIN_MS = 280

# Grace period: don't arm barge-in until the CURRENT contiguous audio burst has
# been playing this long. Without acoustic echo cancellation Aria's own voice
# leaks into the mic; this keeps a short utterance (e.g. the "Let me check."
# filler) from self-triggering barge-in and muting the answer that follows.
_BARGE_GRACE_S = 0.9
# A gap in played audio longer than this starts a NEW burst (so the grace resets
# for the real answer after the silent tool round-trip following the filler).
_AUDIO_GAP_S = 0.4

# Energy gate for opt-in barge-in: a mic frame only counts as the user talking
# over Aria if its level clearly exceeds her current output (echo) by this factor,
# plus a small absolute floor for when she's near-silent.
_ECHO_FACTOR = 2.5
_ECHO_FLOOR = 0.02


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class VoicePipeline:
    def __init__(
        self,
        *,
        stt: STT,
        tts: TTS,
        vad: VAD,
        wakeword: WakeWord,
        audio_cfg: AudioConfig,
        vad_cfg: VADConfig,
        wake_cfg: WakeWordConfig,
        on_state: Callable[[State], None] | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_latency: Callable[[float, float], None] | None = None,
        announcements: asyncio.Queue[str] | None = None,
        mic: Microphone | None = None,
        speaker: Speaker | None = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.vad = vad
        self.wakeword = wakeword
        self.audio_cfg = audio_cfg
        self.vad_cfg = vad_cfg
        self.wake_cfg = wake_cfg
        self._on_state = on_state or (lambda _s: None)
        self._on_transcript = on_transcript or (lambda _t: None)
        # on_latency(end_of_speech_to_first_word, wake_to_first_word) in seconds.
        self._on_latency = on_latency or (lambda _a, _b: None)
        # Proactive-speech channel: the scheduler (and later briefings) push text
        # here; we speak it ONLY when idle so it never collides with a user turn.
        self._announcements = announcements
        self._t_wake = 0.0
        self._t_capture_end = 0.0
        # mic/speaker are injectable so the loop is testable without PortAudio.
        self.mic = mic or Microphone(
            audio_cfg.sample_rate, audio_cfg.block_ms, audio_cfg.input_device
        )
        self.speaker = speaker or Speaker(tts.sample_rate, audio_cfg.output_device)
        self.state = State.IDLE
        # Barge-in arming is time-based: track the start of the current contiguous
        # audio burst and the last played frame. Far-negative => no audio playing.
        self._burst_start = 0.0
        self._last_frame_ts = -1e9
        self._barge_frames = max(1, round(_BARGE_MIN_MS / audio_cfg.block_ms))
        # Smoothed amplitude of the TTS audio Aria is currently playing; the
        # opt-in barge-in energy gate compares mic level against this (echo).
        self._output_rms = 0.0

    def _set_state(self, state: State) -> None:
        self.state = state
        self._on_state(state)

    async def run(self, respond: RespondFn) -> None:
        """Main loop. The SOLE consumer of ``self.mic.frames()``.

        Every frame is dispatched by state. While SPEAKING, the reply is produced
        by a background task that never touches the mic, and this loop feeds each
        frame to the barge-in detector — so there is only ever one iterator on the
        microphone generator (fixes "async generator is already running").
        """
        sr = self.audio_cfg.sample_rate
        silence_frames = max(1, int(self.vad_cfg.silence_ms / self.audio_cfg.block_ms))
        max_frames = max(1, 20_000 // self.audio_cfg.block_ms)  # ~20s utterance cap

        collected: list[np.ndarray] = []
        started = False
        trailing = 0
        barge = 0
        speak_task: asyncio.Task | None = None

        try:
            async for frame in self.mic.frames():
                if self.state is State.IDLE:
                    if self._wake_triggered(frame, sr):
                        self._t_wake = time.perf_counter()
                        self._set_state(State.LISTENING)
                        collected = [frame]
                        started = self.vad.is_speech(frame, sr)
                        trailing = 0
                    elif (announcement := self._pop_announcement()) is not None:
                        # Proactive speech: only ever started while idle, so it
                        # can't collide with a user turn. Spoken via the same
                        # background-task path, so barge-in still applies.
                        self._begin_speaking()
                        speak_task = asyncio.create_task(self._play_text(announcement))
                        barge = 0

                elif self.state is State.LISTENING:
                    collected.append(frame)
                    if self.vad.is_speech(frame, sr):
                        started = True
                        trailing = 0
                    elif started:
                        trailing += 1
                    if (started and trailing >= silence_frames) or len(collected) >= max_frames:
                        speak_task = await self._end_capture(collected, sr, respond)
                        if speak_task is None:
                            self._reset_idle()
                        else:
                            barge = 0

                elif self.state is State.SPEAKING:
                    barge, should_stop = self._barge_check(frame, sr, barge)
                    if should_stop:
                        self.speaker.stop()
                    if speak_task is not None and speak_task.done():
                        self._finish_speaking(speak_task)
                        speak_task = None
        finally:
            # Let any in-flight reply finish (so it isn't left suspended).
            if speak_task is not None and not speak_task.done():
                try:
                    await speak_task
                except BaseException:  # noqa: BLE001 - shutdown best-effort
                    pass

    def _wake_triggered(self, frame: np.ndarray, sr: int) -> bool:
        if not self.wake_cfg.enabled:
            # No wake word: trip on first speech (always-listen mode).
            return self.vad.is_speech(frame, sr)
        return self.wakeword.process(frame, sr) >= self.wake_cfg.threshold

    def _barge_armed(self, now: float) -> bool:
        """Barge-in is armed only while audio is actively playing AND the current
        burst has been going longer than the grace period — so the silent THINKING
        window, the gap before the answer, and short self-spoken fillers can't trip
        it (Aria's own voice would otherwise self-barge with no echo cancellation)."""
        playing = (now - self._last_frame_ts) < _AUDIO_GAP_S
        return playing and (now - self._burst_start) >= _BARGE_GRACE_S

    def _barge_check(self, frame: np.ndarray, sr: int, count: int) -> tuple[int, bool]:
        """Decide barge-in for one SPEAKING-state frame.

        Returns (new_count, should_stop). Requires ``_barge_frames`` of *continuous*
        over-speech once armed; the counter resets on any frame that isn't clearly
        the user. A frame only counts if it is detected speech AND its energy
        clearly exceeds Aria's current output level — so her own voice bleeding
        into the mic (echo) doesn't self-interrupt her.
        """
        if (
            self.vad_cfg.barge_in
            and self._barge_armed(time.perf_counter())
            and self.vad.is_speech(frame, sr)
            and _rms(frame) > _ECHO_FACTOR * self._output_rms + _ECHO_FLOOR
        ):
            count += 1
            return count, count >= self._barge_frames
        return 0, False

    def _reset_idle(self) -> None:
        self.wakeword.reset()
        self.vad.reset()
        self._set_state(State.IDLE)

    async def _end_capture(
        self, collected: list[np.ndarray], sr: int, respond: RespondFn
    ) -> asyncio.Task | None:
        """Transcribe the captured utterance and, if non-empty, kick off speaking
        as a background task. Returns the speak task (or None to go back to idle)."""
        self._t_capture_end = time.perf_counter()
        self._set_state(State.THINKING)
        pcm = np.concatenate(collected) if collected else np.zeros(0, "float32")
        if pcm.size < sr // 4:  # < 0.25s -> noise, ignore
            return None
        transcript = (await self.stt.transcribe(AudioChunk(pcm, sr))).strip()
        if not transcript:
            return None
        self._on_transcript(transcript)
        self._begin_speaking()
        return asyncio.create_task(self._speak(respond, transcript))

    def _begin_speaking(self) -> None:
        """Enter SPEAKING for a fresh utterance: clear any stale barge-in stop
        (else this utterance is muted) and disarm barge-in until audio is audible."""
        self.speaker.reset()
        self._last_frame_ts = -1e9  # nothing playing yet -> barge disarmed
        self._set_state(State.SPEAKING)

    def _pop_announcement(self) -> str | None:
        if self._announcements is None or self._announcements.empty():
            return None
        try:
            return self._announcements.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _finish_speaking(self, speak_task: asyncio.Task) -> None:
        self._reset_idle()
        speak_task.result()  # re-raise a fatal reply error (e.g. auth) to the runtime

    async def _speak(self, respond: RespondFn, transcript: str) -> None:
        """Produce the reply and stream it to TTS. Never reads the microphone;
        barge-in is handled by the run loop via :meth:`Speaker.stop`."""
        await self._stream_to_tts(respond(transcript))

    async def _play_text(self, text: str) -> None:
        """Speak a fixed string (a proactive announcement) through the TTS path."""

        async def _one() -> AsyncIterator[str]:
            yield text

        await self._stream_to_tts(_one())

    async def _stream_to_tts(self, deltas: AsyncIterator[str]) -> None:
        sentences = sentence_chunks(deltas)
        report_latency = True
        async for sentence in sentences:
            if self.speaker.interrupted:
                break
            frames_out = self._track_audio(self.tts.synthesize(sentence), report_latency)
            report_latency = False
            await self.speaker.play(frames_out)

    async def _track_audio(
        self, frames: AsyncIterator[np.ndarray], report_latency: bool
    ) -> AsyncIterator[np.ndarray]:
        """Pass frames through, tracking playback timing for barge-in arming and
        reporting first-word latency. A gap before the first frame starts a fresh
        audio burst, which resets the barge-in grace period."""
        reported = not report_latency
        async for frame in frames:
            now = time.perf_counter()
            if now - self._last_frame_ts > _AUDIO_GAP_S:
                self._burst_start = now  # new contiguous burst -> grace resets
                self._output_rms = 0.0
            self._last_frame_ts = now
            # Track Aria's current output level (smoothed) for the echo gate.
            self._output_rms = max(0.5 * self._output_rms, _rms(frame))
            if not reported:
                self._on_latency(now - self._t_capture_end, now - self._t_wake)
                reported = True
            yield frame
