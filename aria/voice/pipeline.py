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
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from enum import Enum, auto

import numpy as np

from aria.config.schema import (
    ActivationConfig,
    AudioConfig,
    ConversationConfig,
    VADConfig,
    WakeWordConfig,
)
from aria.voice.audio import Microphone, Speaker
from aria.voice.base import STT, TTS, VAD, AudioChunk, WakeWord
from aria.voice.chime import make_chime
from aria.voice.sentencizer import sentence_chunks

# A `respond` takes the transcript and yields assistant text deltas.
RespondFn = Callable[[str], AsyncIterator[str]]
# Decides whether a conversation-window capture was really meant for Aria (vs the
# TV / a side conversation). Only applied to NON-confirmation follow-up captures.
FollowupFilter = Callable[[str], Awaitable[bool]]

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

# When Aria finishes a turn that expects a reply (a pending confirmation), she
# re-opens the mic WITHOUT the wake word and listens for the user's answer for this
# long. If no one starts talking by then, she falls back to wake-word IDLE — the
# pending state is kept, so a later "hey jarvis, yes" still resumes. Kept short:
# past a few seconds an open mic feels like surveillance, not attentiveness.
_FOLLOWUP_WINDOW_S = 4.0

# How a capture began — drives whether the follow-up relevance filter applies.
# "wake" and "confirm" captures are always for Aria; "conversation" captures (the
# open mic after an ordinary answer) might be background speech, so they're gated.
_ORIGIN_WAKE = "wake"
_ORIGIN_CONFIRM = "confirm"
_ORIGIN_CONVERSATION = "conversation"
_ORIGIN_PTT = "ptt"  # push-to-talk: capture runs exactly while the key is held

# Minimum VOICED frames before a capture is transcribed at all. Whisper
# HALLUCINATES text ("Thank you.") on captures that are mostly silence, so a
# lone keyboard click or chair squeak that flips one VAD frame must never reach
# STT — nor may a false wake that's followed by 20s of pure silence. Wake and
# confirmation captures were explicitly user-initiated, so their bar is low;
# the always-open conversation window demands real sustained speech.
_MIN_VOICED_FRAMES = 3  # ~90ms of speech (wake / confirmation captures)
_MIN_VOICED_FRAMES_CONVO = 8  # ~240ms of speech (open-mic conversation window)


def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))


async def _one_frame(pcm: np.ndarray) -> AsyncIterator[np.ndarray]:
    yield pcm


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
        awaiting_reply: Callable[[], bool] | None = None,
        announcements: asyncio.Queue[str] | None = None,
        conversation_cfg: ConversationConfig | None = None,
        followup_filter: FollowupFilter | None = None,
        activation_cfg: ActivationConfig | None = None,
        hotkey=None,  # anything with a `pressed: bool` (see aria.voice.hotkey)
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
        # Returns True when the just-finished turn expects the user's reply, so we
        # re-open the mic without a wake word. Default: never (plain turn-taking).
        self._awaiting_reply = awaiting_reply or (lambda: False)
        # Conversation mode: when enabled, the mic re-opens after EVERY answer (not
        # only confirmations), so back-and-forth needs no wake word. None = off
        # (plain turn-taking, as before).
        self._conversation = conversation_cfg
        self._followup_filter = followup_filter
        # How the current/last capture started (wake word vs follow-up window).
        self._capture_origin = _ORIGIN_WAKE
        # Push-to-talk: `hotkey.pressed` is polled once per audio frame. While a
        # PTT capture runs, the key RELEASE (not silence) ends it.
        self._hotkey = hotkey
        self._ptt_hold = False
        # Wake-word spotting is off in pure hotkey mode (activation_cfg says so).
        self._wake_active = activation_cfg is None or activation_cfg.mode != "hotkey"
        # The "I'm listening" earcon, pre-rendered at the speaker's rate. None
        # (e.g. in tests without an activation config) = silent activation.
        self._chime_pcm = (
            make_chime(tts.sample_rate)
            if activation_cfg is not None and activation_cfg.chime
            else None
        )
        # Deadline (perf_counter) for follow-up listening; 0.0 when not in it.
        self._followup_deadline = 0.0
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
        voiced = 0
        barge = 0
        ptt_prev = False
        speak_task: asyncio.Task | None = None

        # The daemon RE-ENTERS this loop after a mid-turn crash (a network error
        # during STT, a dead speaker, …). The previous run may have died with the
        # machine in THINKING or SPEAKING — states this loop only ever LEAVES, so
        # without a reset every frame would be ignored forever: alive but deaf
        # until a reboot. Always start a run from a clean IDLE.
        self._ptt_hold = False
        self._followup_deadline = 0.0
        self._reset_idle()

        try:
            async for frame in self.mic.frames():
                ptt_now = self._ptt_pressed()
                ptt_edge = ptt_now and not ptt_prev
                ptt_prev = ptt_now

                if self.state is State.IDLE:
                    if ptt_edge:
                        # Push-to-talk: capture for exactly as long as the key is
                        # held; the release is the endpoint (no VAD guessing).
                        self._t_wake = time.perf_counter()
                        self._followup_deadline = 0.0
                        self._capture_origin = _ORIGIN_PTT
                        self._ptt_hold = True
                        self._set_state(State.LISTENING)
                        collected = [frame]
                        started = self.vad.is_speech(frame, sr)
                        trailing = 0
                        voiced = 1 if started else 0
                        await self._ack()
                    elif self._wake_active and self._wake_triggered(frame, sr):
                        self._t_wake = time.perf_counter()
                        self._followup_deadline = 0.0  # wake-driven, not a follow-up
                        self._capture_origin = _ORIGIN_WAKE
                        self._set_state(State.LISTENING)
                        collected = [frame]
                        started = self.vad.is_speech(frame, sr)
                        trailing = 0
                        voiced = 1 if started else 0
                        await self._ack()
                    elif (announcement := self._pop_announcement()) is not None:
                        # Proactive speech: only ever started while idle, so it
                        # can't collide with a user turn. Spoken via the same
                        # background-task path, so barge-in still applies.
                        self._begin_speaking()
                        speak_task = asyncio.create_task(self._play_text(announcement))
                        barge = 0

                elif self.state is State.LISTENING:
                    if ptt_now and not self._ptt_hold:
                        # Key pressed inside an open follow-up window: adopt PTT
                        # semantics — hold the mic for as long as the key is down.
                        self._ptt_hold = True
                        self._capture_origin = _ORIGIN_PTT
                        self._followup_deadline = 0.0
                    if self._ptt_hold:
                        collected.append(frame)
                        if self.vad.is_speech(frame, sr):
                            voiced += 1
                        if (not ptt_now) or len(collected) >= max_frames:
                            self._ptt_hold = False
                            speak_task = await self._end_capture(
                                collected, voiced, sr, respond
                            )
                            if speak_task is None:
                                self._reset_idle()
                            else:
                                barge = 0
                        continue
                    # Follow-up mode: if we re-opened the mic for a reply and the
                    # user hasn't started talking within the window, fall back to
                    # wake-word IDLE (the pending state is kept for "hey jarvis, yes").
                    if (
                        not started
                        and self._followup_deadline
                        and time.perf_counter() > self._followup_deadline
                    ):
                        self._followup_deadline = 0.0
                        collected = []
                        self._reset_idle()
                        continue
                    collected.append(frame)
                    if self.vad.is_speech(frame, sr):
                        started = True
                        trailing = 0
                        voiced += 1
                    elif started:
                        trailing += 1
                    if (started and trailing >= silence_frames) or len(collected) >= max_frames:
                        self._followup_deadline = 0.0
                        speak_task = await self._end_capture(collected, voiced, sr, respond)
                        if speak_task is None:
                            self._reset_idle()
                        else:
                            barge = 0

                elif self.state is State.SPEAKING:
                    if ptt_edge:
                        # The key always means "stop talking and listen to me":
                        # cut playback; the finished speak task re-opens the mic.
                        self.speaker.stop()
                    barge, should_stop = self._barge_check(frame, sr, barge)
                    if should_stop:
                        self.speaker.stop()
                    if speak_task is not None and speak_task.done():
                        if self._finish_speaking(speak_task):
                            # Re-armed into follow-up LISTENING: start a fresh capture
                            # so the user's answer isn't mixed with stale frames.
                            collected = []
                            started = False
                            trailing = 0
                            voiced = 0
                        speak_task = None
        finally:
            # Let any in-flight reply finish (so it isn't left suspended).
            if speak_task is not None and not speak_task.done():
                try:
                    await speak_task
                except BaseException:  # noqa: BLE001 - shutdown best-effort
                    pass

    def _ptt_pressed(self) -> bool:
        return bool(self._hotkey is not None and getattr(self._hotkey, "pressed", False))

    async def _ack(self) -> None:
        """Play the short 'I'm listening' chime (never fatal — it's cosmetic)."""
        if self._chime_pcm is None:
            return
        try:
            self.speaker.reset()
            await self.speaker.play(_one_frame(self._chime_pcm))
        except Exception:  # noqa: BLE001 - a broken speaker mustn't kill the loop
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
        self, collected: list[np.ndarray], voiced: int, sr: int, respond: RespondFn
    ) -> asyncio.Task | None:
        """Transcribe the captured utterance and, if non-empty, kick off speaking
        as a background task. Returns the speak task (or None to go back to idle)."""
        self._t_capture_end = time.perf_counter()
        self._set_state(State.THINKING)
        need = (
            _MIN_VOICED_FRAMES_CONVO
            if self._capture_origin == _ORIGIN_CONVERSATION
            else _MIN_VOICED_FRAMES
        )
        if voiced < need:
            return None  # essentially silence — Whisper would hallucinate a phrase
        pcm = np.concatenate(collected) if collected else np.zeros(0, "float32")
        if pcm.size < sr // 4:  # < 0.25s -> noise, ignore
            return None
        transcript = (await self.stt.transcribe(AudioChunk(pcm, sr))).strip()
        if not transcript:
            return None
        if not await self._meant_for_us(transcript):
            return None  # background speech in the open-mic window — ignore it
        self._on_transcript(transcript)
        self._begin_speaking()
        return asyncio.create_task(self._speak(respond, transcript))

    async def _meant_for_us(self, transcript: str) -> bool:
        """Gate conversation-window captures through the relevance filter. Wake-word
        and confirmation-reply captures are always accepted (they're explicit)."""
        if self._capture_origin != _ORIGIN_CONVERSATION or self._followup_filter is None:
            return True
        try:
            return await self._followup_filter(transcript)
        except Exception:  # noqa: BLE001 - fail open: never eat a real user turn
            return True

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

    def _finish_speaking(self, speak_task: asyncio.Task) -> bool:
        """Resolve a finished speak task. If the turn left Aria expecting a reply
        (a pending confirmation) — or conversation mode is on — re-open the mic
        immediately in follow-up LISTENING mode with no wake word and return True.
        Otherwise go IDLE and return False.

        TTS has fully finished by the time the speak task is done, so arming the mic
        here can't capture the tail of her own question (no acoustic echo)."""
        speak_task.result()  # re-raise a fatal reply error (e.g. auth) first
        if self._awaiting_reply():
            self._begin_followup_listen(_ORIGIN_CONFIRM, _FOLLOWUP_WINDOW_S)
            return True
        if self._conversation is not None and self._conversation.enabled:
            # Keep the conversation flowing: no wake word needed to continue. The
            # relevance filter protects against background speech being answered.
            self._begin_followup_listen(_ORIGIN_CONVERSATION, self._conversation.window_s)
            return True
        self._reset_idle()
        return False

    def _begin_followup_listen(self, origin: str, window_s: float) -> None:
        """Re-arm capture as if a wake word just fired, so the user can answer
        naturally. The run loop resets its capture buffers; we reset VAD/wake and
        timing (so wake→first-word latency stays meaningful) and start the window."""
        self.wakeword.reset()
        self.vad.reset()
        now = time.perf_counter()
        self._t_wake = now
        self._capture_origin = origin
        self._followup_deadline = now + window_s
        self._set_state(State.LISTENING)

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
        """Synthesize and play sentence chunks with one chunk of lookahead: while
        a chunk is playing, the NEXT one is already synthesizing in the background.
        Kokoro runs below real time on CPU, so the previous synth-then-play-then-
        synth serialization left an audible gap before every chunk."""
        sentences = sentence_chunks(deltas)
        queue: asyncio.Queue[list[np.ndarray] | None] = asyncio.Queue(maxsize=1)

        async def synth_ahead() -> None:
            try:
                async for sentence in sentences:
                    if self.speaker.interrupted:
                        break
                    frames = [f async for f in self.tts.synthesize(sentence)]
                    await queue.put(frames)
            finally:
                await queue.put(None)

        producer = asyncio.create_task(synth_ahead())
        report_latency = True
        try:
            while (frames := await queue.get()) is not None:
                if self.speaker.interrupted:
                    continue  # barge-in: drain remaining chunks unplayed
                await self.speaker.play(self._replay(frames, report_latency))
                report_latency = False
            await producer  # surface a synth error (missing voice etc.) to the turn
        finally:
            if not producer.done():
                producer.cancel()
                with suppress(asyncio.CancelledError):
                    await producer

    def _replay(self, frames: list[np.ndarray], report_latency: bool):
        async def _iter() -> AsyncIterator[np.ndarray]:
            for f in frames:
                yield f

        return self._track_audio(_iter(), report_latency)

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
