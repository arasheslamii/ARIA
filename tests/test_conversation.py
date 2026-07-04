"""Conversation mode (0.4.0): the mic re-opens after EVERY answer (no wake word),
background speech is filtered, confirmations understand amendments, spoken lines
vary, and the conversation is remembered beyond the trimmed window."""

from __future__ import annotations

import numpy as np
import pytest

from aria.config.schema import (
    AudioConfig,
    ConversationConfig,
    VADConfig,
    WakeWordConfig,
)
from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.llm.base import ChatResult, ToolCall, assistant, user
from aria.tools.base import Tool, ToolRegistry, ToolResult
from aria.voice.pipeline import VoicePipeline
from aria.voice.vad import EnergyVAD
from tests.conftest import FakeLLM, FakeMic, FakeSpeaker, FakeSTT, FakeTTS
from tests.test_followup import OnceWake
from tests.test_voice_loop import _silence, _speech


# --- pipeline: conversation window ----------------------------------------
def _conv_pipeline(
    mic_frames, *, followup_filter=None, stt_text="and tomorrow?", stt=None,
    tts=None, wakeword=None,
):
    pipeline = VoicePipeline(
        stt=stt or FakeSTT(stt_text),
        tts=tts or FakeTTS(),
        vad=EnergyVAD(threshold=0.01),
        wakeword=wakeword or OnceWake(),  # fires once -> turn 2 must be a follow-up
        audio_cfg=AudioConfig(sample_rate=16000, block_ms=30),
        vad_cfg=VADConfig(silence_ms=300, barge_in=False),
        wake_cfg=WakeWordConfig(enabled=True),
        conversation_cfg=ConversationConfig(enabled=True, window_s=6.0),
        followup_filter=followup_filter,
        mic=FakeMic(mic_frames),
        speaker=FakeSpeaker(),
    )
    return pipeline


async def test_conversation_mode_reopens_mic_after_any_answer():
    # No pending confirmation anywhere — yet turn 2 is heard without a wake word.
    frames = ([_speech() for _ in range(12)] + [_silence() for _ in range(25)]) * 2
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "It's sunny today."

    await _conv_pipeline(frames).run(respond)
    assert len(transcripts) == 2  # follow-up captured with NO second wake word


async def test_background_speech_is_dropped_by_filter():
    frames = ([_speech() for _ in range(12)] + [_silence() for _ in range(25)]) * 2
    transcripts: list[str] = []
    filtered: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "It's sunny today."

    async def reject(text: str) -> bool:
        filtered.append(text)
        return False  # everything in the open-mic window is "the TV"

    await _conv_pipeline(frames, followup_filter=reject).run(respond)
    assert len(transcripts) == 1  # only the wake-word turn was answered
    assert filtered  # the second capture went through the filter and was dropped


async def test_wake_word_capture_bypasses_the_filter():
    # The filter only guards the open-mic window — an explicit wake is always heard.
    frames = [_speech() for _ in range(5)] + [_silence() for _ in range(25)]
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "Hello!"

    async def reject(_text: str) -> bool:
        return False

    await _conv_pipeline(frames, followup_filter=reject).run(respond)
    assert len(transcripts) == 1  # wake-initiated capture was NOT filtered


async def test_filter_failure_fails_open():
    frames = ([_speech() for _ in range(12)] + [_silence() for _ in range(25)]) * 2
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "Sure."

    async def broken(_text: str) -> bool:
        raise RuntimeError("filter LLM down")

    await _conv_pipeline(frames, followup_filter=broken).run(respond)
    assert len(transcripts) == 2  # a real user turn is never eaten by a dead filter


# --- orchestrator: the follow-up relevance gate ----------------------------
async def _bare_orch(llm, tmp_path, registry=None):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    orch = Orchestrator(
        llm=llm, registry=registry or ToolRegistry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    return orch, mem


async def test_accept_followup_reads_fast_model_verdict(tmp_path):
    orch, mem = await _bare_orch(FakeLLM(chat_queue=[ChatResult(content="yes")]), tmp_path)
    assert await orch.accept_followup("and what about tomorrow?") is True

    orch2, mem2 = await _bare_orch(FakeLLM(chat_queue=[ChatResult(content="No.")]), tmp_path)
    assert await orch2.accept_followup("leave it on channel four") is False
    await mem.close()
    await mem2.close()


async def test_accept_followup_obvious_commands_skip_the_llm(tmp_path):
    class Boom:
        async def chat(self, *a, **k):
            raise AssertionError("lexical tier must not call the LLM")

        async def stream(self, *a, **k):
            yield ""

    orch, mem = await _bare_orch(Boom(), tmp_path)
    for utt in ("and what about tomorrow?", "turn the volume down please",
                "remind me to call the dentist", "thanks, that was great",
                "no, I already told you twice", "set a timer for the pasta"):
        assert await orch.accept_followup(utt) is True, utt
    await mem.close()


async def test_accept_followup_fails_open_and_accepts_confirmation_replies(tmp_path):
    class Boom:
        async def chat(self, *a, **k):
            raise RuntimeError("down")

        async def stream(self, *a, **k):
            yield ""

    orch, mem = await _bare_orch(Boom(), tmp_path)
    assert await orch.accept_followup("hello?") is True  # fail open

    orch._pending = object()  # a pending confirmation: reply always accepted
    assert await orch.accept_followup("yes") is True
    orch._pending = None
    await mem.close()


# --- orchestrator: confirmation amendments ---------------------------------
class SpyEmail(Tool):
    name = "send_email"
    description = "Send an email."
    risk = "confirm"

    def __init__(self) -> None:
        self.ran_with: dict | None = None

    async def run(self, **kwargs):
        self.ran_with = kwargs
        return ToolResult(content="sent")


def _route(kind: str) -> ChatResult:
    return ChatResult(content=f'{{"route":"{kind}","needs_tools":[],"reason":"x"}}')


async def test_amendment_replans_instead_of_forcing_yes_no(tmp_path):
    reg = ToolRegistry()
    tool = SpyEmail()
    reg.register(tool)
    chat_queue = [
        _route("agentic"),
        ChatResult(content="", tool_calls=[ToolCall("c1", "send_email", {"to": "bob"})]),
        # The classify call for the amendment reply:
        ChatResult(content="change"),
        # The re-planned tool call, which must be re-confirmed:
        ChatResult(content="", tool_calls=[ToolCall("c2", "send_email", {"to": "alice"})]),
    ]
    llm = FakeLLM(stream_text="Done.", chat_queue=chat_queue)
    orch, mem = await _bare_orch(llm, tmp_path, registry=reg)

    first = "".join([d async for d in orch.respond("email bob")])
    assert "go ahead" in first.lower() and tool.ran_with is None

    second = "".join([d async for d in orch.respond("actually send it to alice instead")])
    assert tool.ran_with is None  # STILL nothing executed — re-confirmation required
    assert orch._pending is not None
    assert orch._pending.calls[0].arguments == {"to": "alice"}  # the amended plan
    assert "alice" in second.lower()  # the new read-back names the change

    third = "".join([d async for d in orch.respond("yes")])
    assert tool.ran_with == {"to": "alice"}  # ran only after the fresh yes
    assert "done" in third.lower()
    await mem.close()


async def test_confirmation_classifier_fails_closed(tmp_path):
    # If the fast model is down, an unclear reply must re-ask — never execute.
    reg = ToolRegistry()
    tool = SpyEmail()
    reg.register(tool)

    class FlakyLLM(FakeLLM):
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if not self.chat_queue:
                raise RuntimeError("down")
            return await super().chat(
                messages, model=model, tools=tools,
                temperature=temperature, max_tokens=max_tokens,
            )

    llm = FlakyLLM(chat_queue=[
        _route("agentic"),
        ChatResult(content="", tool_calls=[ToolCall("c1", "send_email", {"to": "bob"})]),
    ])
    orch, mem = await _bare_orch(llm, tmp_path, registry=reg)
    await _drain(orch.respond("email bob"))
    reply = "".join([d async for d in orch.respond("mumble mumble something unclear")])
    assert "yes or no" in reply.lower()  # re-asked
    assert tool.ran_with is None and orch._pending is not None
    await mem.close()


# --- orchestrator: conversation memory --------------------------------------
async def test_trimmed_history_folds_into_running_summary(tmp_path):
    orch, mem = await _bare_orch(FakeLLM(stream_text="They discussed pizza plans."), tmp_path)
    for i in range(9):  # overflow the 6-exchange window
        orch._history.append(user(f"user turn {i}"))
        orch._history.append(assistant(f"assistant turn {i}"))
    orch._trim_history()
    assert orch._absorb_task is not None
    await orch._absorb_task
    assert orch._summary  # older turns live on as a summary…
    sys_msg = (await orch._base_messages())[0].content
    assert orch._summary in sys_msg  # …and the model actually sees it
    await mem.close()


async def test_warmup_recalls_previous_session(tmp_path):
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    await mem.log_turn("user", "remind me about mum's birthday on friday")
    await mem.log_turn("assistant", "Will do — Friday it is.")
    llm = FakeLLM(stream_text="Talked about mum's birthday reminder for Friday.")
    orch = Orchestrator(
        llm=llm, registry=ToolRegistry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    await orch.warm_up()
    assert "birthday" in orch._prev_session_note.lower()
    sys_msg = (await orch._base_messages())[0].content
    assert "previous conversation" in sys_msg
    await mem.close()


# --- varied lines ------------------------------------------------------------
def test_line_pools_never_repeat_back_to_back():
    from aria.core import lines

    picks = [lines.pick(lines.FILLERS) for _ in range(30)]
    assert all(a != b for a, b in zip(picks, picks[1:], strict=False))
    assert all(p in lines.FILLERS for p in picks)


def test_confirm_frames_keep_action_verbatim_and_ask():
    from aria.core import lines

    action = "send an email to bob@example.com saying 'lunch at 12'"
    for frame in lines.CONFIRM_FRAMES:
        q = frame.format(action=action)
        assert action in q  # the risky detail is NEVER paraphrased away
        assert "?" in q  # and it clearly asks


# --- sentencizer: coalesced prosody chunks ----------------------------------
async def test_sentencizer_coalesces_after_first_sentence():
    from aria.voice.sentencizer import sentence_chunks

    async def deltas():
        text = "Here is the first sentence. Then two. More tiny. Bits follow. The end."
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    out = [c async for c in sentence_chunks(deltas())]
    assert out[0] == "Here is the first sentence."  # flushed alone, fast
    assert len(out) == 2  # everything after is coalesced into one smooth chunk
    assert out[1] == "Then two. More tiny. Bits follow. The end."


# --- role-echo stripping (local Ollama templates) ---------------------------
async def _stream_of(*chunks: str):
    for c in chunks:
        yield c


async def test_role_echo_is_stripped_from_stream_head():
    from aria.llm.openai_compat import strip_role_echo

    out = "".join([d async for d in strip_role_echo(
        _stream_of("assist", "ant", "\n\nHi", " there!")
    )])
    assert out == "Hi there!"


async def test_normal_stream_head_passes_through_untouched():
    from aria.llm.openai_compat import strip_role_echo

    for chunks in (
        ("Hi", " there,", " friend."),
        ("Assistants", " like me help."),  # legit word, no separator -> kept
        (" As", " promised, done.",),
        ("ok",),  # tiny stream, ends while deciding
    ):
        out = "".join([d async for d in strip_role_echo(_stream_of(*chunks))])
        assert out == "".join(chunks)


# --- kokoro wiring ------------------------------------------------------------
def test_voice_catalog_lists_kokoro_first_and_flags_engine():
    from aria.tui.voices import KOKORO_VOICES, VOICES, is_kokoro, voice_catalog

    catalog = list(voice_catalog())
    assert catalog[: len(KOKORO_VOICES)] == list(KOKORO_VOICES)  # most natural first
    assert all(is_kokoro(v) for v in KOKORO_VOICES)
    assert not any(is_kokoro(v) for v in VOICES)


def test_build_tts_kokoro_missing_gives_clear_error(tmp_path, monkeypatch):
    # No Kokoro files anywhere and no Piper fallback -> a FileNotFoundError that
    # names the fix, not an obscure crash.
    from aria import app
    from aria.config.schema import AriaConfig

    monkeypatch.setenv("ARIA_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        "aria.config.loader.state_dir", lambda: tmp_path
    )
    monkeypatch.setattr(app, "resolve_piper_model", lambda cfg: tmp_path / "missing.onnx")
    cfg = AriaConfig()
    cfg.tts.provider = "kokoro"
    cfg.tts.voice = "af_heart"
    with pytest.raises(FileNotFoundError) as exc:
        app.build_tts(cfg)
    msg = str(exc.value).lower()
    assert "kokoro" in msg


async def test_kokoro_tts_slices_frames(monkeypatch, tmp_path):
    # KokoroTTS conforms to the TTS interface using a stubbed kokoro_onnx module.
    import sys
    import types

    class _FakeKokoro:
        def __init__(self, model, voices):
            pass

        def create(self, text, voice, speed, lang):
            return np.zeros(10_000, dtype="float32"), 24_000

    fake_mod = types.ModuleType("kokoro_onnx")
    fake_mod.Kokoro = _FakeKokoro
    monkeypatch.setitem(sys.modules, "kokoro_onnx", fake_mod)

    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"x")
    voices.write_bytes(b"x")

    from aria.voice.tts_kokoro import KokoroTTS

    tts = KokoroTTS(model, voices, voice="af_heart")
    frames = [f async for f in tts.synthesize("Hello there.")]
    assert len(frames) >= 2  # sliced for barge-in responsiveness
    assert sum(f.size for f in frames) == 10_000
    assert all(f.dtype == np.float32 for f in frames)


async def _drain(aiter):
    return [x async for x in aiter]


# --- ghost-turn hardening (0.4.1) -------------------------------------------
# Whisper hallucinates phrases ("Thank you.") on near-silent captures, which made
# Aria answer typing noise and false wakes, then re-open the mic and keep going.
class CountingSTT(FakeSTT):
    def __init__(self, text: str = "Thank you.") -> None:
        super().__init__(text)
        self.calls = 0

    async def transcribe(self, audio, *, language=None) -> str:
        self.calls += 1
        return self.text


class FirstFrameWake(OnceWake):
    """Fires on the very first frame even if it is silence — a FALSE wake."""

    def process(self, frame: np.ndarray, sample_rate: int) -> float:
        if self._used:
            return 0.0
        self._used = True
        return 1.0


async def test_typing_click_in_conversation_window_never_reaches_stt():
    # Turn 1 is real speech; the conversation window then catches a keyboard
    # click (2 voiced frames) — far below the sustained-speech minimum, so the
    # capture must be dropped BEFORE STT (which would hallucinate a phrase).
    frames = (
        [_speech() for _ in range(12)] + [_silence() for _ in range(25)]
        + [_speech() for _ in range(2)] + [_silence() for _ in range(25)]
    )
    stt = CountingSTT("and tomorrow?")
    transcripts: list[str] = []

    async def respond(transcript: str):
        transcripts.append(transcript)
        yield "Sunny."

    await _conv_pipeline(frames, stt=stt).run(respond)
    assert transcripts == ["and tomorrow?"]  # only the real turn
    assert stt.calls == 1  # the click capture never even hit the network


async def test_false_wake_with_no_speech_never_reaches_stt():
    # A spurious wake-word hit followed by silence used to capture the full 20s
    # cap and ship pure silence to Whisper -> guaranteed hallucinated turn.
    frames = [_silence() for _ in range(680)]  # > the ~667-frame utterance cap
    stt = CountingSTT()
    spoken: list[str] = []

    async def respond(transcript: str):
        spoken.append(transcript)
        yield "hello?"

    await _conv_pipeline(frames, stt=stt, wakeword=FirstFrameWake()).run(respond)
    assert stt.calls == 0
    assert spoken == []


async def test_stt_ghost_phrases_are_rejected_without_the_llm(tmp_path):
    class Boom:
        async def chat(self, *a, **k):
            raise AssertionError("ghost rejection must not call the LLM")

        async def stream(self, *a, **k):
            yield ""

    orch, mem = await _bare_orch(Boom(), tmp_path)
    for utt in ("Thank you.", "thanks", "Thanks for watching!", "Bye.", "you",
                "Okay.", "THANK YOU SO MUCH FOR WATCHING"):
        assert await orch.accept_followup(utt) is False, utt
    # A real request that merely OPENS with thanks still fast-accepts…
    assert await orch.accept_followup("thanks, now set a timer for ten minutes") is True
    # …and a pending confirmation reply is accepted before any ghost check.
    orch._pending = object()
    assert await orch.accept_followup("Thank you.") is True
    orch._pending = None
    await mem.close()


# --- latency: first-clause flush + synth/playback overlap (0.4.1) ----------
async def test_sentencizer_flushes_first_clause_of_a_long_opening_sentence():
    from aria.voice.sentencizer import sentence_chunks

    text = (
        "Well, considering everything you told me about the trip so far, "
        "I would honestly recommend the earlier train. It is quieter."
    )

    async def deltas():
        for i in range(0, len(text), 7):
            yield text[i : i + 7]

    out = [c async for c in sentence_chunks(deltas())]
    # The voice starts at the clause break instead of waiting out the whole
    # 100+ char opening sentence (Kokoro synthesizes slower than real time).
    assert out[0] == "Well, considering everything you told me about the trip so far,"
    assert len(out) == 2
    assert out[1] == "I would honestly recommend the earlier train. It is quieter."


async def test_tts_lookahead_plays_all_chunks_in_order():
    tts = FakeTTS()
    pipeline = _conv_pipeline([], tts=tts)

    async def deltas():
        yield "First sentence here. Second one follows. "
        yield "And a third to finish."

    await pipeline._stream_to_tts(deltas())
    assert tts.spoken[0] == "First sentence here."
    assert len(tts.spoken) == 2  # remainder coalesced, nothing dropped
    assert len(pipeline.speaker.played) == len(tts.spoken)  # every chunk audible


async def test_tts_error_propagates_through_lookahead():
    class BoomTTS(FakeTTS):
        async def synthesize(self, text):
            raise RuntimeError("voice model missing")
            yield  # pragma: no cover - marks this as an async generator

    pipeline = _conv_pipeline([], tts=BoomTTS())

    async def deltas():
        yield "Hello there, friend."

    with pytest.raises(RuntimeError, match="voice model missing"):
        await pipeline._stream_to_tts(deltas())
