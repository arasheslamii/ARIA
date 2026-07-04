"""Assembly of the full proactive voice stack, shared by the interactive runner
and the background daemon.

A VoiceSession bundles the pipeline + orchestrator + scheduler + memory + the
announcement queue, and knows how to tear itself down. Both `aria` (foreground)
and `aria daemon` (headless) build one of these — only their callbacks and
error handling differ.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from aria.app import build_orchestrator, build_stt, build_tts
from aria.config.keyring import SecretStore
from aria.config.schema import AriaConfig
from aria.core.memory import Memory
from aria.core.orchestrator import Orchestrator
from aria.core.scheduler import SchedulerService, desktop_notify
from aria.voice.pipeline import State, VoicePipeline
from aria.voice.vad import make_vad
from aria.voice.wakeword import make_wakeword


@dataclass
class VoiceSession:
    pipeline: VoicePipeline
    orchestrator: Orchestrator
    scheduler: SchedulerService
    memory: Memory
    announcements: asyncio.Queue[str]
    managers: list = field(default_factory=list)

    async def aclose(self) -> None:
        await self.scheduler.stop()
        await self.memory.close()
        for m in self.managers:
            await m.aclose()


async def build_voice_session(
    config: AriaConfig,
    *,
    on_state: Callable[[State], None] | None = None,
    on_transcript: Callable[[str], None] | None = None,
    on_latency: Callable[[float, float], None] | None = None,
) -> VoiceSession:
    """Build the proactive voice stack. Starts the scheduler and pre-warms the
    models. Raises MissingSecret (no key) or FileNotFoundError (no voice model)
    after cleaning up anything it had already started."""
    secrets = SecretStore()
    memory = Memory()
    await memory.open()

    announcements: asyncio.Queue[str] = asyncio.Queue()
    user_name = await memory.recall("user_name")
    scheduler = SchedulerService(
        announce=announcements.put_nowait,
        notify=desktop_notify,
        name_provider=lambda: user_name,
    )
    await scheduler.start()

    managers: list = []
    try:
        orch, managers = await build_orchestrator(
            config, secrets, memory, scheduler=scheduler, voice=True
        )
        await orch.warm_up()
        stt = build_stt(config, secrets)
        tts = build_tts(config)
    except BaseException:
        # Roll back partial startup so a failure never leaks a db/loop.
        await scheduler.stop()
        await memory.close()
        for m in managers:
            await m.aclose()
        raise

    vad = make_vad(config.vad.backend, config.vad.speech_threshold)
    wake = make_wakeword(
        config.wakeword.enabled, config.wakeword.model, config.wakeword.threshold
    )

    # Hold-to-talk (activation mode hotkey/hybrid). If the key can't be watched
    # (no evdev / no input-group access), degrade to wake word and say why —
    # Aria must never come up deaf.
    import logging

    activation = config.activation
    hotkey = None
    if activation.mode in ("hotkey", "hybrid"):
        from aria.voice.hotkey import HotkeyListener

        listener = HotkeyListener(activation.hotkey)
        if await listener.start():
            hotkey = listener
            managers.append(listener)
        else:
            logging.getLogger("aria").warning(
                "Hold-to-talk unavailable (%s)%s.",
                listener.reason,
                " — falling back to wake-word activation"
                if activation.mode == "hotkey" else "",
            )
            if activation.mode == "hotkey":
                activation = activation.model_copy(update={"mode": "wake_word"})

    pipeline = VoicePipeline(
        stt=stt,
        tts=tts,
        vad=vad,
        wakeword=wake,
        audio_cfg=config.audio,
        vad_cfg=config.vad,
        wake_cfg=config.wakeword,
        on_state=on_state,
        on_transcript=on_transcript,
        on_latency=on_latency,
        # Hold the floor for the user's reply (e.g. a yes/no after a confirmation):
        # re-open the mic with no wake word while the orchestrator is awaiting one.
        awaiting_reply=lambda: orch.awaiting_reply,
        announcements=announcements,
        # Conversation mode: the mic re-opens after every answer; the orchestrator's
        # fast-model gate drops background speech so she never butts in uninvited.
        conversation_cfg=config.conversation,
        followup_filter=orch.accept_followup,
        activation_cfg=activation,
        hotkey=hotkey,
    )
    return VoiceSession(pipeline, orch, scheduler, memory, announcements, managers)
