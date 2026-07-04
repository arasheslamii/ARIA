"""Runtime entry points: the live voice loop and a text REPL.

Both share the same Orchestrator, so a turn behaves identically whether it came
from speech or the keyboard — which is exactly what the mockable smoke test
exercises.
"""

from __future__ import annotations

import asyncio
import sys

from rich.console import Console

from aria import APP_NAME
from aria.app import MissingSecret, build_orchestrator
from aria.config.keyring import SecretStore
from aria.config.schema import AriaConfig
from aria.core.memory import Memory
from aria.core.scheduler import SchedulerService, desktop_notify
from aria.core.session import build_voice_session
from aria.llm.base import LLMAuthError, LLMConnectionError, LLMRateLimitError
from aria.voice.pipeline import State

console = Console()

_BAD_KEY_MSG = (
    "Your Groq API key looks invalid or missing — run `aria setup` to fix it."
)
_OFFLINE_MSG = "I couldn't reach Groq — check your internet connection and try again."
_RATE_LIMIT_MSG = (
    "I've hit my usage limit for the moment — let's try again in a few minutes."
)


def friendly_error(exc: BaseException) -> str | None:
    """Map an exception to a one-line user message, or None if it isn't one we
    handle gracefully (caller should then let it propagate)."""
    if isinstance(exc, LLMRateLimitError):
        return _RATE_LIMIT_MSG
    if isinstance(exc, (LLMAuthError, MissingSecret)):
        return _BAD_KEY_MSG
    if isinstance(exc, LLMConnectionError):
        return _OFFLINE_MSG
    return None

_STATE_HINT = {
    State.IDLE: "[dim]· waiting for wake word …[/dim]",
    State.LISTENING: "[bold green]● listening[/bold green]",
    State.THINKING: "[yellow]… thinking[/yellow]",
    State.SPEAKING: f"[cyan]🔊 {APP_NAME} speaking[/cyan]",
}


async def run_voice(config: AriaConfig) -> None:
    try:
        session = await build_voice_session(
            config,
            on_state=lambda s: console.print(_STATE_HINT.get(s, ""), end="\r"),
            on_transcript=lambda t: console.print(f"\n[bold]You:[/bold] {t}"),
            on_latency=lambda resp, wake: console.print(
                f"[dim]⚡ {resp:.2f}s to first word "
                f"(end-of-speech→speak){'  ✓' if resp < 1.2 else '  ⚠ over 1.2s'}"
                f" · {wake:.2f}s from wake[/dim]"
            ),
        )
    except MissingSecret:
        console.print(f"[red]{_BAD_KEY_MSG}[/red]")
        return
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]\nRun [bold]aria setup[/bold] to fetch a voice.")
        return

    wake_hint = f"Say “{config.wakeword.model.replace('_', ' ')}”" if config.wakeword.enabled \
        else "Just start talking"
    console.print(f"[bold cyan]{APP_NAME} is listening.[/bold cyan] {wake_hint}. Ctrl-C to quit.")
    # Risky actions are confirmed in-band: the orchestrator asks aloud and the
    # user's next utterance ("yes"/"no") is interpreted as the answer.
    try:
        await session.pipeline.run(session.orchestrator.respond)
    except KeyboardInterrupt:
        pass
    except BaseException as exc:  # noqa: BLE001
        msg = friendly_error(exc)
        if msg is None:
            raise
        console.print(f"\n[red]{msg}[/red]")
    finally:
        await session.aclose()


async def run_text(config: AriaConfig) -> None:
    """Keyboard REPL — same brain, no audio. Great for dev and demos."""
    secrets = SecretStore()
    memory = Memory()
    await memory.open()
    # Timers/reminders fire to the console here (no voice in text mode).
    scheduler = SchedulerService(
        announce=lambda text: console.print(f"\n[magenta]🔔 {text}[/magenta]"),
        notify=desktop_notify,
        name_provider=lambda: None,
    )
    await scheduler.start()
    try:
        orch, managers = await build_orchestrator(config, secrets, memory, scheduler=scheduler)
    except MissingSecret:
        console.print(f"[red]{_BAD_KEY_MSG}[/red]")
        await scheduler.stop()
        await memory.close()
        return
    await orch.warm_up()

    console.print(f"[bold cyan]{APP_NAME}[/bold cyan] text mode. Type 'exit' to quit.\n")
    try:
        while True:
            try:
                line = await asyncio.to_thread(input, "You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if line.strip().lower() in {"exit", "quit"}:
                break
            if not line.strip():
                continue
            console.print(f"[cyan]{APP_NAME}:[/cyan] ", end="")
            try:
                async for delta in orch.respond(line):
                    console.print(delta, end="")
                    sys.stdout.flush()
                console.print()
            except BaseException as exc:  # noqa: BLE001
                msg = friendly_error(exc)
                if msg is None:
                    raise
                console.print(f"\n[red]{msg}[/red]")
    finally:
        await scheduler.stop()
        await memory.close()
        for m in managers:
            await m.aclose()
