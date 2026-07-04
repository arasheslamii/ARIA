"""Headless background daemon: the same voice loop + scheduler with no terminal.

Run as `aria daemon` (used by the systemd user service). It:
  * logs to a rotating file in the state dir AND to stdout (journald),
  * holds a single-instance lock so a second copy refuses to start,
  * shuts down cleanly on SIGTERM/SIGINT (stop scheduler, close db),
  * retries opening the mic with backoff so it never crash-loops at login if the
    audio session isn't ready yet.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import logging.handlers
import os
import signal
from contextlib import suppress
from pathlib import Path

import groq
import httpx

from aria.app import MissingSecret
from aria.config.loader import state_dir
from aria.config.schema import AriaConfig
from aria.core.runtime import friendly_error
from aria.core.session import build_voice_session
from aria.llm.base import LLMAuthError, LLMConnectionError, LLMRateLimitError
from aria.voice.audio import AudioError

log = logging.getLogger("aria.daemon")

_LOCK_NAME = "daemon.lock"
_LOG_NAME = "aria.log"
_MAX_BACKOFF_S = 30.0
# A rate-limit (free-tier daily cap) won't clear in seconds — wait at least this
# long before retrying so we don't hammer the API, but never exit the daemon.
_RATE_LIMIT_FLOOR_S = 60.0

# Network/DNS failures the daemon should ride out (NOT exit) — at boot the WiFi
# often isn't up yet, and a mid-session blip must never kill a long-lived daemon.
# We match both our normalized LLMConnectionError and the RAW provider/transport
# types, because some paths (e.g. Groq Whisper STT in stt_groq) call the client
# directly and don't translate their exceptions.
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    LLMConnectionError,
    groq.APIConnectionError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    OSError,  # socket.gaierror "Name or service not known" when DNS isn't ready
)


def classify_failure(exc: BaseException) -> str:
    """Bucket a voice-loop exception by how the daemon should react:

    * ``"fatal"`` — unrecoverable (bad/missing API key); stop the daemon, since
      retrying can't fix it.
    * ``"ratelimit"`` — provider rate-limited; back off a long floor and keep
      looping (the cap clears on its own).
    * ``"connection"`` — network/DNS unreachable; back off and keep looping (the
      WiFi/boot case and transient blips).
    * ``"unknown"`` — unexpected crash; log with a traceback, back off, keep going.

    Only ``"fatal"`` ends the daemon — everything else is retryable so a transient
    network problem at boot can't make Aria look dead.
    """
    if isinstance(exc, (LLMAuthError, MissingSecret)):
        return "fatal"
    if isinstance(exc, LLMRateLimitError):
        return "ratelimit"
    if isinstance(exc, _CONNECTION_ERRORS):
        return "connection"
    return "unknown"


def log_path() -> Path:
    return state_dir() / _LOG_NAME


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger("aria")
    if any(getattr(h, "_aria_daemon", False) for h in root.handlers):
        return
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        log_path(), maxBytes=1_000_000, backupCount=3
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()  # -> stdout, captured by journald
    stream_handler.setFormatter(fmt)
    for h in (file_handler, stream_handler):
        h._aria_daemon = True  # type: ignore[attr-defined]
        root.addHandler(h)


class SingleInstanceLock:
    """An exclusive flock on a state-dir file, auto-released when the process
    exits (even on crash). A second daemon can't acquire it."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (state_dir() / _LOCK_NAME)
        self._fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            with suppress(OSError):
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            self._fd = None


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


async def run_with_mic_retry(pipeline, respond, stop: asyncio.Event) -> None:
    """Run the voice loop, retrying transient failures with backoff so the daemon
    never looks dead/flapping. A not-yet-ready mic, a network/DNS hiccup (common in
    the first seconds after a reboot, before WiFi is up), or a rate-limit all back
    off and keep looping. ONLY a genuinely unrecoverable startup error (bad/missing
    key) stops the loop — see :func:`classify_failure`."""
    backoff = 1.0
    while not stop.is_set():
        try:
            await pipeline.run(respond)
            stop.set()  # frames exhausted (shouldn't happen with a live mic)
            return
        except asyncio.CancelledError:
            raise
        except AudioError as exc:
            log.warning("Audio device not ready (%s); retrying in %.0fs.", exc, backoff)
            await _sleep_or_stop(stop, backoff)
            backoff = min(backoff * 2, _MAX_BACKOFF_S)
        except BaseException as exc:  # noqa: BLE001
            kind = classify_failure(exc)
            if kind == "fatal":
                log.error("%s", friendly_error(exc) or exc)
                stop.set()  # retrying can't fix a bad key — let the daemon exit
                return
            if kind == "ratelimit":
                wait = max(backoff, _RATE_LIMIT_FLOOR_S)
                log.warning(
                    "Hit the API usage limit (%s); backing off %.0fs and staying up.",
                    exc, wait,
                )
                await _sleep_or_stop(stop, wait)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
            elif kind == "connection":
                log.warning(
                    "Network not reachable yet (%s); retrying in %.0fs (staying up).",
                    exc, backoff,
                )
                await _sleep_or_stop(stop, backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)
            else:  # unknown crash — keep the daemon alive but capture the traceback
                log.exception("Voice loop crashed; restarting in %.0fs.", backoff)
                await _sleep_or_stop(stop, backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF_S)


async def run_daemon(config: AriaConfig) -> int:
    setup_logging()
    lock = SingleInstanceLock()
    if not lock.acquire():
        log.error("Aria is already running in the background. Not starting a second copy.")
        return 1

    log.info("Aria daemon starting (pid %d).", os.getpid())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    def _log_latency(from_speech_end: float, from_wake: float) -> None:
        log.info(
            "Turn latency: %.1fs from end of speech to first audio (%.1fs from wake).",
            from_speech_end, from_wake,
        )

    try:
        session = await build_voice_session(config, on_latency=_log_latency)
    except MissingSecret:
        log.error("No Groq API key found. Run `aria setup` once, then `aria enable`.")
        lock.release()
        return 1
    except FileNotFoundError as exc:
        log.error("%s — run `aria setup` to fetch a voice.", exc)
        lock.release()
        return 1
    except BaseException:  # noqa: BLE001
        log.exception("Failed to start Aria.")
        lock.release()
        return 1

    log.info("Aria is listening in the background. Say the wake word any time.")
    runner = asyncio.create_task(
        run_with_mic_retry(session.pipeline, session.orchestrator.respond, stop)
    )
    try:
        await stop.wait()
        log.info("Shutting down.")
    finally:
        runner.cancel()
        with suppress(asyncio.CancelledError):
            await runner
        await session.aclose()
        lock.release()
        log.info("Aria daemon stopped.")
    return 0
