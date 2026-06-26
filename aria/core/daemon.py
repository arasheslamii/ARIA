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

from aria.app import MissingSecret
from aria.config.loader import state_dir
from aria.config.schema import AriaConfig
from aria.core.runtime import friendly_error
from aria.core.session import build_voice_session
from aria.voice.audio import AudioError

log = logging.getLogger("aria.daemon")

_LOCK_NAME = "daemon.lock"
_LOG_NAME = "aria.log"
_MAX_BACKOFF_S = 30.0


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
    """Run the voice loop, retrying audio-device failures with backoff so a not-
    yet-ready mic at login doesn't crash the service. Fatal, unrecoverable errors
    (bad key, offline) are logged once and stop the loop rather than crash-loop."""
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
            friendly = friendly_error(exc)
            if friendly is not None:
                log.error("%s", friendly)
                stop.set()  # not recoverable by retrying — let the daemon exit
                return
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

    try:
        session = await build_voice_session(config)
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
