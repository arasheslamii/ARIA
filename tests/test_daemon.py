"""Background daemon, single-instance lock, control CLI, and service file."""

from __future__ import annotations

import asyncio
import configparser
import os
import signal
import types
from pathlib import Path

import pytest

from aria.core.daemon import (
    SingleInstanceLock,
    run_daemon,
    run_with_mic_retry,
)
from aria.core.service import control_command, logs_command, run_control
from aria.voice.audio import AudioError


def _dummy_request():
    """A minimal httpx.Request so we can construct a real groq.APIConnectionError
    (its __init__ requires a request) the way the SDK raises one at boot."""
    import httpx

    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


# --- single-instance lock -------------------------------------------------
def test_single_instance_lock(tmp_path):
    path = tmp_path / "daemon.lock"
    first = SingleInstanceLock(path)
    second = SingleInstanceLock(path)
    assert first.acquire() is True
    assert second.acquire() is False  # a second copy can't start
    first.release()
    assert second.acquire() is True  # freed after the holder releases
    second.release()


# --- control CLI -> systemctl --user commands -----------------------------
def test_control_command_builds_systemctl():
    assert control_command("enable") == [
        "systemctl", "--user", "enable", "--now", "aria.service"
    ]
    assert control_command("disable") == [
        "systemctl", "--user", "disable", "--now", "aria.service"
    ]
    assert control_command("start") == ["systemctl", "--user", "start", "aria.service"]
    assert control_command("stop") == ["systemctl", "--user", "stop", "aria.service"]
    assert control_command("status") == [
        "systemctl", "--user", "status", "--no-pager", "aria.service"
    ]
    with pytest.raises(ValueError):
        control_command("bogus")


def test_logs_command():
    cmd = logs_command()
    assert cmd[:5] == ["journalctl", "--user", "-u", "aria.service", "-n"]
    assert cmd[-1] == "-f"
    assert logs_command(follow=False)[-1] != "-f"


def test_run_control_uses_runner():
    calls: list[list[str]] = []
    rc = run_control("enable", runner=lambda cmd: calls.append(cmd) or 0)
    assert rc == 0
    # enable turns on linger (so Aria boots without a login) THEN enables the unit.
    assert calls == [
        ["loginctl", "enable-linger"],
        ["systemctl", "--user", "enable", "--now", "aria.service"],
    ]
    run_control("logs", runner=lambda cmd: calls.append(cmd) or 0)
    assert calls[-1][0] == "journalctl"


def test_enable_linger_is_best_effort_and_non_fatal():
    from aria.core.service import enable_linger_command

    assert enable_linger_command() == ["loginctl", "enable-linger"]

    # If loginctl is missing the runner raises — `aria enable` must NOT fail; the
    # systemctl enable still runs and we return its code.
    seen: list[list[str]] = []

    def flaky(cmd):
        seen.append(cmd)
        if cmd[0] == "loginctl":
            raise FileNotFoundError("no loginctl")
        return 0

    rc = run_control("enable", runner=flaky)
    assert rc == 0
    assert ["systemctl", "--user", "enable", "--now", "aria.service"] in seen


def test_other_actions_do_not_touch_linger():
    calls: list[list[str]] = []
    run_control("start", runner=lambda cmd: calls.append(cmd) or 0)
    assert calls == [["systemctl", "--user", "start", "aria.service"]]  # no linger


# --- mic retry loop -------------------------------------------------------
async def test_mic_retry_backs_off_then_stops(monkeypatch):
    monkeypatch.setattr("aria.core.daemon._MAX_BACKOFF_S", 0.01)
    attempts = {"n": 0}
    stop = asyncio.Event()

    class Pipe:
        async def run(self, _respond):
            attempts["n"] += 1
            if attempts["n"] >= 3:
                stop.set()  # mic "came up" / we're done — exit the retry loop
            raise AudioError("mic busy")

    await asyncio.wait_for(run_with_mic_retry(Pipe(), lambda t: None, stop), timeout=2)
    assert attempts["n"] >= 3  # retried with backoff instead of crashing


async def test_mic_retry_stops_on_fatal_error():
    from aria.llm.base import LLMAuthError

    stop = asyncio.Event()

    class Pipe:
        async def run(self, _respond):
            raise LLMAuthError("401")

    await asyncio.wait_for(run_with_mic_retry(Pipe(), lambda t: None, stop), timeout=2)
    assert stop.is_set()  # unrecoverable -> signals the daemon to exit, no loop


@pytest.mark.parametrize(
    "exc_factory",
    [
        pytest.param(lambda: __import__("aria.llm.base", fromlist=["LLMConnectionError"])
                     .LLMConnectionError("dns"), id="normalized"),
        pytest.param(
            lambda: __import__("groq").APIConnectionError(request=_dummy_request()),
            id="raw-groq",
        ),
        pytest.param(lambda: OSError(-2, "Name or service not known"), id="gaierror"),
    ],
)
async def test_mic_retry_survives_connection_error_and_keeps_looping(monkeypatch, exc_factory):
    # BOOT case: WiFi/DNS isn't up yet. A connection error must back off and RETRY,
    # never stop the daemon (otherwise Aria looks dead right after a reboot).
    monkeypatch.setattr("aria.core.daemon._MAX_BACKOFF_S", 0.01)
    attempts = {"n": 0}
    stop = asyncio.Event()

    class Pipe:
        async def run(self, _respond):
            attempts["n"] += 1
            if attempts["n"] >= 3:  # "network came up" -> let the test end the loop
                stop.set()
            raise exc_factory()

    await asyncio.wait_for(run_with_mic_retry(Pipe(), lambda t: None, stop), timeout=2)
    assert attempts["n"] >= 3  # kept retrying through the offline window


async def test_mic_retry_survives_rate_limit(monkeypatch):
    from aria.llm.base import LLMRateLimitError

    # Rate-limit is retryable (with a long floor), never fatal.
    monkeypatch.setattr("aria.core.daemon._MAX_BACKOFF_S", 0.01)
    monkeypatch.setattr("aria.core.daemon._RATE_LIMIT_FLOOR_S", 0.01)
    attempts = {"n": 0}
    stop = asyncio.Event()

    class Pipe:
        async def run(self, _respond):
            attempts["n"] += 1
            if attempts["n"] >= 2:
                stop.set()
            raise LLMRateLimitError("429 daily cap")

    await asyncio.wait_for(run_with_mic_retry(Pipe(), lambda t: None, stop), timeout=2)
    assert attempts["n"] >= 2  # stayed up across the cap instead of exiting


def test_classify_failure_buckets():
    import groq

    from aria.app import MissingSecret
    from aria.core.daemon import classify_failure
    from aria.llm.base import LLMAuthError, LLMConnectionError, LLMRateLimitError

    assert classify_failure(LLMAuthError("bad")) == "fatal"
    assert classify_failure(MissingSecret("no key")) == "fatal"
    assert classify_failure(LLMRateLimitError("429")) == "ratelimit"
    assert classify_failure(LLMConnectionError("down")) == "connection"
    assert classify_failure(groq.APIConnectionError(request=_dummy_request())) == "connection"
    assert classify_failure(OSError(-2, "Name or service not known")) == "connection"
    assert classify_failure(ValueError("???")) == "unknown"


# --- full daemon lifecycle: clean shutdown on SIGINT ----------------------
async def test_run_daemon_clean_shutdown_on_signal(monkeypatch, tmp_path):
    monkeypatch.setattr("aria.core.daemon.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        "aria.core.daemon.SingleInstanceLock", lambda: SingleInstanceLock(tmp_path / "d.lock")
    )

    started = asyncio.Event()
    closed: list[bool] = []

    class FakePipeline:
        async def run(self, _respond):
            started.set()
            await asyncio.Event().wait()  # block until cancelled by shutdown

    class FakeSession:
        pipeline = FakePipeline()
        orchestrator = types.SimpleNamespace(respond=lambda t: None)

        async def aclose(self):
            closed.append(True)

    async def fake_build(config, **kw):
        return FakeSession()

    monkeypatch.setattr("aria.core.daemon.build_voice_session", fake_build)

    task = asyncio.create_task(run_daemon(config=None))
    await asyncio.wait_for(started.wait(), timeout=2)
    os.kill(os.getpid(), signal.SIGINT)  # caught by the daemon's loop handler
    rc = await asyncio.wait_for(task, timeout=3)

    assert rc == 0
    assert closed == [True]  # session torn down cleanly


async def test_run_daemon_refuses_second_instance(monkeypatch, tmp_path):
    monkeypatch.setattr("aria.core.daemon.setup_logging", lambda *a, **k: None)
    lock_path = tmp_path / "d.lock"
    monkeypatch.setattr(
        "aria.core.daemon.SingleInstanceLock", lambda: SingleInstanceLock(lock_path)
    )
    holder = SingleInstanceLock(lock_path)
    assert holder.acquire() is True
    try:
        rc = await run_daemon(config=None)  # build_voice_session never reached
        assert rc == 1
    finally:
        holder.release()


# --- the systemd unit file is valid --------------------------------------
def test_service_file_parses():
    path = Path(__file__).parents[1] / "aria" / "packaging" / "aria.service"
    cp = configparser.ConfigParser(strict=False)
    cp.read(path)
    assert cp.get("Service", "ExecStart").endswith("aria daemon")
    assert cp.get("Service", "Restart") == "on-failure"
    assert cp.get("Install", "WantedBy") == "default.target"
    # Don't start before the network is up (DNS not ready right after a reboot).
    assert "network-online.target" in cp.get("Unit", "After")
    assert "network-online.target" in cp.get("Unit", "Wants")
