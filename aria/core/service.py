"""Control CLI: thin wrappers over `systemctl --user` for the Aria daemon.

`aria enable | disable | start | stop | status | logs` so the user never has to
remember systemctl incantations. Kept dependency-free and mockable: the command
builders return arg lists, and a runner executes them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from contextlib import suppress

SERVICE = "aria.service"

# Actions that map to a single `systemctl --user <verb...> aria.service`.
_CONTROL = {
    "enable": ("enable", "--now"),   # start now AND on every login
    "disable": ("disable", "--now"),
    "start": ("start",),
    "stop": ("stop",),
    "status": ("status", "--no-pager"),
}


def control_command(action: str) -> list[str]:
    if action not in _CONTROL:
        raise ValueError(f"unknown control action: {action!r}")
    return ["systemctl", "--user", *_CONTROL[action], SERVICE]


def logs_command(*, follow: bool = True, lines: int = 200) -> list[str]:
    cmd = ["journalctl", "--user", "-u", SERVICE, "-n", str(lines)]
    if follow:
        cmd.append("-f")
    return cmd


def enable_linger_command() -> list[str]:
    """Enable *linger* for the current user so their systemd --user manager — and
    therefore Aria — starts at BOOT, before (or without) a graphical login.

    Without linger, ``systemctl --user enable`` only starts Aria when the user logs
    in interactively, so after a headless reboot she'd look dead. ``aria enable``
    runs as the user, so no arg is needed (enables linger for self)."""
    return ["loginctl", "enable-linger"]


def run_control(action: str, *, runner: Callable[[list[str]], int] = subprocess.call) -> int:
    """Run a control action. ``action`` is one of the _CONTROL keys or 'logs'."""
    if action == "enable":
        # Best-effort and idempotent: turn on linger so Aria auto-starts at boot
        # without a login. Never fail `aria enable` if loginctl is absent/denied.
        with suppress(Exception):
            runner(enable_linger_command())
    cmd = logs_command() if action == "logs" else control_command(action)
    return int(runner(cmd))
