"""Control CLI: thin wrappers over `systemctl --user` for the Aria daemon.

`aria enable | disable | start | stop | status | logs` so the user never has to
remember systemctl incantations. Kept dependency-free and mockable: the command
builders return arg lists, and a runner executes them.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable

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


def run_control(action: str, *, runner: Callable[[list[str]], int] = subprocess.call) -> int:
    """Run a control action. ``action`` is one of the _CONTROL keys or 'logs'."""
    cmd = logs_command() if action == "logs" else control_command(action)
    return int(runner(cmd))
