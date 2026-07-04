"""Hold-to-talk: a global keyboard listener reading /dev/input directly.

Reading the kernel input devices is the only approach that works everywhere
Aria runs — X11, Wayland, AND the headless systemd --user daemon (desktop
hotkey APIs need a compositor, and can't see key-RELEASE events anyway, which
hold-to-talk depends on). It's done in pure Python: the ``input_event`` struct
and the keycode numbers are a stable kernel ABI, so no evdev C extension (and
no compile-time Python headers) are needed.

The cost: the user must be able to read /dev/input, i.e. be in the ``input``
group. Everything degrades gracefully — if access is denied, the caller falls
back to the wake word and logs the one-line fix.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import struct
from contextlib import suppress
from pathlib import Path

log = logging.getLogger("aria.voice")

# Friendly names offered in the wizard -> Linux keycodes (input-event-codes.h,
# a frozen kernel ABI). Chosen keys don't type anything / aren't part of normal
# shortcuts, so holding one to talk never fights another app.
KEY_CHOICES: dict[str, int] = {
    "right ctrl": 97,   # KEY_RIGHTCTRL
    "left ctrl": 29,    # KEY_LEFTCTRL
    "right alt": 100,   # KEY_RIGHTALT
    "caps lock": 58,    # KEY_CAPSLOCK
    "scroll lock": 70,  # KEY_SCROLLLOCK
    "pause": 119,       # KEY_PAUSE
    "menu": 127,        # KEY_COMPOSE
    "f8": 66,           # KEY_F8
    "f9": 67,           # KEY_F9
}

PERMISSION_FIX = (
    "add yourself to the input group (`sudo usermod -aG input $USER`), then log "
    "out and back in"
)

_EV_KEY = 1  # event type for key press/release
# struct input_event: struct timeval (2 longs) + u16 type + u16 code + s32 value.
_EVENT_FORMAT = "llHHi"
_EVENT_SIZE = struct.calcsize(_EVENT_FORMAT)

_PROC_DEVICES = "/proc/bus/input/devices"
_DEV_DIR = "/dev/input"


def resolve_key(name: str) -> int:
    """Friendly name -> Linux keycode. Raises ValueError with the valid choices
    on garbage, so a config typo fails loudly (and visibly) at startup."""
    n = (name or "").strip().lower()
    if n in KEY_CHOICES:
        return KEY_CHOICES[n]
    raise ValueError(
        f"Unknown hold-to-talk key {name!r}. Choices: {', '.join(KEY_CHOICES)}."
    )


def _bit_set(hex_words: str, bit: int) -> bool:
    """Test a bit in a /proc bitmask line ('B: KEY=... '): space-separated hex
    words, most-significant first, 64 bits per word (native long)."""
    words = list(reversed(hex_words.split()))
    idx, off = bit // 64, bit % 64
    if idx >= len(words):
        return False
    try:
        return bool((int(words[idx], 16) >> off) & 1)
    except ValueError:
        return False


def keyboards_with_key(keycode: int, proc_text: str | None = None) -> list[str]:
    """Paths of /dev/input/event* devices that (a) emit key events and (b) have
    ``keycode`` on them, parsed from /proc/bus/input/devices."""
    if proc_text is None:
        try:
            proc_text = Path(_PROC_DEVICES).read_text()
        except OSError:
            return []
    paths: list[str] = []
    for block in proc_text.split("\n\n"):
        handlers = re.search(r"^H: Handlers=.*?\b(event\d+)", block, re.MULTILINE)
        key_mask = re.search(r"^B: KEY=([0-9a-fA-F ]+)$", block, re.MULTILINE)
        if not handlers or not key_mask:
            continue
        if _bit_set(key_mask.group(1), keycode):
            paths.append(f"{_DEV_DIR}/{handlers.group(1)}")
    return paths


def access_problem() -> str | None:
    """Quick probe for the wizard: can hold-to-talk work here at all? Returns a
    human reason if not, else None."""
    try:
        events = sorted(Path(_DEV_DIR).glob("event*"))
    except OSError:
        events = []
    if not events:
        return "no input devices found under /dev/input"
    if not any(os.access(p, os.R_OK) for p in events):
        return f"no permission to read the keyboard — {PERMISSION_FIX}"
    return None


class HotkeyListener:
    """Watches every readable keyboard for the configured key and exposes a
    plain ``pressed`` bool the voice pipeline polls once per audio frame."""

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        self.pressed = False
        self.reason = ""  # why start() said no, for the log/wizard
        self._fds: list[int] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> bool:
        try:
            keycode = resolve_key(self.key_name)
        except ValueError as exc:
            self.reason = str(exc)
            return False
        paths = keyboards_with_key(keycode)
        if not paths:
            self.reason = f"no keyboard with a '{self.key_name}' key was found"
            return False
        self._loop = asyncio.get_running_loop()
        denied = False
        for path in paths:
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                denied = True
                continue
            except OSError:
                continue
            self._fds.append(fd)
            self._loop.add_reader(fd, self._on_readable, fd, keycode)
        if not self._fds:
            self.reason = (
                f"no permission to read the keyboard — {PERMISSION_FIX}"
                if denied else "the keyboard device could not be opened"
            )
            return False
        log.info(
            "Hold-to-talk armed on '%s' (%d keyboard(s)).", self.key_name, len(self._fds)
        )
        return True

    def _on_readable(self, fd: int, keycode: int) -> None:
        try:
            data = os.read(fd, _EVENT_SIZE * 64)
        except BlockingIOError:
            return
        except OSError:  # keyboard unplugged — drop it, never crash the loop
            self._drop(fd)
            return
        for off in range(0, len(data) - _EVENT_SIZE + 1, _EVENT_SIZE):
            _s, _u, etype, code, value = struct.unpack_from(_EVENT_FORMAT, data, off)
            if etype == _EV_KEY and code == keycode:
                # value: 1 = press, 2 = auto-repeat (still held), 0 = release
                self.pressed = value != 0

    def _drop(self, fd: int) -> None:
        if self._loop is not None:
            with suppress(Exception):
                self._loop.remove_reader(fd)
        with suppress(OSError):
            os.close(fd)
        if fd in self._fds:
            self._fds.remove(fd)
        if not self._fds:
            self.pressed = False

    async def aclose(self) -> None:
        for fd in list(self._fds):
            self._drop(fd)
