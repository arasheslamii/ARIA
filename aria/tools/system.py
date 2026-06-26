"""System-control tools for Linux desktops.

Each tool shells out to the standard utility for its job and degrades with a
clear spoken error if the utility is missing. We try modern tools first
(wpctl/PipeWire, brightnessctl) and fall back where sensible. Anything that
changes system state or is irreversible is marked ``confirm``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any

from aria.tools.base import Tool, ToolError, ToolResult


async def _run(cmd: list[str]) -> str:
    if not shutil.which(cmd[0]):
        raise ToolError(f"'{cmd[0]}' is not installed")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise ToolError(err.decode().strip() or f"{cmd[0]} failed")
    return out.decode().strip()


class VolumeTool(Tool):
    name = "set_volume"
    description = "Set, mute, or unmute system volume. level is 0-100, or use action mute/unmute."
    parameters = {
        "type": "object",
        "properties": {
            "level": {"type": "integer", "description": "0-100 volume percentage."},
            "action": {"type": "string", "enum": ["set", "mute", "unmute", "toggle"]},
        },
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "set")
        sink = "@DEFAULT_AUDIO_SINK@"
        if action in ("mute", "unmute", "toggle"):
            flag = {"mute": "1", "unmute": "0", "toggle": "toggle"}[action]
            await _run(["wpctl", "set-mute", sink, flag])
            return ToolResult(content=f"{action}d", spoken=f"{action.capitalize()}d.")
        level = int(kwargs.get("level", 50))
        level = max(0, min(100, level))
        await _run(["wpctl", "set-volume", sink, f"{level}%"])
        return ToolResult(content=f"volume {level}%", spoken=f"Volume set to {level} percent.")


class BrightnessTool(Tool):
    name = "set_brightness"
    description = "Set screen brightness to a percentage (0-100)."
    parameters = {
        "type": "object",
        "properties": {"level": {"type": "integer", "description": "0-100."}},
        "required": ["level"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        level = max(1, min(100, int(kwargs.get("level", 50))))
        await _run(["brightnessctl", "set", f"{level}%"])
        return ToolResult(content=f"brightness {level}%", spoken=f"Brightness {level} percent.")


class MediaTool(Tool):
    name = "media_control"
    description = "Control media playback: play, pause, playpause, next, previous, stop."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["play", "pause", "play-pause", "next", "previous", "stop"],
            }
        },
        "required": ["action"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "play-pause")
        await _run(["playerctl", action])
        return ToolResult(content=action, spoken="Done.")


class ScreenshotTool(Tool):
    name = "screenshot"
    description = "Take a screenshot and save it to the Pictures folder."
    parameters = {"type": "object", "properties": {}}
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        from datetime import datetime
        from pathlib import Path

        out = Path.home() / "Pictures" / f"aria-{datetime.now():%Y%m%d-%H%M%S}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        for cmd in (["grim", str(out)], ["gnome-screenshot", "-f", str(out)],
                    ["scrot", str(out)]):
            if shutil.which(cmd[0]):
                await _run(cmd)
                return ToolResult(content=str(out), spoken="Screenshot saved.")
        raise ToolError("no screenshot tool found (grim/gnome-screenshot/scrot)")


class ClipboardTool(Tool):
    name = "clipboard"
    description = "Read or write the system clipboard. action=read|write, text for write."
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["read", "write"]},
            "text": {"type": "string"},
        },
        "required": ["action"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        action = kwargs["action"]
        wl, xc = shutil.which("wl-copy"), shutil.which("xclip")
        if action == "read":
            if shutil.which("wl-paste"):
                text = await _run(["wl-paste", "--no-newline"])
            elif xc:
                text = await _run(["xclip", "-selection", "clipboard", "-o"])
            else:
                raise ToolError("no clipboard tool (wl-paste/xclip)")
            return ToolResult(content=text, spoken="Here's what's on your clipboard.")
        text = str(kwargs.get("text", ""))
        proc_cmd = ["wl-copy"] if wl else ["xclip", "-selection", "clipboard"]
        proc = await asyncio.create_subprocess_exec(*proc_cmd, stdin=asyncio.subprocess.PIPE)
        await proc.communicate(text.encode())
        return ToolResult(content="copied", spoken="Copied to clipboard.")


# Real lockers that actually engage the screensaver, with the desktop they suit.
# Ordered so the session's native locker is tried first. `loginctl lock-session`
# is deliberately NOT here: it returns 0 on desktops (e.g. Cinnamon) where it does
# nothing, so it must never be treated as a strong success signal.
_STRONG_LOCKERS: list[tuple[list[str], str | None]] = [
    (["cinnamon-screensaver-command", "--lock"], "cinnamon"),
    (["mate-screensaver-command", "--lock"], "mate"),
    (["xfce4-screensaver-command", "--lock"], "xfce"),
    (["gnome-screensaver-command", "-l"], "gnome"),
    (["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"], "kde"),
    (["xdg-screensaver", "lock"], None),  # generic X11 fallback
    (["swaylock"], "wayland"),
    (["hyprlock"], "wayland"),
]


def _ordered_lockers() -> list[list[str]]:
    """Strong lockers, with the one matching $XDG_CURRENT_DESKTOP bubbled first."""
    desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    ranked = sorted(
        _STRONG_LOCKERS, key=lambda item: 0 if item[1] and item[1] in desktop else 1
    )
    return [cmd for cmd, _ in ranked]


class LockScreenTool(Tool):
    name = "lock_screen"
    description = "Lock the screen."
    parameters = {"type": "object", "properties": {}}
    risk = "confirm"  # outward/disruptive — confirm first

    async def run(self, **kwargs: Any) -> ToolResult:
        # Prefer a real locker that actually engages on this desktop.
        for cmd in _ordered_lockers():
            if shutil.which(cmd[0]):
                await _run(cmd)
                return ToolResult(content="locked", spoken="Locking the screen.")
        # Weak last resort: loginctl returns success even when it doesn't lock, so
        # report honestly rather than claiming the screen is locked.
        if shutil.which("loginctl"):
            await _run(["loginctl", "lock-session"])
            return ToolResult(
                content="lock requested via loginctl (best-effort; may not lock on this desktop)",
                spoken="I asked your session to lock, but it may not have worked on this desktop.",
            )
        raise ToolError("I couldn't lock the screen — no screen locker found.")


# Friendly names -> ordered lists of real executables to try (first found wins).
_APP_ALIASES: dict[str, list[str]] = {
    "terminal": ["x-terminal-emulator", "gnome-terminal", "konsole", "kitty",
                 "alacritty", "xfce4-terminal", "mate-terminal", "xterm"],
    "browser": ["x-www-browser", "firefox", "google-chrome", "chromium", "chromium-browser"],
    "files": ["nautilus", "nemo", "dolphin", "thunar", "pcmanfm"],
    "file manager": ["nautilus", "nemo", "dolphin", "thunar", "pcmanfm"],
    "editor": ["code", "gnome-text-editor", "gedit", "kate", "mousepad"],
    "text editor": ["code", "gnome-text-editor", "gedit", "kate", "mousepad"],
    "calculator": ["gnome-calculator", "kcalc", "galculator"],
}


async def _spawn(cmd: list[str]) -> None:
    await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )


class OpenAppTool(Tool):
    name = "open_app"
    description = (
        "Launch a desktop application by name. Understands generic names like "
        "'terminal', 'browser', 'files', 'editor', 'calculator', plus specific "
        "executables (firefox, code). Reports honestly if nothing matches."
    )
    parameters = {
        "type": "object",
        "properties": {"app": {"type": "string", "description": "App name or command."}},
        "required": ["app"],
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        app = str(kwargs.get("app", "")).strip()
        if not app:
            raise ToolError("Which app would you like me to open?")
        # Resolve generic names to real binaries, else try the literal command.
        candidates = _APP_ALIASES.get(app.lower(), [app])
        for cand in candidates:
            exe = cand.split()[0]
            if shutil.which(exe):
                await _spawn(cand.split())
                return ToolResult(content=f"launched {exe}", spoken=f"Opening {app}.")
        # Fall back to launching a .desktop entry by id (e.g. "firefox").
        if shutil.which("gtk-launch"):
            try:
                await _run(["gtk-launch", app])
                return ToolResult(content=f"launched {app}", spoken=f"Opening {app}.")
            except ToolError:
                pass
        raise ToolError(f"I couldn't find an app called {app}.")


async def _power_action(candidates: list[list[str]], *, spoken: str, fail: str) -> ToolResult:
    """Run the first available power command (which()-detected); honest error if
    none exist. Used by reboot/poweroff/suspend/logout — all confirm-gated."""
    for cmd in candidates:
        if shutil.which(cmd[0]):
            await _run(cmd)
            return ToolResult(content=" ".join(cmd), spoken=spoken)
    raise ToolError(fail)


class RebootTool(Tool):
    name = "reboot"
    description = "Restart (reboot) the computer."
    parameters = {"type": "object", "properties": {}}
    risk = "confirm"

    async def run(self, **kwargs: Any) -> ToolResult:
        return await _power_action(
            [["systemctl", "reboot"], ["shutdown", "-r", "now"], ["reboot"]],
            spoken="Restarting now.",
            fail="I can't restart this machine — no reboot command is available.",
        )


class PowerOffTool(Tool):
    name = "power_off"
    description = "Shut down / power off the computer."
    parameters = {"type": "object", "properties": {}}
    risk = "confirm"

    async def run(self, **kwargs: Any) -> ToolResult:
        return await _power_action(
            [["systemctl", "poweroff"], ["shutdown", "-h", "now"], ["poweroff"]],
            spoken="Shutting down now.",
            fail="I can't power off this machine — no shutdown command is available.",
        )


class SuspendTool(Tool):
    name = "suspend"
    description = "Suspend / put the computer to sleep."
    parameters = {"type": "object", "properties": {}}
    risk = "confirm"

    async def run(self, **kwargs: Any) -> ToolResult:
        return await _power_action(
            [["systemctl", "suspend"], ["loginctl", "suspend"]],
            spoken="Going to sleep.",
            fail="I can't suspend this machine — no suspend command is available.",
        )


def _logout_candidates() -> list[list[str]]:
    de = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
    cmds: list[list[str]] = []
    if "cinnamon" in de:
        cmds.append(["cinnamon-session-quit", "--logout", "--no-prompt"])
    if "gnome" in de:
        cmds.append(["gnome-session-quit", "--logout", "--no-prompt"])
    if "mate" in de:
        cmds.append(["mate-session-save", "--logout"])
    if "xfce" in de:
        cmds.append(["xfce4-session-logout", "--logout"])
    # Generic graceful options, then the blunt logind fallback.
    cmds.append(["gnome-session-quit", "--logout", "--no-prompt"])
    user = os.environ.get("USER", "")
    if user:
        cmds.append(["loginctl", "terminate-user", user])
    return cmds


class LogOutTool(Tool):
    name = "log_out"
    description = "Log out of the current desktop session."
    parameters = {"type": "object", "properties": {}}
    risk = "confirm"

    async def run(self, **kwargs: Any) -> ToolResult:
        return await _power_action(
            _logout_candidates(),
            spoken="Logging you out.",
            fail="I can't log out from here — no session command is available.",
        )


def system_tools() -> list[Tool]:
    return [
        VolumeTool(),
        BrightnessTool(),
        MediaTool(),
        ScreenshotTool(),
        ClipboardTool(),
        LockScreenTool(),
        OpenAppTool(),
        RebootTool(),
        PowerOffTool(),
        SuspendTool(),
        LogOutTool(),
    ]
