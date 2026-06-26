"""Power tools (confirm-gated), reliable open_app, and honest failures."""

from __future__ import annotations

import pytest

import aria.tools.system as sysmod
from aria.safety.permissions import classify
from aria.tools.base import ToolError
from aria.tools.system import (
    LogOutTool,
    OpenAppTool,
    PowerOffTool,
    RebootTool,
    SuspendTool,
)


def _patch_run(monkeypatch, available: set[str], recorder: list[list[str]]):
    monkeypatch.setattr(
        sysmod.shutil, "which", lambda c: f"/usr/bin/{c}" if c in available else None
    )

    async def fake_run(cmd):
        recorder.append(cmd)
        return ""

    monkeypatch.setattr(sysmod, "_run", fake_run)


# --- power tools are confirm-gated ---------------------------------------
@pytest.mark.parametrize("tool", [RebootTool(), PowerOffTool(), SuspendTool(), LogOutTool()])
def test_power_tools_classify_as_confirm(tool):
    assert classify(tool, {}).risk == "confirm"  # fires the two-turn confirmation


async def test_reboot_runs_first_available(monkeypatch):
    ran: list[list[str]] = []
    _patch_run(monkeypatch, {"systemctl"}, ran)
    res = await RebootTool().run()
    assert ran == [["systemctl", "reboot"]]
    assert "restart" in res.spoken.lower()


async def test_reboot_falls_back_when_no_systemctl(monkeypatch):
    ran: list[list[str]] = []
    _patch_run(monkeypatch, {"shutdown"}, ran)
    await RebootTool().run()
    assert ran == [["shutdown", "-r", "now"]]


async def test_power_off_honest_error_when_unavailable(monkeypatch):
    _patch_run(monkeypatch, set(), [])
    with pytest.raises(ToolError, match="power off"):
        await PowerOffTool().run()


async def test_logout_prefers_desktop_specific(monkeypatch):
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "X-Cinnamon")
    ran: list[list[str]] = []
    _patch_run(monkeypatch, {"cinnamon-session-quit", "loginctl"}, ran)
    await LogOutTool().run()
    assert ran[0][0] == "cinnamon-session-quit"


# --- open_app reliability + honest failure -------------------------------
async def test_open_app_resolves_generic_alias(monkeypatch):
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        sysmod.shutil, "which", lambda c: "/usr/bin/" + c if c == "gnome-terminal" else None
    )

    async def fake_spawn(cmd):
        spawned.append(cmd)

    monkeypatch.setattr(sysmod, "_spawn", fake_spawn)
    res = await OpenAppTool().run(app="terminal")
    assert spawned == [["gnome-terminal"]]
    assert "opening" in res.spoken.lower()


async def test_open_app_honest_failure(monkeypatch):
    monkeypatch.setattr(sysmod.shutil, "which", lambda c: None)
    with pytest.raises(ToolError, match="couldn't find"):
        await OpenAppTool().run(app="nonsuchapp")
