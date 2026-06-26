"""Unit tests for native tools and the safety/executor layer."""

from __future__ import annotations

import pytest

from aria.core.executor import ExecConfig, ToolExecutor
from aria.safety.audit import AuditLog
from aria.safety.permissions import classify
from aria.tools.math_tool import MathTool
from aria.tools.timers import parse_duration


async def test_math_basic():
    res = await MathTool().run(expression="0.18 * 240")
    assert res.data["value"] == pytest.approx(43.2)
    assert res.spoken == "43.2"


async def test_math_functions():
    res = await MathTool().run(expression="sqrt(2)")
    assert res.data["value"] == pytest.approx(1.41421356, rel=1e-6)


async def test_math_rejects_code_injection():
    from aria.tools.base import ToolError

    with pytest.raises(ToolError):
        await MathTool().run(expression="__import__('os').system('echo hi')")


def test_parse_duration():
    assert parse_duration("10 minutes") == 600
    assert parse_duration("1h30m") == 5400
    assert parse_duration("90 seconds") == 90


_LITE_HTML = """
<table>
<tr><td><a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x"
    class='result-link'>First &amp; Best Result</a></td></tr>
<tr><td class='result-snippet'>A <b>great</b> snippet about A.</td></tr>
<tr><td><a href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb"
    class='result-link'>Second Result</a></td></tr>
<tr><td class='result-snippet'>Snippet about B.</td></tr>
</table>
"""


async def test_web_search_parses_organic_results(monkeypatch):
    import httpx

    from aria.tools.search import WebSearchTool

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, text=_LITE_HTML, request=httpx.Request("POST", url))

    async def fake_get(self, url, **kwargs):  # Instant Answer -> no abstract
        return httpx.Response(200, json={"AbstractText": ""}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    res = await WebSearchTool().run(query="anything", max_results=5)
    results = res.data["results"]
    assert results[0]["url"] == "https://example.com/a"
    assert results[0]["title"] == "First & Best Result"  # entities unescaped, tags stripped
    assert results[0]["snippet"] == "A great snippet about A."
    assert results[1]["url"] == "https://example.org/b"


async def test_web_search_abstract_is_surfaced_first(monkeypatch):
    import httpx

    from aria.tools.search import WebSearchTool

    async def fake_post(self, url, **kwargs):
        return httpx.Response(200, text=_LITE_HTML, request=httpx.Request("POST", url))

    async def fake_get(self, url, **kwargs):
        return httpx.Response(
            200,
            json={
                "AbstractText": "Authoritative fact.",
                "Heading": "Topic",
                "AbstractURL": "https://wikipedia.org/topic",
            },
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)

    res = await WebSearchTool().run(query="topic")
    results = res.data["results"]
    assert results[0]["snippet"] == "Authoritative fact."
    assert results[0]["url"] == "https://wikipedia.org/topic"


def test_permission_classification():
    tool = MathTool()
    assert classify(tool, {"expression": "1+1"}).risk == "safe"
    # Dangerous-looking args escalate even a 'safe' tool to confirm.
    assert classify(tool, {"expression": "rm -rf /"}).risk == "confirm"


async def test_executor_blocks_unconfirmed_risky(tmp_path):
    from aria.tools.system import LockScreenTool

    audit = AuditLog(tmp_path / "audit.log")
    execu = ToolExecutor(audit, ExecConfig(require_confirmation=True))
    # No confirm callback -> risky action is declined, not run.
    result = await execu.execute(LockScreenTool(), {}, confirm=None)
    assert "declined" in result.content
    assert (tmp_path / "audit.log").exists()


async def test_executor_runs_after_confirm(tmp_path):
    from aria.tools.base import Tool, ToolResult

    class Risky(Tool):
        name = "risky"
        description = "x"
        risk = "confirm"

        async def run(self, **kwargs):
            return ToolResult(content="done")

    async def yes(_name):
        return True

    execu = ToolExecutor(AuditLog(tmp_path / "a.log"), ExecConfig(require_confirmation=True))
    result = await execu.execute(Risky(), {}, confirm=yes)
    assert result.content == "done"


# --- lock_screen picks a REAL locker and reports honestly -----------------
def _patch_lockers(monkeypatch, available: set[str], recorder: list[list[str]]):
    import aria.tools.system as sysmod

    monkeypatch.setattr(sysmod.shutil, "which", lambda c: f"/usr/bin/{c}" if c in available else None)

    async def fake_run(cmd):
        recorder.append(cmd)
        return ""

    monkeypatch.setattr(sysmod, "_run", fake_run)


async def test_lock_screen_prefers_real_locker(monkeypatch):
    from aria.tools.system import LockScreenTool

    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "X-Cinnamon")
    ran: list[list[str]] = []
    # Both a real locker and the weak loginctl are available; the real one wins.
    _patch_lockers(monkeypatch, {"cinnamon-screensaver-command", "loginctl"}, ran)

    res = await LockScreenTool().run()
    assert ran == [["cinnamon-screensaver-command", "--lock"]]
    assert res.content == "locked"


async def test_lock_screen_loginctl_is_flagged_best_effort(monkeypatch):
    from aria.tools.system import LockScreenTool

    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "X-Cinnamon")
    ran: list[list[str]] = []
    # Only the weak fallback exists — it's used but NOT reported as a real lock.
    _patch_lockers(monkeypatch, {"loginctl"}, ran)

    res = await LockScreenTool().run()
    assert ran == [["loginctl", "lock-session"]]
    assert res.content != "locked"
    assert "best-effort" in res.content
    assert "may not have worked" in (res.spoken or "")


async def test_lock_screen_no_locker_errors_honestly(monkeypatch):
    from aria.tools.base import ToolError
    from aria.tools.system import LockScreenTool

    _patch_lockers(monkeypatch, set(), [])
    with pytest.raises(ToolError, match="no screen locker"):
        await LockScreenTool().run()


async def test_lock_screen_respects_desktop_order(monkeypatch):
    from aria.tools.system import LockScreenTool

    # On GNOME with both lockers present, the GNOME one is bubbled to the front.
    monkeypatch.setenv("XDG_CURRENT_DESKTOP", "GNOME")
    ran: list[list[str]] = []
    _patch_lockers(monkeypatch, {"cinnamon-screensaver-command", "gnome-screensaver-command"}, ran)

    await LockScreenTool().run()
    assert ran[0] == ["gnome-screensaver-command", "-l"]
