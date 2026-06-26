"""Branded desktop notifications (FIX 3) and the never-silent guarantee (FIX 2)."""

from __future__ import annotations

from aria.core.memory import Memory
from aria.core.orchestrator import _NO_RESPONSE_FALLBACK, Orchestrator
from aria.llm.base import ChatResult
from aria.tools.base import ToolRegistry


async def test_desktop_notify_is_branded(monkeypatch):
    import desktop_notifier

    recorded: dict = {}

    class FakeNotifier:
        def __init__(self, **kwargs):
            recorded["init"] = kwargs

        async def send(self, **kwargs):
            recorded["send"] = kwargs

    monkeypatch.setattr(desktop_notifier, "DesktopNotifier", FakeNotifier)

    from aria.core.scheduler import desktop_notify

    await desktop_notify("Aria", "your laundry timer is up")

    assert recorded["init"]["app_name"] == "Aria"  # not "python"
    assert recorded["init"].get("app_icon")  # an Aria icon is supplied
    assert recorded["send"]["title"] == "Aria"
    assert recorded["send"]["message"] == "your laundry timer is up"
    # FIX B: sound must NOT be passed as a bool (newer desktop_notifier crashes).
    assert not isinstance(recorded["send"].get("sound"), bool)
    assert "sound" not in recorded["send"]


async def test_desktop_notify_never_raises_on_backend_error(monkeypatch):
    # FIX B: the exact failure that crashed timers — a send that raises must be
    # swallowed, never propagated, so a timer firing is never broken by it.
    import desktop_notifier

    class ExplodingNotifier:
        def __init__(self, **kwargs):
            pass

        async def send(self, **kwargs):
            raise AttributeError("'bool' object has no attribute 'is_named'")

    monkeypatch.setattr(desktop_notifier, "DesktopNotifier", ExplodingNotifier)

    from aria.core.scheduler import desktop_notify

    # Must return cleanly (no exception).
    assert await desktop_notify("Aria", "ping") is None


async def test_turn_never_ends_silent():
    # FIX 2: a turn that would otherwise yield nothing must still SAY something.
    class SilentLLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            return ChatResult(content='{"route":"chitchat","needs_tools":[],"reason":"x"}')

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            return
            yield ""  # pragma: no cover - makes this an (empty) async generator

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=SilentLLM(), registry=ToolRegistry(), memory=mem,
        reasoning_model="big", fast_model="small",
    )
    spoken = "".join([d async for d in orch.respond("uhhh")])
    assert spoken.strip() == _NO_RESPONSE_FALLBACK
    # And it was persisted as the assistant turn.
    assert (await mem.recent_turns())[-1] == ("assistant", _NO_RESPONSE_FALLBACK)
    await mem.close()
