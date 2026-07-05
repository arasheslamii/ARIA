"""Real-world errands (0.5.0): flights, hotels, shopping — deep links opened in
the USER'S default browser, where they (and only they) approve and pay."""

from __future__ import annotations

import pytest

from aria.tools.base import ToolError
from aria.tools.errands import (
    BookFlightTool,
    BookHotelTool,
    OpenInBrowserTool,
    ShopOnlineTool,
    errand_tools,
    open_in_default_browser,
)


class SpyOpener:
    def __init__(self) -> None:
        self.opened: list[str] = []

    async def __call__(self, url: str) -> None:
        self.opened.append(url)


async def test_book_flight_opens_google_flights_deep_link():
    opener = SpyOpener()
    res = await BookFlightTool(opener).run(
        origin="Edinburgh", destination="London", depart_date="2026-07-10"
    )
    assert len(opener.opened) == 1
    url = opener.opened[0]
    assert url.startswith("https://www.google.com/travel/flights?q=")
    assert "Edinburgh" in url and "London" in url and "2026-07-10" in url
    assert "one+way" in url
    # The model must never be told a booking happened.
    assert "Nothing is booked or paid yet" in res.content
    assert "book it" in (res.spoken or "")  # the USER books, on their screen


async def test_book_flight_return_trip_and_bad_dates():
    opener = SpyOpener()
    tool = BookFlightTool(opener)
    res = await tool.run(
        origin="EDI", destination="LHR",
        depart_date="2026-07-10", return_date="2026-07-14", adults=2,
    )
    assert "returning" in opener.opened[0] and "2026-07-14" in opener.opened[0]
    assert "return" in res.content

    with pytest.raises(ToolError, match="ISO date"):
        await tool.run(origin="EDI", destination="LHR", depart_date="July 10")
    with pytest.raises(ToolError, match="not a real calendar date"):
        await tool.run(origin="EDI", destination="LHR", depart_date="2026-02-30")
    with pytest.raises(ToolError, match="before depart_date"):
        await tool.run(
            origin="EDI", destination="LHR",
            depart_date="2026-07-10", return_date="2026-07-01",
        )
    assert len(opener.opened) == 1  # none of the failures opened anything


async def test_book_hotel_builds_booking_dot_com_search():
    opener = SpyOpener()
    res = await BookHotelTool(opener).run(
        destination="London", checkin="2026-07-10", checkout="2026-07-12"
    )
    url = opener.opened[0]
    assert url.startswith("https://www.booking.com/searchresults.html?")
    assert "ss=London" in url
    assert "checkin=2026-07-10" in url and "checkout=2026-07-12" in url
    assert "group_adults=2" in url and "no_rooms=1" in url  # sane defaults
    assert "2 nights" in res.content
    assert "Nothing is booked or paid yet" in res.content

    with pytest.raises(ToolError, match="after checkin"):
        await BookHotelTool(opener).run(
            destination="London", checkin="2026-07-12", checkout="2026-07-12"
        )


async def test_shop_online_opens_shopping_results():
    opener = SpyOpener()
    res = await ShopOnlineTool(opener).run(query="usb-c hub 4 ports")
    assert "tbm=shop" in opener.opened[0]
    assert "usb-c+hub" in opener.opened[0]
    assert "Nothing is booked or paid yet" in res.content


async def test_open_in_browser_allows_only_http_links():
    opener = SpyOpener()
    tool = OpenInBrowserTool(opener)
    res = await tool.run(url="https://example.com/checkout", purpose="finish the booking")
    assert opener.opened == ["https://example.com/checkout"]
    assert "finish the booking" in (res.spoken or "")

    for bad in ("javascript:alert(1)", "file:///etc/passwd", "ftp://x", ""):
        with pytest.raises(ToolError):
            await tool.run(url=bad)
    assert len(opener.opened) == 1


async def test_errand_tools_are_ungated_and_never_pay():
    tools = errand_tools(SpyOpener())
    assert {t.name for t in tools} == {
        "open_in_browser", "book_flight", "book_hotel", "shop_online",
    }
    # Opening a page spends nothing — no confirmation friction on any of them.
    assert all(t.risk == "safe" for t in tools)


async def test_opener_requires_a_display(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(ToolError, match="screen"):
        await open_in_default_browser("https://example.com")


def test_errands_specialist_is_registered():
    from aria.agents.specialists import build_specialists
    from tests.conftest import FakeLLM

    agents = build_specialists(FakeLLM(), "big", "small", [])
    names = {a.name for a in agents}
    assert "agent_errands" in names
    errands = next(a for a in agents if a.name == "agent_errands")
    inner = errands._agent._registry.names()
    assert {"book_flight", "book_hotel", "shop_online", "open_in_browser"} <= set(inner)
    assert "never book" in errands._agent.system_prompt


def test_orchestrator_prompt_teaches_the_last_click_rule():
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    assert "book_flight" in ORCHESTRATOR_SYSTEM
    assert "book_hotel" in ORCHESTRATOR_SYSTEM
    assert "shop_online" in ORCHESTRATOR_SYSTEM
    assert "open_in_browser" in ORCHESTRATOR_SYSTEM
    # The golden rule is spelled out for the model.
    assert "never book, buy, or pay" in ORCHESTRATOR_SYSTEM


# --- agentic reservations (0.8.0): pick the best, stop before payment --------
from aria.agents import web_agent  # noqa: E402
from aria.config.schema import CommerceConfig  # noqa: E402
from aria.tools.commerce import (  # noqa: E402
    ReserveFlightTool,
    ReserveHotelTool,
    build_flight_task,
    build_hotel_task,
)


def test_hotel_task_encodes_budget_preferences_and_the_stop_rule():
    task = build_hotel_task(
        "Paris", "2026-07-10", "2026-07-15",
        budget="100 pounds", preferences="near the centre, breakfast included",
    )
    assert "Paris" in task and "2026-07-10" in task and "2026-07-15" in task
    assert "5 nights" in task
    assert "HARD budget: 100 pounds" in task
    assert "near the centre" in task
    assert "booking.com" in task  # starts from the deterministic deep link
    assert "STOP at the guest-details/payment step" in task
    assert "NO personal or card details" in task
    with pytest.raises(ToolError, match="after checkin"):
        build_hotel_task("Paris", "2026-07-15", "2026-07-10")


def test_flight_task_encodes_route_budget_and_the_stop_rule():
    task = build_flight_task(
        "Edinburgh", "London", "2026-07-10", budget="£80", preferences="morning, nonstop"
    )
    assert "Edinburgh" in task and "London" in task and "2026-07-10" in task
    assert "HARD budget: £80" in task and "nonstop" in task
    assert "google.com/travel/flights" in task
    assert "STOP there" in task and "NO passenger or payment details" in task


def _commerce_cfg():
    return CommerceConfig(headful=True, max_steps=40, max_seconds=240.0)


async def test_reserve_hotel_reports_the_pick_and_never_claims_booked(monkeypatch):
    seen = {}

    async def fake_run(task, engine, **kwargs):
        seen["task"] = task
        return {
            "summary": "Hotel Le Petit, 8.6, double room, £480 total, free cancellation",
            "stopped_at_payment": True,
        }

    monkeypatch.setattr(web_agent, "_display_available", lambda: True)
    monkeypatch.setattr(web_agent, "_run_agent", fake_run)
    monkeypatch.setattr("aria.tools.commerce.bring_browser_to_front", lambda: None)

    tool = ReserveHotelTool(_commerce_cfg, lambda: {"model": "m", "base_url": "u"})
    res = await tool.run(
        destination="Paris", checkin="2026-07-10", checkout="2026-07-15",
        budget="100 pounds",
    )
    assert web_agent.GUARDRAIL in seen["task"]  # the never-pay rule is injected
    assert "Hotel Le Petit" in res.content
    assert "NOTHING is reserved or paid yet" in res.content
    # Confirmation read-back carries the criteria.
    summary = tool.confirm_summary({"destination": "Paris", "budget": "100 pounds"})
    assert "Paris" in summary and "100 pounds" in summary and "stop before" in summary


async def test_reserve_flight_stuck_hands_over_honestly(monkeypatch):
    async def fake_run(task, engine, **kwargs):
        return {"stuck": "a captcha", "summary": "reached the airline page"}

    monkeypatch.setattr(web_agent, "_display_available", lambda: True)
    monkeypatch.setattr(web_agent, "_run_agent", fake_run)
    monkeypatch.setattr("aria.tools.commerce.bring_browser_to_front", lambda: None)

    tool = ReserveFlightTool(_commerce_cfg, lambda: {"model": "m", "base_url": "u"})
    res = await tool.run(origin="EDI", destination="LHR", depart_date="2026-07-10")
    assert res.content.startswith("stuck: a captcha")
    assert "take over" in (res.spoken or "")


def test_reserve_tools_are_confirm_gated_slow_and_in_the_prompt():
    from aria.core.orchestrator import _SLOW_TOOLS
    from aria.core.prompts import ORCHESTRATOR_SYSTEM

    hotel = ReserveHotelTool(_commerce_cfg, dict)
    flight = ReserveFlightTool(_commerce_cfg, dict)
    assert hotel.risk == "confirm" and flight.risk == "confirm"
    assert hotel.slow_filler and flight.slow_filler  # no dead air for minutes
    assert {"reserve_hotel", "reserve_flight"} <= _SLOW_TOOLS
    assert "reserve_hotel" in ORCHESTRATOR_SYSTEM


# --- 0.8.2: the reserve loop-of-confirmations bug ----------------------------
async def test_executor_lets_long_tools_override_the_quick_timeout():
    import asyncio as aio

    from aria.core.executor import ExecConfig, ToolExecutor
    from aria.safety.audit import AuditLog
    from aria.tools.base import Tool, ToolResult

    class SlowBrowse(Tool):
        name = "slow_browse"
        description = "x"
        timeout_s = 5.0  # declares itself long-running

        async def run(self, **kwargs):
            await aio.sleep(0.3)  # longer than the executor default below
            return ToolResult(content="done")

    ex = ToolExecutor(AuditLog(), ExecConfig(timeout_s=0.05, retries=0,
                                             require_confirmation=False))
    res = await ex.execute(SlowBrowse(), {})
    assert res.content == "done"  # NOT killed at the quick-lookup timeout


def test_browser_tools_outlive_the_executor_default():
    from aria.core.executor import ExecConfig
    from aria.tools.commerce import BrowseWebTool, OrderFoodTool

    default = ExecConfig().timeout_s
    for tool in (
        BrowseWebTool(dict),
        OrderFoodTool(_commerce_cfg, dict),
        ReserveHotelTool(_commerce_cfg, dict),
        ReserveFlightTool(_commerce_cfg, dict),
    ):
        # Must exceed both the executor default AND the browse budget itself.
        assert tool.timeout_s > default
        assert tool.timeout_s > _commerce_cfg().max_seconds


async def test_failed_confirmed_tool_is_never_reconfirmed_in_a_loop(tmp_path):
    """The user's exact bug: reserve_hotel fails after the yes; the model retries
    the same call, which re-triggered the confirmation question — forever."""
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ChatResult, ToolCall
    from aria.tools.base import Tool, ToolRegistry
    from tests.conftest import FakeLLM

    class ExplodingReserve(Tool):
        name = "reserve_hotel"
        description = "reserve"
        risk = "confirm"

        def __init__(self) -> None:
            self.runs = 0

        async def run(self, **kwargs):
            self.runs += 1
            raise ToolError("the browsing engine is not installed")

    reg = ToolRegistry()
    tool = ExplodingReserve()
    reg.register(tool)
    call = ToolCall("c1", "reserve_hotel", {"destination": "Paris"})
    retry = ToolCall("c2", "reserve_hotel", {"destination": "Paris"})
    llm = FakeLLM(
        stream_text="I couldn't run the booking browser — the engine failed.",
        chat_queue=[
            ChatResult(content='{"route":"tool","needs_tools":["reserve_hotel"],"reason":"x"}'),
            ChatResult(content="", tool_calls=[call]),   # first plan -> gated
            ChatResult(content="", tool_calls=[retry]),  # after failure: same call!
            ChatResult(content="It failed."),            # forced honest wrap-up
        ],
    )
    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(llm=llm, registry=reg, memory=mem,
                        reasoning_model="big", fast_model="small")

    first = "".join([d async for d in orch.respond("book the best hotel in paris")])
    assert "go ahead" in first.lower()  # asked ONCE

    second = "".join([d async for d in orch.respond("yes")])
    assert tool.runs == 1  # ran once; the identical retry was refused
    assert "go ahead" not in second.lower()  # and NEVER re-asked
    assert orch._pending is None  # turn finished cleanly
    assert "engine failed" in second  # the failure was reported out loud
    await mem.close()


# --- 0.8.3: no dead air after "yes", consent walls, visible failures ---------
def test_browser_tasks_deal_with_consent_walls_first():
    # Booking.com's GDPR dialog blocks EVERYTHING; the agent must be told.
    assert "cookie/consent" in build_hotel_task("Paris", "2026-07-10", "2026-07-12")
    assert "cookie/consent" in build_flight_task("EDI", "LHR", "2026-07-10")


def test_plain_go_ahead_is_instant_but_amendments_still_classify():
    from aria.core.orchestrator import _is_plain_go_ahead

    for t in ("yes, find whatever you can", "sure go for it", "yeah do it please",
              "okay whatever looks good"):
        assert _is_plain_go_ahead(t), t
    for t in ("yes but make it two nights", "yes, actually change the dates",
              "no, find something cheaper", "yes add breakfast",
              "leave it on channel four"):
        assert not _is_plain_go_ahead(t), t


async def test_long_emphatic_yes_needs_no_llm_and_runs_immediately(tmp_path):
    """The live bug: 'yes find whatever you can' went to the confirmation
    classifier — minutes of silence on the local fallback before anything ran."""
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ChatResult, ToolCall
    from aria.tools.base import Tool, ToolRegistry, ToolResult
    from tests.conftest import FakeLLM

    class SpyReserve(Tool):
        name = "reserve_hotel"
        description = "reserve"
        risk = "confirm"

        def __init__(self) -> None:
            self.runs = 0

        async def run(self, **kwargs):
            self.runs += 1
            return ToolResult(content="reserved up to the payment step")

    reg = ToolRegistry()
    tool = SpyReserve()
    reg.register(tool)
    llm = FakeLLM(
        stream_text="All set — it's on your screen.",
        chat_queue=[
            ChatResult(content='{"route":"tool","needs_tools":["reserve_hotel"],"reason":"x"}'),
            ChatResult(content="", tool_calls=[ToolCall("c1", "reserve_hotel", {})]),
            # NOTE: no classifier entry — if the yes went to the LLM, this queue
            # would be consumed out of order and the test would fail.
            ChatResult(content="Done — ready to pay."),
        ],
    )
    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(llm=llm, registry=reg, memory=mem,
                        reasoning_model="big", fast_model="small")
    "".join([d async for d in orch.respond("book the best hotel in paris")])
    out = "".join([d async for d in orch.respond("yes, find whatever you can")])
    assert tool.runs == 1  # executed straight away, no classify round-trip
    assert "go ahead" not in out.lower()
    await mem.close()


async def test_switch_to_local_brain_is_announced_once(monkeypatch):
    import aria.llm.local_fallback as lf
    from aria.llm.local_fallback import LocalFallbackProvider
    from aria.llm.ollama import ModelInfo
    from tests.test_local_fallback import LimitedCloud, LocalSpy

    monkeypatch.setattr(lf, "detect_ollama",
                        lambda base: [ModelInfo(name="qwen2.5:3b", params_b=3.1)])
    provider = LocalFallbackProvider(LimitedCloud(), fast_model="fast", local=LocalSpy())
    switches = []
    provider.on_switch = lambda: switches.append(1)
    await provider.chat([], model="big")
    await provider.chat([], model="big")
    assert switches == [1]  # told the user ONCE, not every call
