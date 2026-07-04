"""Agentic food ordering: guardrail always injected, STOP-at-payment, profile
injection + address redaction, Gemini engine (no Groq), and the commerce safety
tier. All browser interaction is mocked — no Playwright/Chromium/network."""

from __future__ import annotations

import pytest

import aria.agents.web_agent as web_agent
from aria.agents.web_agent import GUARDRAIL, compose_task
from aria.config.schema import AriaConfig, CommerceConfig
from aria.core.executor import ExecConfig, ToolExecutor
from aria.safety.audit import AuditLog
from aria.tools.base import Tool, ToolError, ToolResult
from aria.tools.commerce import BrowseWebTool, OrderFoodTool, build_order_task


def _engine() -> dict:
    return {"engine": "gemini", "base_url": "https://gem/v1",
            "model": "gemini-2.0-flash", "api_key": "k"}


# --- the guardrail is non-negotiable --------------------------------------
def test_guardrail_is_always_injected_first():
    t = compose_task("buy me a pizza")
    assert t.index(GUARDRAIL) == 0  # ALWAYS prepended, ahead of the task
    assert "buy me a pizza" in t


def test_guardrail_forbids_payment():
    assert "NEVER submit payment" in GUARDRAIL
    assert "Place order" in GUARDRAIL and "Pay" in GUARDRAIL
    assert "STOP" in GUARDRAIL


# --- reaching "Pay" stops with a summary, never a pay click ---------------
async def test_reaching_payment_stops_with_summary(monkeypatch):
    captured: dict = {}

    async def fake_run(composed_task, engine, **kw):
        captured["task"] = composed_task
        # The agent navigated all the way to the Pay page and STOPPED there.
        return {
            "summary": "At checkout", "shop": "Tony's Pizza",
            "items": "1x large pepperoni", "total": "£14.50", "eta": "35 min",
            "stopped_at_payment": True,
        }

    monkeypatch.setattr(web_agent, "_display_available", lambda: True)
    monkeypatch.setattr(web_agent, "_run_agent", fake_run)
    monkeypatch.setattr("aria.tools.commerce.bring_browser_to_front", lambda: None)

    tool = OrderFoodTool(lambda: CommerceConfig(delivery_address="10 Foo St"), _engine)
    res = await tool.run(request="a large pepperoni")

    assert "Tony's Pizza" in res.content and "£14.50" in res.content
    assert "payment" in res.content.lower()  # stopped AT payment, didn't pay
    assert res.data["stopped_at_payment"] is True
    # The browser was driven under the guardrail (no-pay) the whole time.
    assert GUARDRAIL in captured["task"]


# --- profile injected into the task; address redacted from the audit ------
async def test_profile_injected_and_address_redacted(tmp_path, monkeypatch):
    captured: dict = {}

    async def fake_run(composed_task, engine, **kw):
        captured["task"] = composed_task
        return {"summary": "At checkout", "shop": "Veg Place", "total": "£9"}

    monkeypatch.setattr(web_agent, "_display_available", lambda: True)
    monkeypatch.setattr(web_agent, "_run_agent", fake_run)
    monkeypatch.setattr("aria.tools.commerce.bring_browser_to_front", lambda: None)

    cfg = CommerceConfig(
        delivery_address="221B Baker Street",
        dietary_prefs="vegetarian",
        favorite_vendors=["Tony's"],
    )
    tool = OrderFoodTool(lambda: cfg, _engine)

    audit_path = tmp_path / "audit.log"
    execu = ToolExecutor(AuditLog(audit_path), ExecConfig(require_confirmation=False))

    async def yes(_name):
        return True

    await execu.execute(tool, {"request": "a margherita to 221B Baker Street"}, confirm=yes)

    # The profile reached the (local) browser task...
    assert "221B Baker Street" in captured["task"]
    assert "vegetarian" in captured["task"] and "Tony's" in captured["task"]
    # ...but the audit trail NEVER records the address or the order text.
    logged = audit_path.read_text()
    assert "221B Baker Street" not in logged
    assert "margherita" not in logged
    assert "_redacted_fields" in logged  # only field names, not values
    assert '"risk": "commerce"' in logged


# --- the engine is Gemini OpenAI-compat with no Groq dependency ------------
def test_commerce_engine_is_gemini_and_keyless_of_groq():
    from aria.app import commerce_engine

    asked: list[str] = []

    class _Secrets:
        def get(self, name):
            asked.append(name)
            return "gem-key" if name == "commerce_api_key" else None

    eng = commerce_engine(AriaConfig(), _Secrets())  # engine defaults to gemini
    assert "generativelanguage.googleapis.com" in eng["base_url"]
    assert eng["model"] == "gemini-2.0-flash"
    assert eng["api_key"] == "gem-key"
    assert "commerce_api_key" in asked
    assert "groq_api_key" not in asked  # never touches the Groq budget/key


# --- the commerce safety tier --------------------------------------------
async def test_commerce_tier_declined_without_confirm_callback(tmp_path):
    class SpyOrder(Tool):
        name = "order_food"
        description = "x"
        risk = "commerce"

        def __init__(self):
            self.ran = False

        async def run(self, **kwargs):
            self.ran = True
            return ToolResult(content="ran")

    tool = SpyOrder()
    execu = ToolExecutor(AuditLog(tmp_path / "a.log"), ExecConfig(require_confirmation=True))
    res = await execu.execute(tool, {"request": "pizza"}, confirm=None)  # no callback
    assert tool.ran is False  # commerce is NEVER auto-approved
    assert res.content == "user declined"


def test_browse_web_is_confirm_and_order_food_is_commerce():
    assert BrowseWebTool(_engine).risk == "confirm"
    assert OrderFoodTool(lambda: CommerceConfig(), _engine).risk == "commerce"
    assert OrderFoodTool(lambda: CommerceConfig(), _engine).sensitive is True


# --- graceful degradation -------------------------------------------------
async def test_order_food_requires_address(monkeypatch):
    monkeypatch.setattr(web_agent, "_display_available", lambda: True)
    tool = OrderFoodTool(lambda: CommerceConfig(), _engine)  # no address set
    with pytest.raises(ToolError, match="address"):
        await tool.run(request="a pizza")


async def test_no_display_says_it_needs_the_screen(monkeypatch):
    monkeypatch.setattr(web_agent, "_display_available", lambda: False)
    tool = OrderFoodTool(lambda: CommerceConfig(delivery_address="x"), _engine)
    with pytest.raises(ToolError, match="screen"):
        await tool.run(request="a pizza")


async def test_order_food_gated_and_narrated_through_orchestrator(tmp_path, monkeypatch):
    # End-to-end: a commerce tool is gated by the orchestrator (asks, runs only on
    # yes) and its slow_filler narrates the browse after the yes.
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ChatResult, ToolCall
    from aria.tools.base import ToolRegistry
    from tests.conftest import FakeLLM

    ran = {"v": False}

    class FakeOrder(Tool):
        name = "order_food"
        description = "order food"
        risk = "commerce"
        slow_filler = "Finding you a good spot. "

        async def run(self, **kwargs):
            ran["v"] = True
            return ToolResult(content="At checkout for Tony's, total £14. Payment page open.")

    reg = ToolRegistry()
    reg.register(FakeOrder())
    route = ChatResult(content='{"route":"agentic","needs_tools":["order_food"],"reason":"x"}')
    call = ChatResult(content="", tool_calls=[ToolCall("c1", "order_food", {"request": "pizza"})])
    llm = FakeLLM(stream_text="Opening the payment page.", chat_queue=[route, call,
                  ChatResult(content='{"route":"x","needs_tools":[],"reason":"y"}')])
    mem = Memory(tmp_path / "m.sqlite3")
    await mem.open()
    orch = Orchestrator(llm=llm, registry=reg, memory=mem, reasoning_model="big",
                        fast_model="small")

    first = "".join([d async for d in orch.respond("order me a pizza")])
    assert "go ahead" in first.lower()  # asked for confirmation (varied frame)
    assert ran["v"] is False  # commerce NEVER runs before the yes

    second = "".join([d async for d in orch.respond("yes")])
    assert ran["v"] is True  # ran only after confirmation
    assert "Finding you a good spot" in second  # narrated during the slow browse
    await mem.close()


async def test_commerce_tools_register_in_the_full_registry(tmp_path):
    # build_registry must wire order_food/browse_web with their config+engine
    # providers, and importing them must NOT require the heavy browser deps.
    from aria.app import build_registry
    from aria.config.keyring import SecretStore
    from aria.core.memory import Memory
    from aria.core.scheduler import SchedulerService

    mem = Memory(":memory:")
    await mem.open()
    sch = SchedulerService(db_path=":memory:")
    await sch.open()
    reg, managers = await build_registry(
        AriaConfig(), llm=object(), memory=mem, scheduler=sch, secrets=SecretStore()
    )
    assert "order_food" in reg.names() and "browse_web" in reg.names()
    assert reg.get("order_food").risk == "commerce"
    for m in managers:
        await m.aclose()
    await mem.close()


def test_build_order_task_includes_profile_and_stop_rule():
    cfg = CommerceConfig(
        delivery_address="1 A Street", dietary_prefs="no nuts",
        favorite_vendors=["X", "Y"], default_food_app="Uber Eats", max_order_value=25,
    )
    t = build_order_task("a large pepperoni", cfg)
    assert "large pepperoni" in t
    assert "1 A Street" in t and "no nuts" in t and "Uber Eats" in t
    assert "X, Y" in t and "25" in t
    assert "STOP" in t and "do NOT pay" in t
