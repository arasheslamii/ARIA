"""Agentic web tools: read-only browsing (Stage 1) and food ordering (Stage 2).

Both drive a real Chromium via :mod:`aria.agents.web_agent`, which ALWAYS stops at
the payment page. ``order_food`` is ``risk="commerce"`` (a tier above confirm) and
``sensitive`` (its arguments — which may contain the delivery address — are redacted
from the audit trail, like the Gmail tools). No card data is ever handled or stored;
the human pays on the live page.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aria.agents.web_agent import (
    WebAgentError,
    bring_browser_to_front,
    run_web_task,
)
from aria.config.schema import CommerceConfig
from aria.tools.base import Tool, ToolError, ToolResult
from aria.tools.errands import _parse_date, flight_search_url, hotel_search_url

EngineProvider = Callable[[], dict]
ConfigProvider = Callable[[], CommerceConfig]


def build_order_task(request: str, cfg: CommerceConfig) -> str:
    """Turn a spoken request + the saved profile into a concrete browser task. The
    address/dietary/vendor details are injected here (used by the LOCAL browser LLM;
    never spoken back, and redacted from the audit trail)."""
    lines = [f"Order food delivery: {request.strip()}."]
    if cfg.default_food_app:
        lines.append(f"Prefer the {cfg.default_food_app} site/app.")
    if cfg.favorite_vendors:
        lines.append(
            "Prefer these vendors if one has what's asked and is open and well-rated: "
            + ", ".join(cfg.favorite_vendors)
            + ". Otherwise pick a well-rated, currently-open shop."
        )
    else:
        lines.append("Pick a well-rated, currently-open shop.")
    if cfg.dietary_prefs:
        lines.append(f"Respect these dietary preferences: {cfg.dietary_prefs}.")
    if cfg.delivery_address:
        lines.append(f"Deliver to: {cfg.delivery_address}.")
    if cfg.max_order_value:
        lines.append(
            f"Keep the order total at or below {cfg.max_order_value:g}; if nothing "
            "suitable fits, STOP and report that instead."
        )
    lines.append(
        "Build the cart, enter the delivery address, and go to the checkout/payment "
        "page. STOP there — do NOT pay. Report the shop, items, total, and ETA."
    )
    return " ".join(lines)


class BrowseWebTool(Tool):
    name = "browse_web"
    description = (
        "Open a REAL web browser and navigate, search, sign in, or read a site for "
        "the user (no purchases). Use for 'check X on the website', 'log me into Y', "
        "'look something up on a specific site'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {"type": "string", "description": "What to do in the browser."}
        },
        "required": ["task"],
    }
    risk = "confirm"  # opening a real browser session is an explicit, gated action
    slow_filler = "Sure — opening the browser and taking a look. One moment."

    def __init__(self, engine_provider: EngineProvider) -> None:
        self._engine = engine_provider

    async def run(self, **kwargs: Any) -> ToolResult:
        task = str(kwargs.get("task") or "").strip()
        if not task:
            raise ToolError("What would you like me to look up in the browser?")
        try:
            result = await run_web_task(task, self._engine(), headful=True)
        except WebAgentError as exc:
            raise ToolError(str(exc)) from exc
        if result.stuck:
            return ToolResult(
                content=f"stuck: {result.stuck}. {result.summary}".strip(),
                spoken=f"I got stuck — {result.stuck}. I've left the browser open for you.",
                data={"stuck": result.stuck, "url": result.url},
            )
        return ToolResult(content=result.summary or "Done.", data={"url": result.url})


class OrderFoodTool(Tool):
    name = "order_food"
    description = (
        "Order food, coffee, or drink delivery for the user: find a good, open, "
        "well-rated shop, build the cart from the request and saved profile, enter "
        "the delivery address, go to checkout, and STOP before paying so the user "
        "can pay. Use for 'order a pizza', 'get me a coffee', 'get me food', "
        "'order a large pepperoni'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "What to order, e.g. 'a large pepperoni pizza'.",
            }
        },
        "required": ["request"],
    }
    risk = "commerce"  # spends money / places a real order — gated, never auto-run
    sensitive = True   # arguments (may include address) redacted from the audit log
    slow_filler = (
        "Okay — finding you a good spot and building your order. "
        "This takes a moment, and I'll stop before paying."
    )

    def __init__(self, config_provider: ConfigProvider, engine_provider: EngineProvider) -> None:
        self._config = config_provider
        self._engine = engine_provider

    def confirm_summary(self, arguments: dict[str, Any]) -> str:
        request = str(arguments.get("request") or "food")
        cfg = self._config()
        where = f" to {cfg.delivery_address}" if cfg.delivery_address else ""
        return f"order {request}{where} and stop at the payment page before paying"

    async def run(self, **kwargs: Any) -> ToolResult:
        request = str(kwargs.get("request") or "").strip()
        if not request:
            raise ToolError("What would you like to order?")
        cfg = self._config()
        if not cfg.delivery_address:
            raise ToolError(
                "I don't have your delivery address yet — set it in `aria setup` "
                "under the delivery profile, then ask again."
            )
        task = build_order_task(request, cfg)
        try:
            result = await run_web_task(
                task,
                self._engine(),
                headful=cfg.headful,
                max_steps=cfg.max_steps,
                max_seconds=cfg.max_seconds,
            )
        except WebAgentError as exc:
            raise ToolError(str(exc)) from exc

        if result.stuck:
            bring_browser_to_front()
            return ToolResult(
                content=f"stuck: {result.stuck}. {result.summary}".strip(),
                spoken=(
                    f"I got stuck — {result.stuck}. I've left the browser open at "
                    "that point so you can take over."
                ),
                data={"stuck": result.stuck},
            )

        # Reached checkout: surface the live browser at the payment page for the human.
        bring_browser_to_front()
        shop = result.shop or "the shop"
        bits = [f"At checkout for {shop}"]
        if result.items:
            bits.append(f"items: {result.items}")
        if result.total:
            bits.append(f"total {result.total}")
        if result.eta:
            bits.append(f"ETA {result.eta}")
        content = (
            ". ".join(bits)
            + ". Stopped before payment — the payment page is open for you to pay."
        )
        return ToolResult(
            content=content,
            spoken=(
                f"All set at {shop}"
                + (f" — {result.total}" if result.total else "")
                + ". Opening the payment page; pay whenever you're ready."
            ),
            # Vendor/items/total only — never card data. (Args are redacted via
            # sensitive=True; this result content isn't written to the audit log.)
            data={
                "shop": result.shop,
                "items": result.items,
                "total": result.total,
                "eta": result.eta,
                "stopped_at_payment": result.stopped_at_payment,
            },
        )


def build_hotel_task(
    destination: str, checkin: str, checkout: str,
    budget: str | None = None, preferences: str | None = None,
) -> str:
    """The browser task for an agentic hotel reservation: filter, pick the BEST
    fit, select a room, and stop at the final step — the human confirms and pays."""
    ci = _parse_date(checkin, "checkin")
    co = _parse_date(checkout, "checkout")
    if co <= ci:
        raise ToolError("checkout must be after checkin — check the dates.")
    lines = [
        f"Reserve a hotel room in {destination}: check-in {ci.isoformat()}, "
        f"check-out {co.isoformat()} ({(co - ci).days} nights).",
        f"Start at: {hotel_search_url(destination, ci, co)}",
    ]
    if budget:
        lines.append(
            f"HARD budget: {budget} per night — never pick anything above it; if "
            "nothing decent fits, STOP and report the cheapest good option instead."
        )
    if preferences:
        lines.append(f"The user wants: {preferences}.")
    lines.append(
        "Use the site's own filters and sorting (price, review score). Pick the "
        "BEST option: the highest-rated property that fits the budget in a sensible "
        "location; prefer free cancellation and a rating of 8+ when available. Open "
        "it, choose the cheapest room type that fits the whole stay, and click "
        "through the reservation flow (Reserve / See availability). STOP at the "
        "guest-details/payment step — enter NO personal or card details and confirm "
        "nothing. Report: hotel name, review score, room type, TOTAL price for the "
        "stay, and whether cancellation is free."
    )
    return " ".join(lines)


def build_flight_task(
    origin: str, destination: str, depart_date: str, return_date: str | None = None,
    budget: str | None = None, preferences: str | None = None,
) -> str:
    """The browser task for an agentic flight pick: compare, choose the best
    price/duration fit, and stop at the airline handoff — the human pays."""
    dep = _parse_date(depart_date, "depart_date")
    ret = _parse_date(return_date, "return_date") if return_date else None
    if ret is not None and ret < dep:
        raise ToolError("return_date is before depart_date — check the dates.")
    lines = [
        f"Pick the best flight from {origin} to {destination}, departing "
        f"{dep.isoformat()}" + (f", returning {ret.isoformat()}" if ret else " (one-way)") + ".",
        f"Start at: {flight_search_url(origin, destination, dep, ret)}",
    ]
    if budget:
        lines.append(f"HARD budget: {budget} total — never pick a flight above it.")
    if preferences:
        lines.append(f"The user wants: {preferences}.")
    lines.append(
        "Pick the BEST option: balance price, total duration, and stops (the top "
        "'Best' ranked result usually wins; prefer nonstop when the price is close). "
        "Select it and continue to the booking handoff (the airline's or agent's "
        "booking page). STOP there — enter NO passenger or payment details and "
        "confirm nothing. Report: airline, departure and arrival times, duration, "
        "stops, and the price."
    )
    return " ".join(lines)


class ReserveHotelTool(Tool):
    name = "reserve_hotel"
    description = (
        "Agentically PICK the best hotel and get it ready to reserve: drives a real "
        "browser, applies the user's budget/preferences, chooses the best-rated fit, "
        "selects a room, and STOPS at the final reservation step so the user "
        "confirms and pays. Use when the user gives criteria (a budget, 'best', "
        "'cheapest', stars, area) or asks you to choose for them. Dates ISO "
        "YYYY-MM-DD. For just showing options, use book_hotel instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "City / area."},
            "checkin": {"type": "string", "description": "Check-in date, YYYY-MM-DD."},
            "checkout": {"type": "string", "description": "Check-out date, YYYY-MM-DD."},
            "budget": {
                "type": "string",
                "description": "Max price per night, as said: '100 pounds', '€80'.",
            },
            "preferences": {
                "type": "string",
                "description": "Anything else that matters: area, stars, breakfast…",
            },
        },
        "required": ["destination", "checkin", "checkout"],
    }
    risk = "confirm"  # a long real-browser session on the user's machine
    slow_filler = (
        "On it — comparing hotels and setting the best one up. This takes a couple "
        "of minutes, and the final click stays yours. "
    )

    def __init__(self, config_provider: ConfigProvider, engine_provider: EngineProvider) -> None:
        self._config = config_provider
        self._engine = engine_provider

    def confirm_summary(self, arguments: dict[str, Any]) -> str:
        where = arguments.get("destination", "there")
        budget = arguments.get("budget")
        extra = f" within {budget} a night" if budget else ""
        return (
            f"find the best hotel in {where}{extra}, set it up ready to reserve, "
            "and stop before any payment"
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        destination = str(kwargs.get("destination") or "").strip()
        if not destination:
            raise ToolError("Where should the hotel be?")
        task = build_hotel_task(
            destination, kwargs.get("checkin"), kwargs.get("checkout"),
            budget=kwargs.get("budget"), preferences=kwargs.get("preferences"),
        )
        return await _run_reservation(task, self._config(), self._engine(), kind="hotel")


class ReserveFlightTool(Tool):
    name = "reserve_flight"
    description = (
        "Agentically PICK the best flight and get it ready to book: drives a real "
        "browser, compares options against the user's budget/preferences, chooses "
        "the best price/duration fit, and STOPS at the airline's booking page so "
        "the user confirms and pays. Use when the user gives criteria or asks you "
        "to choose. Dates ISO YYYY-MM-DD. For just showing options, use "
        "book_flight instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Departure city or airport."},
            "destination": {"type": "string", "description": "Arrival city or airport."},
            "depart_date": {"type": "string", "description": "Departure date, YYYY-MM-DD."},
            "return_date": {
                "type": "string", "description": "Return date YYYY-MM-DD; omit for one-way.",
            },
            "budget": {"type": "string", "description": "Max total price, as said."},
            "preferences": {
                "type": "string",
                "description": "Anything else: nonstop, morning, airline, cabin…",
            },
        },
        "required": ["origin", "destination", "depart_date"],
    }
    risk = "confirm"
    slow_filler = (
        "Alright — comparing flights and lining up the best one. A couple of "
        "minutes, and you make the final call. "
    )

    def __init__(self, config_provider: ConfigProvider, engine_provider: EngineProvider) -> None:
        self._config = config_provider
        self._engine = engine_provider

    def confirm_summary(self, arguments: dict[str, Any]) -> str:
        route = f"{arguments.get('origin', '?')} to {arguments.get('destination', '?')}"
        budget = arguments.get("budget")
        extra = f" within {budget}" if budget else ""
        return (
            f"find the best flight from {route}{extra}, take it to the airline's "
            "booking page, and stop before any payment"
        )

    async def run(self, **kwargs: Any) -> ToolResult:
        origin = str(kwargs.get("origin") or "").strip()
        destination = str(kwargs.get("destination") or "").strip()
        if not origin or not destination:
            raise ToolError("I need both where you're flying from and to.")
        task = build_flight_task(
            origin, destination, kwargs.get("depart_date"),
            return_date=kwargs.get("return_date"),
            budget=kwargs.get("budget"), preferences=kwargs.get("preferences"),
        )
        return await _run_reservation(task, self._config(), self._engine(), kind="flight")


async def _run_reservation(
    task: str, cfg: CommerceConfig, engine: dict, *, kind: str
) -> ToolResult:
    """Shared run/report path for the agentic reserve tools. Mirrors order_food:
    stuck → honest handoff; success → surface the live browser and report the
    pick, always stressing that nothing is booked or paid."""
    try:
        result = await run_web_task(
            task, engine,
            headful=cfg.headful, max_steps=cfg.max_steps, max_seconds=cfg.max_seconds,
        )
    except WebAgentError as exc:
        raise ToolError(str(exc)) from exc

    if result.stuck:
        bring_browser_to_front()
        return ToolResult(
            content=f"stuck: {result.stuck}. {result.summary}".strip(),
            spoken=(
                f"I got stuck — {result.stuck}. I've left the browser open at that "
                "point so you can take over."
            ),
            data={"stuck": result.stuck},
        )

    bring_browser_to_front()
    return ToolResult(
        content=(
            f"Best {kind} selected and taken to the final reservation step: "
            f"{result.summary or 'see the open browser page'}. "
            "NOTHING is reserved or paid yet — the page is open for the user to "
            "review, confirm, and pay themselves."
        ),
        data={"summary": result.summary, "stopped_at_payment": result.stopped_at_payment},
    )


def commerce_tools(
    config_provider: ConfigProvider, engine_provider: EngineProvider
) -> list[Tool]:
    return [
        BrowseWebTool(engine_provider),
        OrderFoodTool(config_provider, engine_provider),
        ReserveHotelTool(config_provider, engine_provider),
        ReserveFlightTool(config_provider, engine_provider),
    ]
