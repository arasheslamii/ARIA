"""Real-world errands finished in the user's OWN default browser.

The fast path for "book me a flight", "book a hotel", "buy me X": build the
right deep link (Google Flights / Booking.com / a shopping search), open it in
the user's default browser (xdg-open → Firefox/Chrome/whatever they actually
use), and let the HUMAN make the final selection and payment there. Nothing
here spends money or touches card data — these tools only open pages; the very
last click is always the user's.

Heavier cart-building flows (order_food, browse_web) live in
:mod:`aria.tools.commerce` and drive the agent's own Chromium instead, because
a cart is cookie-bound to the browser session that built it.
"""

from __future__ import annotations

import asyncio
import os
import re
import webbrowser
from datetime import date
from typing import Any
from urllib.parse import quote_plus, urlencode

from aria.tools.base import Tool, ToolError, ToolResult

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Every errand result reminds the model of the ground truth so the spoken
# answer can never claim a booking happened.
_NOT_BOOKED = (
    "Nothing is booked or paid yet — the user completes and pays for it "
    "themselves on the page that just opened."
)


def _parse_date(value: Any, label: str) -> date:
    text = str(value or "").strip()
    if not _ISO_DATE.match(text):
        raise ToolError(
            f"{label} must be an ISO date like 2026-07-10 (got {text!r}). "
            "Convert the user's spoken date first."
        )
    try:
        y, m, d = map(int, text.split("-"))
        return date(y, m, d)
    except ValueError as exc:
        raise ToolError(f"{label} {text!r} is not a real calendar date.") from exc


def _display_available() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def flight_search_url(
    origin: str, destination: str, depart: date, ret: date | None = None, adults: int = 1
) -> str:
    """Google Flights deep link for the route/dates (also the agentic reserve
    tool's starting page)."""
    query = f"flights from {origin} to {destination} on {depart.isoformat()}"
    if ret is not None:
        query += f" returning {ret.isoformat()}"
    else:
        query += " one way"
    if adults > 1:
        query += f" for {adults} adults"
    return "https://www.google.com/travel/flights?q=" + quote_plus(query)


def hotel_search_url(
    destination: str, checkin: date, checkout: date, adults: int = 2, rooms: int = 1
) -> str:
    """Booking.com results deep link (also the agentic reserve tool's start)."""
    params = {
        "ss": destination,
        "checkin": checkin.isoformat(),
        "checkout": checkout.isoformat(),
        "group_adults": adults,
        "no_rooms": rooms,
        "group_children": 0,
    }
    return "https://www.booking.com/searchresults.html?" + urlencode(params)


async def open_in_default_browser(url: str) -> None:
    """Open ``url`` in the user's default browser (the injectable seam for tests).

    ``webbrowser`` resolves the user's real default (BROWSER env, xdg settings),
    which is exactly the UX wanted for the final approve-and-pay step."""
    if not _display_available():
        raise ToolError(
            "I need your screen for that — log into the desktop session and ask again."
        )
    ok = await asyncio.to_thread(webbrowser.open, url)
    if not ok:
        raise ToolError("I couldn't open your browser for that page.")


class _ErrandTool(Tool):
    """Shared plumbing: an injectable opener so tests never launch a browser."""

    def __init__(self, opener=open_in_default_browser) -> None:
        self._open = opener


class OpenInBrowserTool(_ErrandTool):
    name = "open_in_browser"
    description = (
        "Open a specific web page (URL) in the user's own default browser — for "
        "handing them a page to read, approve, or pay on. Use a real http(s) URL "
        "you found this turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The http(s) URL to open."},
            "purpose": {
                "type": "string",
                "description": "What the user will do there, e.g. 'finish the booking'.",
            },
        },
        "required": ["url"],
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        url = str(kwargs.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ToolError("I can only open http(s) links in the browser.")
        await self._open(url)
        purpose = str(kwargs.get("purpose") or "").strip()
        return ToolResult(
            content=f"Opened {url} in the user's default browser. {_NOT_BOOKED}",
            spoken=(
                f"It's on your screen{' — ' + purpose if purpose else ''}. "
                "Take it from there whenever you're ready."
            ),
            data={"url": url},
        )


class BookFlightTool(_ErrandTool):
    name = "book_flight"
    description = (
        "Show live flight results (Google Flights) for the route and dates in the "
        "user's own browser, where THEY pick a flight and pay. Use for a plain "
        "'book/find me a flight' with no criteria. If they give a budget or ask "
        "you to pick the best, use reserve_flight instead. Dates ISO YYYY-MM-DD."
    )
    parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "description": "Departure city or airport."},
            "destination": {"type": "string", "description": "Arrival city or airport."},
            "depart_date": {"type": "string", "description": "Departure date, YYYY-MM-DD."},
            "return_date": {
                "type": "string",
                "description": "Return date YYYY-MM-DD; omit for one-way.",
            },
            "adults": {"type": "integer", "description": "Travellers (default 1)."},
        },
        "required": ["origin", "destination", "depart_date"],
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        origin = str(kwargs.get("origin") or "").strip()
        destination = str(kwargs.get("destination") or "").strip()
        if not origin or not destination:
            raise ToolError("I need both where you're flying from and to.")
        depart = _parse_date(kwargs.get("depart_date"), "depart_date")
        ret = kwargs.get("return_date")
        adults = int(kwargs.get("adults") or 1)

        trip = "one-way"
        back = None
        if ret:
            back = _parse_date(ret, "return_date")
            if back < depart:
                raise ToolError("return_date is before depart_date — check the dates.")
            trip = "return"
        url = flight_search_url(origin, destination, depart, back, adults)
        await self._open(url)
        spoken_date = depart.strftime("%-d %B")
        return ToolResult(
            content=(
                f"Opened live {trip} flight results {origin} -> {destination}, "
                f"departing {depart.isoformat()}, in the user's browser. {_NOT_BOOKED}"
            ),
            spoken=(
                f"Flights from {origin} to {destination} on {spoken_date} are on "
                "your screen — pick the one you like and book it right there."
            ),
            data={"url": url, "origin": origin, "destination": destination},
        )


class BookHotelTool(_ErrandTool):
    name = "book_hotel"
    description = (
        "Show live hotel results (Booking.com) for the place and dates in the "
        "user's own browser, where THEY choose a room and pay. Use for a plain "
        "'book/find me a hotel' with no criteria. If they give a budget or ask "
        "you to pick the best, use reserve_hotel instead. Dates ISO YYYY-MM-DD."
    )
    parameters = {
        "type": "object",
        "properties": {
            "destination": {"type": "string", "description": "City / area / hotel name."},
            "checkin": {"type": "string", "description": "Check-in date, YYYY-MM-DD."},
            "checkout": {"type": "string", "description": "Check-out date, YYYY-MM-DD."},
            "adults": {"type": "integer", "description": "Guests (default 2)."},
            "rooms": {"type": "integer", "description": "Rooms (default 1)."},
        },
        "required": ["destination", "checkin", "checkout"],
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        destination = str(kwargs.get("destination") or "").strip()
        if not destination:
            raise ToolError("Where should the hotel be?")
        checkin = _parse_date(kwargs.get("checkin"), "checkin")
        checkout = _parse_date(kwargs.get("checkout"), "checkout")
        if checkout <= checkin:
            raise ToolError("checkout must be after checkin — check the dates.")
        url = hotel_search_url(
            destination, checkin, checkout,
            adults=int(kwargs.get("adults") or 2), rooms=int(kwargs.get("rooms") or 1),
        )
        await self._open(url)
        nights = (checkout - checkin).days
        return ToolResult(
            content=(
                f"Opened hotel results for {destination}, {checkin.isoformat()} to "
                f"{checkout.isoformat()} ({nights} night{'s' if nights != 1 else ''}), "
                f"in the user's browser. {_NOT_BOOKED}"
            ),
            spoken=(
                f"Hotels in {destination} for those dates are on your screen — "
                "pick your favourite and book it there."
            ),
            data={"url": url, "destination": destination, "nights": nights},
        )


class ShopOnlineTool(_ErrandTool):
    name = "shop_online"
    description = (
        "Shop for a physical product to buy: opens live shopping results for it in "
        "the user's own browser, where THEY choose and pay. Use for 'buy me X', "
        "'I need a new Y', 'find me a cheap Z'. NOT for food delivery (use "
        "order_food) or flights/hotels."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to buy, with any specifics: 'usb-c hub 4 ports'.",
            }
        },
        "required": ["query"],
    }

    async def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            raise ToolError("What should I shop for?")
        url = "https://www.google.com/search?tbm=shop&q=" + quote_plus(query)
        await self._open(url)
        return ToolResult(
            content=(
                f"Opened shopping results for {query!r} in the user's browser. "
                f"{_NOT_BOOKED}"
            ),
            spoken=(
                f"Shopping results for {query} are on your screen — pick the one "
                "you want and check out there."
            ),
            data={"url": url, "query": query},
        )


def errand_tools(opener=open_in_default_browser) -> list[Tool]:
    return [
        OpenInBrowserTool(opener),
        BookFlightTool(opener),
        BookHotelTool(opener),
        ShopOnlineTool(opener),
    ]
