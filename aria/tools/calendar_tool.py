"""Google Calendar tools: list and create events.

The Google client is sync, so every API call runs in a worker thread. A
``service_provider`` callable is injected (so tests can mock it); it builds the
Calendar service or raises GoogleNotConnected, which becomes a friendly spoken
error telling the user to run `aria connect google`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from aria.integrations.google_auth import (
    GoogleNotConnected,
    friendly_google_error,
    run_blocking,
)
from aria.tools.base import Tool, ToolError, ToolResult

ServiceProvider = Callable[[], Any]

_NOT_CONNECTED = "You're not connected to Google yet — run `aria connect google` first."


def _range_bounds(text: str, now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or datetime.now().astimezone()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    t = (text or "").lower().strip()
    if "today" in t:
        return now, midnight + timedelta(days=1)
    if "tomorrow" in t:
        start = midnight + timedelta(days=1)
        return start, start + timedelta(days=1)
    if "month" in t:
        return now, now + timedelta(days=30)
    if "week" in t or not t:
        return now, now + timedelta(days=7)
    return now, now + timedelta(days=7)


def _event_when(event: dict) -> tuple[datetime | None, str]:
    """Return (start_dt or None for all-day, spoken time phrase)."""
    start = event.get("start", {})
    if start.get("date"):  # all-day
        d = datetime.fromisoformat(start["date"])
        return None, d.strftime("%A all day")
    if start.get("dateTime"):
        dt = datetime.fromisoformat(start["dateTime"])
        return dt, dt.strftime("%-I:%M %p").lstrip("0")
    return None, "sometime"


class _CalendarTool(Tool):
    def __init__(self, service_provider: ServiceProvider) -> None:
        self._provider = service_provider

    async def _service(self):
        try:
            return await run_blocking(self._provider)
        except GoogleNotConnected as exc:
            raise ToolError(_NOT_CONNECTED) from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(self._err(exc)) from exc

    async def _call(self, fn):
        try:
            return await run_blocking(fn)
        except Exception as exc:  # noqa: BLE001 - bounded; never hangs the turn
            raise ToolError(self._err(exc)) from exc

    @staticmethod
    def _err(exc: BaseException) -> str:
        return f"I couldn't reach your calendar ({friendly_google_error(exc)})."


class ListEventsTool(_CalendarTool):
    name = "list_events"
    description = (
        "List the user's Google Calendar events for a period ('today', 'tomorrow', "
        "'this week'). Use for 'what's on my schedule', 'what's on today'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "range": {"type": "string", "description": "today | tomorrow | this week"}
        },
    }
    risk = "safe"

    async def run(self, **kwargs: Any) -> ToolResult:
        service = await self._service()
        time_min, time_max = _range_bounds(str(kwargs.get("range", "")))

        def _list():
            return (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=time_min.isoformat(),
                    timeMax=time_max.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=25,
                )
                .execute()
            )

        result = await self._call(_list)
        events = result.get("items", [])
        if not events:
            period = str(kwargs.get("range") or "the next week")
            return ToolResult(
                content="no events", spoken=f"You've got nothing on {period} — clear."
            )

        parts = []
        for ev in events:
            _, when = _event_when(ev)
            title = ev.get("summary", "untitled")
            parts.append(f"{title} at {when}" if "all day" not in when else f"{title}, {when}")
        n = len(parts)
        listed = "; ".join(parts)
        spoken = f"You've got {n} thing{'s' if n != 1 else ''}: {listed}."
        return ToolResult(content=listed, data={"count": n, "events": parts}, spoken=spoken)


class CreateEventTool(_CalendarTool):
    name = "create_event"
    description = (
        "Add an event to the user's Google Calendar. Use for 'add a meeting Friday "
        "at 3pm'. Provide a title and a start time; end defaults to one hour later."
    )
    parameters = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Event title."},
            "start": {"type": "string", "description": "Start time, e.g. 'Friday 3pm'."},
            "end": {"type": "string", "description": "Optional end time."},
            "attendees": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional attendee emails.",
            },
        },
        "required": ["title", "start"],
    }
    risk = "confirm"  # outward action — read back + confirm before creating

    def confirm_summary(self, arguments: dict[str, Any]) -> str:
        title = arguments.get("title", "an event")
        start = arguments.get("start", "")
        when = f" for {start}" if start else ""
        return f"add '{title}' to your calendar{when}"

    async def run(self, **kwargs: Any) -> ToolResult:
        from aria.tools.timers import parse_when  # natural time parsing (reused)

        title = str(kwargs.get("title") or "").strip()
        if not title:
            raise ToolError("What should I call the event?")
        try:
            start_dt = datetime.fromtimestamp(parse_when(str(kwargs.get("start", "")))[0])
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"I couldn't understand that start time ({exc}).") from exc
        if kwargs.get("end"):
            end_dt = datetime.fromtimestamp(parse_when(str(kwargs["end"]))[0])
        else:
            end_dt = start_dt + timedelta(hours=1)

        attendees = [a for a in (kwargs.get("attendees") or []) if "@" in str(a)]
        body: dict = {
            "summary": title,
            "start": {"dateTime": start_dt.astimezone().isoformat()},
            "end": {"dateTime": end_dt.astimezone().isoformat()},
        }
        if attendees:
            body["attendees"] = [{"email": a} for a in attendees]

        service = await self._service()

        def _insert():
            return (
                service.events()
                .insert(
                    calendarId="primary",
                    body=body,
                    sendUpdates="all" if attendees else "none",
                )
                .execute()
            )

        created = await self._call(_insert)
        # Only claim success if the API actually returned a created event id.
        event_id = created.get("id")
        if not event_id:
            raise ToolError("That didn't go through — the event wasn't created.")

        # Echo the time the API ACTUALLY scheduled (read back from the response),
        # not the time we parsed — so the spoken line can't drift from reality.
        scheduled_dt, _ = _event_when(created)
        scheduled_dt = scheduled_dt or start_dt
        when = scheduled_dt.strftime("%A at %-I:%M %p").replace(" 0", " ")
        return ToolResult(
            content=f"created '{title}' {when} (id {event_id})",
            data={"id": event_id, "title": title, "when": when,
                  "start": scheduled_dt.astimezone().isoformat()},
            spoken=f"Done — added {title} to your calendar for {when}.",
        )


def calendar_tools(service_provider: ServiceProvider) -> list[Tool]:
    return [ListEventsTool(service_provider), CreateEventTool(service_provider)]
