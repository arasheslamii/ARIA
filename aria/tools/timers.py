"""Timer / reminder tools, backed by the persistent SchedulerService.

These replace the old in-process asyncio timers: alarms now survive restarts and
are announced ALOUD (proactive speech) as well as via desktop notification.

  * set_timer     — relative countdown ("10 minutes", "1h30m").
  * set_reminder  — absolute/recurring ("at 8am", "every day at 9", "in 2 hours").
  * manage_timers — list / cancel (by label) / snooze.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Any

from dateutil import parser as dateparser

from aria.core.scheduler import SchedulerService
from aria.tools.base import Tool, ToolError, ToolResult

_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "weds": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}
# Longest names first so "wednesday" wins over "wed".
_WEEKDAY_RE = re.compile(
    r"\b(" + "|".join(sorted(_WEEKDAYS, key=len, reverse=True)) + r")\b"
)


def _upcoming_weekday(now_dt: datetime, weekday: int, force_next: bool):
    """The date of the next given weekday (today if it matches and not 'next')."""
    days_ahead = (weekday - now_dt.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7 if force_next else 0
    elif force_next:
        days_ahead += 7
    return (now_dt + timedelta(days=days_ahead)).date()

_UNIT_SECONDS = {"s": 1, "sec": 1, "second": 1, "m": 60, "min": 60, "minute": 60,
                 "h": 3600, "hr": 3600, "hour": 3600}
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|seconds?|secs?|[hms])")


def parse_duration(text: str) -> float:
    """Parse '1h30m', '90 seconds', '10 minutes' -> seconds."""
    total = 0.0
    for value, unit in _DURATION_RE.findall(text.lower()):
        key = unit.rstrip("s")
        key = {"hr": "h", "hrs": "h", "min": "m", "sec": "s"}.get(key, key)
        total += float(value) * _UNIT_SECONDS.get(key, 0)
    if total == 0:
        raise ToolError(f"could not understand duration: {text!r}")
    return total


def parse_when(text: str, now: float | None = None) -> tuple[float, str]:
    """Parse an absolute/recurring time phrase -> (fire_at_epoch, recurrence).

    Handles "in 2 hours", "at 8am", "every day at 9", "every 30 minutes",
    "tomorrow at 7". Recurrence is 'none' | 'daily' | 'weekly' | 'interval:<sec>'.
    """
    now_ts = now if now is not None else time.time()
    now_dt = datetime.fromtimestamp(now_ts)
    t = " ".join(text.lower().split())

    recurrence = "none"
    if re.search(r"\bevery day\b|\bdaily\b|\beach day\b", t):
        recurrence = "daily"
    elif re.search(r"\bevery week\b|\bweekly\b", t):
        recurrence = "weekly"
    else:
        m = re.search(r"\bevery\s+(\d+)\s*(seconds?|minutes?|hours?)\b", t)
        if m:
            n = int(m.group(1))
            unit = m.group(2)[0]
            recurrence = f"interval:{n * {'s': 1, 'm': 60, 'h': 3600}[unit]}"

    # "in <duration>" relative form, and interval recurrence both fire from now.
    rel = re.match(r"\s*in\s+(.+)", t)
    if rel and recurrence == "none":
        return now_ts + parse_duration(rel.group(1)), "none"
    if recurrence.startswith("interval:"):
        return now_ts + float(recurrence.split(":", 1)[1]), recurrence

    # Resolve the target DATE: an explicit weekday ("Friday", "next Monday"),
    # "tomorrow", or today — anchored to the REAL current local date.
    day_offset = 1 if "tomorrow" in t else 0
    force_next = bool(re.search(r"\bnext\b", t))
    wm = _WEEKDAY_RE.search(t)
    weekday = _WEEKDAYS[wm.group(1)] if wm else None
    if weekday is not None:
        base_date = _upcoming_weekday(now_dt, weekday, force_next)
        explicit_date = True
    else:
        base_date = (now_dt + timedelta(days=day_offset)).date()
        explicit_date = day_offset > 0

    # Explicit clock time. We only treat a bare number as o'clock when there's an
    # am/pm, explicit minutes, the word "at", or a daily/weekly recurrence — so we
    # don't misread stray numbers, and so "every day at 9" means 9 o'clock.
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", t)
    has_time_cue = bool(m and (m.group(3) or m.group(2) or re.search(r"\bat\b", t)))
    if m and (has_time_cue or recurrence in ("daily", "weekly")):
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        ap = (m.group(3) or "").replace(".", "")
        if ap == "pm" and hour < 12:
            hour += 12
        elif ap == "am" and hour == 12:
            hour = 0
        if hour > 23 or minute > 59:
            raise ToolError(f"could not understand time: {text!r}")
        target = datetime.combine(base_date, dt_time(hour, minute))
        if target <= now_dt and not explicit_date:  # passed today, no explicit date
            target += timedelta(days=1)
        elif target <= now_dt and weekday is not None:  # that weekday already passed
            target += timedelta(days=7)
        return target.timestamp(), recurrence

    # A date but no time (e.g. "Friday", "tomorrow") -> default to 9am.
    if explicit_date:
        target = datetime.combine(base_date, dt_time(9, 0))
        if target <= now_dt:
            target += timedelta(days=7 if weekday is not None else 1)
        return target.timestamp(), recurrence

    # Fallback: named dates ("June 25 3pm").
    cleaned = re.sub(r"\b(remind me|remind|set|to|at|tomorrow|next)\b", " ", t).strip()
    try:
        when_dt = dateparser.parse(cleaned, default=now_dt, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ToolError(f"could not understand time: {text!r}") from exc
    if when_dt is None:
        raise ToolError(f"could not understand time: {text!r}")
    if when_dt <= now_dt:
        when_dt += timedelta(days=1)
    return when_dt.timestamp(), recurrence


def _humanize(seconds: float) -> str:
    if seconds < 60:
        n = int(seconds)
        return f"{n} second{'s' if n != 1 else ''}"
    if seconds < 3600:
        n = round(seconds / 60)
        return f"{n} minute{'s' if n != 1 else ''}"
    hours = seconds / 3600
    if abs(hours - round(hours)) < 0.05:  # whole hours read cleaner ("2 hours")
        n = round(hours)
        return f"{n} hour{'s' if n != 1 else ''}"
    return f"{hours:.1f} hours"


def _describe_when(fire_at: float, recurrence: str, now: float | None = None) -> str:
    when = datetime.fromtimestamp(fire_at)
    clock = when.strftime("%-I:%M %p").lstrip("0")
    if recurrence == "daily":
        return f"every day at {clock}"
    if recurrence == "weekly":
        return f"every week at {clock}"
    if recurrence.startswith("interval:"):
        return f"every {_humanize(float(recurrence.split(':', 1)[1]))}"
    # One-shot: short countdowns read naturally as relative ("in 2 minutes"); only
    # far-off / next-day ones use an absolute clock time.
    delta = fire_at - (now if now is not None else time.time())
    if delta <= 0:
        return "now"
    if delta < 6 * 3600:
        return f"in {_humanize(delta)}"
    if delta < 18 * 3600:
        return f"at {clock}"
    return when.strftime("at %-I:%M %p on %A").replace(" 0", " ")


class SetTimerTool(Tool):
    name = "set_timer"
    description = (
        "Set a countdown timer, e.g. '10 minutes' or '1h30m'. Aria announces it "
        "out loud (and notifies) when it's done. Survives restarts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "duration": {"type": "string", "description": "e.g. '10 minutes', '1h30m'."},
            "label": {"type": "string", "description": "What the timer is for."},
        },
        "required": ["duration"],
    }
    risk = "safe"

    def __init__(self, scheduler: SchedulerService) -> None:
        self._sched = scheduler

    async def run(self, **kwargs: Any) -> ToolResult:
        seconds = parse_duration(str(kwargs.get("duration", "")))
        label = str(kwargs.get("label") or "timer")
        alarm_id = await self._sched.add(label, time.time() + seconds, "none")
        return ToolResult(
            content=f"timer #{alarm_id} set for {label} in {_humanize(seconds)}",
            data={"id": alarm_id, "seconds": seconds},
            spoken=f"Okay, {label} in {_humanize(seconds)}.",
        )


class SetReminderTool(Tool):
    name = "set_reminder"
    description = (
        "Set a reminder at an absolute or recurring time: 'at 8am', "
        "'every day at 9', 'in 2 hours', 'tomorrow at 7'. Announced aloud."
    )
    parameters = {
        "type": "object",
        "properties": {
            "when": {"type": "string", "description": "e.g. 'at 8am', 'every day at 9'."},
            "label": {"type": "string", "description": "What to remind about."},
        },
        "required": ["when"],
    }
    risk = "safe"

    def __init__(self, scheduler: SchedulerService) -> None:
        self._sched = scheduler

    async def run(self, **kwargs: Any) -> ToolResult:
        fire_at, recurrence = parse_when(str(kwargs.get("when", "")))
        label = str(kwargs.get("label") or "reminder")
        alarm_id = await self._sched.add(label, fire_at, recurrence)
        phrase = _describe_when(fire_at, recurrence)
        return ToolResult(
            content=f"reminder #{alarm_id} ({label}) {phrase}",
            data={"id": alarm_id, "fire_at": fire_at, "recurrence": recurrence},
            spoken=f"Okay, I'll remind you about {label} {phrase}.",
        )


class ManageTimersTool(Tool):
    name = "manage_timers"
    description = (
        "List, cancel, or snooze your timers and reminders. action: 'list', "
        "'cancel' (match by label), or 'snooze' (by label or the soonest one)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "cancel", "snooze"]},
            "label": {"type": "string", "description": "Which timer/reminder to act on."},
            "minutes": {"type": "number", "description": "Snooze duration (default 5)."},
        },
        "required": ["action"],
    }
    risk = "safe"

    def __init__(self, scheduler: SchedulerService) -> None:
        self._sched = scheduler

    async def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "list"))
        label = str(kwargs.get("label") or "").strip().lower()
        alarms = await self._sched.list_active()

        if action == "list":
            if not alarms:
                return ToolResult(content="no active timers", spoken="You have no timers set.")
            parts = [f"{a.label} {_describe_when(a.fire_at, a.recurrence)}" for a in alarms]
            return ToolResult(
                content="; ".join(parts),
                data={"count": len(alarms)},
                spoken="You have " + ", ".join(parts) + ".",
            )

        match = self._match(alarms, label)
        if action == "cancel":
            if match is None:
                return ToolResult(content="no match", spoken="I couldn't find that timer.")
            await self._sched.cancel(match.id)
            return ToolResult(
                content=f"cancelled {match.label}", spoken=f"Cancelled {match.label}."
            )

        if action == "snooze":
            target = match or (alarms[0] if alarms else None)
            if target is None:
                return ToolResult(content="nothing to snooze", spoken="There's nothing to snooze.")
            minutes = float(kwargs.get("minutes") or 5)
            await self._sched.snooze(target.id, minutes * 60)
            return ToolResult(
                content=f"snoozed {target.label} {minutes:g} min",
                spoken=f"Snoozed {target.label} for {minutes:g} minutes.",
            )

        raise ToolError(f"unknown action: {action}")

    @staticmethod
    def _match(alarms: list, label: str):
        if not label:
            return None
        for a in alarms:
            if label in a.label.lower() or a.label.lower() in label:
                return a
        return None


def timer_tools(scheduler: SchedulerService) -> list[Tool]:
    return [SetTimerTool(scheduler), SetReminderTool(scheduler), ManageTimersTool(scheduler)]
