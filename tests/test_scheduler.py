"""Persistent scheduler + timer/reminder tools (Phase A + Phase B time fixes)."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from aria.core.scheduler import SchedulerService, _next_wallclock
from aria.tools.timers import (
    ManageTimersTool,
    SetReminderTool,
    SetTimerTool,
    _describe_when,
    parse_when,
)


# --- persistence across a simulated restart -------------------------------
async def test_alarm_persists_and_reloads(tmp_path):
    db = tmp_path / "alarms.sqlite3"
    s1 = SchedulerService(db_path=db, now=lambda: 1000.0)
    await s1.open()
    aid = await s1.add("laundry", 5000.0, "none")
    await s1.close()

    s2 = SchedulerService(db_path=db, now=lambda: 1000.0)  # fresh process/service
    await s2.open()
    active = await s2.list_active()
    await s2.close()
    assert len(active) == 1
    assert active[0].id == aid and active[0].label == "laundry" and active[0].fire_at == 5000.0


# --- fire enqueues a spoken announcement AND notifies ---------------------
async def test_fire_announces_and_notifies(tmp_path):
    spoken: list[str] = []
    notified: list[tuple[str, str]] = []

    async def notify(title, message):
        notified.append((title, message))

    s = SchedulerService(
        db_path=tmp_path / "a.db",
        announce=spoken.append,
        notify=notify,
        name_provider=lambda: "Arash",
        now=lambda: 2000.0,
    )
    await s.open()
    await s.add("laundry timer", 1900.0, "none")
    alarm = (await s.list_active())[0]
    await s._fire(alarm)
    await s.close()

    assert spoken and "Arash" in spoken[0] and "laundry timer" in spoken[0]
    assert notified == [("Aria", "⏰ laundry timer")]


async def test_fire_without_name_is_still_natural(tmp_path):
    spoken: list[str] = []
    s = SchedulerService(
        db_path=tmp_path / "a.db", announce=spoken.append,
        name_provider=lambda: None, now=lambda: 10.0,
    )
    await s.open()
    await s.add("tea", 5.0, "none")
    await s._fire((await s.list_active())[0])
    await s.close()
    assert spoken and "tea" in spoken[0] and "Hey" not in spoken[0]


# --- recurrence + catch-up ------------------------------------------------
async def test_one_shot_deactivates_recurring_reschedules(tmp_path):
    now = [5000.0]
    s = SchedulerService(db_path=tmp_path / "a.db", announce=lambda _t: None, now=lambda: now[0])
    await s.open()
    await s.add("once", 5000.0, "none")
    await s.add("standup", 5000.0, "interval:60")
    for alarm in await s.list_active():
        await s._fire(alarm)
    active = await s.list_active()
    await s.close()

    labels = {a.label: a for a in active}
    assert "once" not in labels  # one-shot deactivated
    assert labels["standup"].fire_at == 5060.0  # advanced one interval


async def test_recurring_skips_missed_occurrences(tmp_path):
    now = [100_000.0]
    s = SchedulerService(db_path=tmp_path / "a.db", announce=lambda _t: None, now=lambda: now[0])
    await s.open()
    await s.add("pills", 100_000.0 - 5 * 86_400, "daily")  # 5 days overdue
    await s._fire((await s.list_active())[0])
    nxt = (await s.list_active())[0].fire_at
    await s.close()
    assert now[0] < nxt <= now[0] + 86_400  # advanced to the NEXT future occurrence


async def test_catch_up_fires_recent_once_and_drops_stale(tmp_path):
    spoken: list[str] = []
    s = SchedulerService(
        db_path=tmp_path / "a.db", announce=spoken.append, now=lambda: 10_000.0, stale_after=3600.0
    )
    await s.open()
    await s.add("recent", 9_000.0, "none")   # 1000s overdue (< stale) -> fire once
    await s.add("ancient", 1_000.0, "none")  # 9000s overdue (> stale) -> dropped
    await s._catch_up()
    active = await s.list_active()
    await s.close()

    assert any("recent" in t and "while you were away" in t for t in spoken)
    assert not any("ancient" in t for t in spoken)
    assert active == []  # recent fired+deactivated, ancient dropped


# --- the loop actually fires (real clock, short sleep) --------------------
async def test_loop_fires_due_alarm(tmp_path):
    spoken: list[str] = []
    s = SchedulerService(db_path=tmp_path / "a.db", announce=spoken.append)
    await s.start()
    await s.add("ping", time.time() + 0.05, "none")
    await asyncio.sleep(0.25)
    await s.stop()
    assert any("ping" in t for t in spoken)


# --- time parsing ---------------------------------------------------------
def test_parse_when_variants():
    base = datetime(2026, 6, 24, 12, 0, 0).timestamp()  # a Wednesday noon

    fa, rec = parse_when("at 8am", now=base)
    assert rec == "none" and datetime.fromtimestamp(fa).hour == 8 and fa > base

    fa, rec = parse_when("every day at 9", now=base)
    assert rec == "daily" and datetime.fromtimestamp(fa).hour == 9

    fa, rec = parse_when("in 2 hours", now=base)
    assert rec == "none" and abs(fa - (base + 7200)) < 1

    fa, rec = parse_when("every 30 minutes", now=base)
    assert rec == "interval:1800" and abs(fa - (base + 1800)) < 1

    fa, rec = parse_when("tomorrow at 7", now=base)
    when = datetime.fromtimestamp(fa)
    assert rec == "none" and when.hour == 7 and when.day == 25


def test_parse_when_weekdays():
    base = datetime(2026, 6, 24, 12, 0, 0).timestamp()  # Wednesday 24 June 2026

    # "Friday at 3pm" must anchor to the UPCOMING Friday (26th), not today.
    when = datetime.fromtimestamp(parse_when("Friday at 3pm", now=base)[0])
    assert when.weekday() == 4 and when.day == 26 and when.hour == 15

    # Bare weekday -> that day, default 9am.
    when = datetime.fromtimestamp(parse_when("friday", now=base)[0])
    assert when.weekday() == 4 and when.day == 26 and when.hour == 9

    # "next monday" -> Monday of the following week (6 July), not the coming one.
    when = datetime.fromtimestamp(parse_when("next monday", now=base)[0])
    assert when.weekday() == 0 and when.day == 6 and when.month == 7

    # A weekday that already passed today rolls a week (Wed said on Wed afternoon).
    when = datetime.fromtimestamp(parse_when("Wednesday at 9am", now=base)[0])
    assert when.weekday() == 2 and when.day == 1 and when.month == 7  # next Wed


# --- tools end-to-end -----------------------------------------------------
async def test_set_reminder_persists_with_recurrence(tmp_path):
    s = SchedulerService(db_path=tmp_path / "a.db")
    await s.open()
    res = await SetReminderTool(s).run(when="every day at 9", label="meds")
    assert res.data["recurrence"] == "daily"
    active = await s.list_active()
    await s.close()
    assert len(active) == 1 and active[0].label == "meds" and active[0].recurrence == "daily"


async def test_manage_timers_list_cancel_snooze(tmp_path):
    s = SchedulerService(db_path=tmp_path / "a.db")
    await s.open()
    await SetTimerTool(s).run(duration="10 minutes", label="laundry")
    await SetTimerTool(s).run(duration="5 minutes", label="tea")
    mgr = ManageTimersTool(s)

    listing = await mgr.run(action="list")
    assert "laundry" in listing.content and "tea" in listing.content

    await mgr.run(action="cancel", label="laundry")
    assert {a.label for a in await s.list_active()} == {"tea"}

    before = (await s.list_active())[0].fire_at
    await mgr.run(action="snooze", label="tea", minutes=30)
    after = (await s.list_active())[0].fire_at
    await s.close()
    assert after > before  # snoozed further out (tea was ~5min, now ~30min)


# --- Part 1: friendlier relative phrasing ---------------------------------
def test_describe_when_short_one_shot_is_relative():
    now = 1_000_000.0
    assert _describe_when(now + 60, "none", now=now) == "in 1 minute"
    assert _describe_when(now + 45, "none", now=now) == "in 45 seconds"
    assert _describe_when(now + 2 * 3600, "none", now=now) == "in 2 hours"
    # Far-off one-shot stays absolute; recurrences keep their phrasing.
    assert _describe_when(now + 20 * 3600, "none", now=now).startswith("at ")
    assert _describe_when(now, "daily", now=now).startswith("every day at ")


# --- Part 1: DST-safe daily/weekly recurrence -----------------------------
def test_daily_recurrence_holds_local_time_across_dst(monkeypatch):
    # US DST springs forward Sun 2026-03-08 02:00 -> 03:00. A 09:00 daily alarm
    # on 03-07 must next fire at 09:00 LOCAL on 03-08, not shift to 08:00.
    old_tz = time.tzname
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        base = datetime(2026, 3, 7, 9, 0, 0).timestamp()  # 09:00 EST
        just_after = base + 1
        nxt = _next_wallclock(base, 1, just_after)
        when = datetime.fromtimestamp(nxt)
        assert (when.year, when.month, when.day) == (2026, 3, 8)
        assert when.hour == 9 and when.minute == 0  # local wall-clock held
        # A naive fixed +86400 drifts off 09:00 across the DST jump — prove it differs.
        assert datetime.fromtimestamp(base + 86_400).hour != 9
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()
        assert time.tzname or old_tz  # tz restored
