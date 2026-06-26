"""Persistent, SQLite-backed scheduler for timers/alarms/reminders.

Unlike the old in-process asyncio timers, alarms here survive `aria` restarts and
fire even after a reboot (catch-up on start). A single async loop sleeps until the
NEXT due alarm rather than one task per timer.

This is also the first PROACTIVE-SPEECH path: on fire, the scheduler pushes a
spoken announcement onto a shared queue (the producer side) which the
VoicePipeline drains when idle. The same channel is reused for briefings later.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import aiosqlite

from aria.config.loader import state_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alarms (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT NOT NULL,
    fire_at    REAL NOT NULL,
    recurrence TEXT NOT NULL DEFAULT 'none',
    created_at REAL NOT NULL,
    active     INTEGER NOT NULL DEFAULT 1
);
"""

# recurrence is one of: 'none' | 'daily' | 'weekly' | 'interval:<seconds>'
AnnounceFn = Callable[[str], None]
NotifyFn = Callable[[str, str], Awaitable[None]]
NameProvider = Callable[[], "str | None"]

# Drop one-shot alarms that are this far past due on startup (assume missed/stale).
_STALE_AFTER_S = 24 * 3600.0


@dataclass
class Alarm:
    id: int
    label: str
    fire_at: float
    recurrence: str
    created_at: float
    active: bool


def _next_wallclock(base_epoch: float, step_days: int, now_epoch: float) -> float:
    """Next occurrence at the SAME local wall-clock time, DST-correct.

    Advancing daily/weekly by a fixed 86400/604800 seconds drifts by an hour
    across a DST transition. Instead we keep the local hour:minute and recompute
    the epoch for each candidate calendar date — ``datetime.fromtimestamp`` /
    ``.timestamp()`` apply the OS timezone's DST rules for that date
    automatically, with no hardcoded offsets, so "every day at 9" is 9:00 local
    on every machine and across DST.
    """
    base = datetime.fromtimestamp(base_epoch)
    wall = base.time()
    candidate = datetime.combine(base.date() + timedelta(days=step_days), wall)
    while candidate.timestamp() <= now_epoch:
        candidate = datetime.combine(candidate.date() + timedelta(days=step_days), wall)
    return candidate.timestamp()


def _period_seconds(recurrence: str) -> float | None:
    if recurrence == "daily":
        return 24 * 3600.0
    if recurrence == "weekly":
        return 7 * 24 * 3600.0
    if recurrence.startswith("interval:"):
        try:
            return float(recurrence.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _icon_path() -> Path | None:
    """Path to the Aria icon: installed themed icon, else the in-repo SVG."""
    for p in (
        Path("/usr/share/icons/hicolor/scalable/apps/aria.svg"),
        Path(__file__).resolve().parents[1] / "packaging" / "aria.svg",
    ):
        if p.exists():
            return p
    return None


def _build_icon(dn) -> object | None:  # noqa: ANN001 - the desktop_notifier module
    """Build a desktop_notifier ``Icon`` (6.x). Must be a proper Icon object — a
    plain string path makes 6.x call ``.is_named`` on it and crash."""
    icon_cls = getattr(dn, "Icon", None)
    if icon_cls is None:  # very old version without Icon resource type
        return None
    try:
        path = _icon_path()
        return icon_cls(path=path) if path else icon_cls(name="aria")
    except Exception:  # noqa: BLE001
        return None


async def desktop_notify(title: str, message: str) -> None:
    """Best-effort, BRANDED desktop notification that NEVER raises.

    Shows "Aria" + the Aria icon (not a generic "python" banner). The icon is built
    as a proper ``Icon`` object and NO ``sound`` is passed — newer desktop-notifier
    expects ``Icon``/``Sound`` objects and crashes on a string/bool
    (``'… has no is_named'``). Any backend error is swallowed and logged so a flaky
    notification can never break a timer firing.
    """
    try:
        import desktop_notifier as dn

        icon = _build_icon(dn)
        notifier_kwargs: dict = {"app_name": "Aria"}
        send_kwargs: dict = {"title": title, "message": message}
        if icon is not None:
            notifier_kwargs["app_icon"] = icon
            send_kwargs["icon"] = icon
        urgency = getattr(dn, "Urgency", None)
        if urgency is not None:
            send_kwargs["urgency"] = urgency.Normal

        try:
            notifier = dn.DesktopNotifier(**notifier_kwargs)
        except TypeError:  # older versions name the args differently
            notifier = dn.DesktopNotifier(app_name="Aria")
        try:
            await notifier.send(**send_kwargs)
        except TypeError:  # minimal fallback signature
            await notifier.send(title=title, message=message)
    except Exception as exc:  # noqa: BLE001 - notifications are optional, never fatal
        logging.getLogger("aria").debug("desktop notification failed: %s", exc)


class SchedulerService:
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        announce: AnnounceFn | None = None,
        notify: NotifyFn | None = None,
        name_provider: NameProvider | None = None,
        now: Callable[[], float] = time.time,
        stale_after: float = _STALE_AFTER_S,
    ) -> None:
        self.db_path = db_path or (state_dir() / "alarms.sqlite3")
        self._announce = announce
        self._notify = notify
        self._name = name_provider or (lambda: None)
        self._now = now
        self._stale_after = stale_after
        self._db: aiosqlite.Connection | None = None
        self._wakeup = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stopping = False

    # --- lifecycle -----------------------------------------------------
    async def open(self) -> None:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            await self._db.executescript(_SCHEMA)
            await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def start(self) -> None:
        """Open the db, fire/advance anything due while we were away, then run."""
        await self.open()
        await self._catch_up()
        self._stopping = False
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stopping = True
        self._wakeup.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.close()

    async def aclose(self) -> None:  # so it can live in the runtime's manager list
        await self.stop()

    # --- public API ----------------------------------------------------
    async def add(self, label: str, fire_at: float, recurrence: str = "none") -> int:
        cur = await self.db.execute(
            "INSERT INTO alarms(label, fire_at, recurrence, created_at, active) "
            "VALUES(?,?,?,?,1)",
            (label, fire_at, recurrence, self._now()),
        )
        await self.db.commit()
        self._wakeup.set()  # the next-due time may have changed
        return int(cur.lastrowid)

    async def cancel(self, alarm_id: int) -> bool:
        cur = await self.db.execute(
            "UPDATE alarms SET active=0 WHERE id=? AND active=1", (alarm_id,)
        )
        await self.db.commit()
        self._wakeup.set()
        return cur.rowcount > 0

    async def snooze(self, alarm_id: int, seconds: float) -> float | None:
        new_fire = self._now() + seconds
        cur = await self.db.execute(
            "UPDATE alarms SET fire_at=?, active=1 WHERE id=?", (new_fire, alarm_id)
        )
        await self.db.commit()
        self._wakeup.set()
        return new_fire if cur.rowcount > 0 else None

    async def list_active(self) -> list[Alarm]:
        async with self.db.execute(
            "SELECT id, label, fire_at, recurrence, created_at, active "
            "FROM alarms WHERE active=1 ORDER BY fire_at"
        ) as cur:
            rows = await cur.fetchall()
        return [Alarm(r[0], r[1], r[2], r[3], r[4], bool(r[5])) for r in rows]

    # --- internals -----------------------------------------------------
    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SchedulerService.open() not called")
        return self._db

    async def _loop(self) -> None:
        while not self._stopping:
            self._wakeup.clear()  # clear BEFORE reading, so changes aren't lost
            nxt = await self._next_due()
            if nxt is None:
                await self._wakeup.wait()  # nothing scheduled — wait for a change
                continue
            delay = nxt.fire_at - self._now()
            if delay > 0:
                try:
                    await asyncio.wait_for(self._wakeup.wait(), timeout=delay)
                    continue  # something changed -> recompute the next-due alarm
                except TimeoutError:
                    pass  # the delay elapsed -> it's due
            await self._fire(nxt)

    async def _next_due(self) -> Alarm | None:
        active = await self.list_active()
        return active[0] if active else None

    async def _fire(self, alarm: Alarm, *, while_away: bool = False) -> None:
        text = self._announcement_text(alarm, while_away)
        if self._announce is not None:
            self._announce(text)
        if self._notify is not None:
            await self._notify("Aria", f"⏰ {alarm.label}")
        nxt = self._next_occurrence(alarm)
        if nxt is None:
            await self._deactivate(alarm.id)
        else:
            await self._reschedule(alarm.id, nxt)

    def _next_occurrence(self, alarm: Alarm) -> float | None:
        rec = alarm.recurrence
        now = self._now()
        # Calendar recurrences re-anchor to local wall-clock time (DST-safe);
        # interval recurrences are a genuinely fixed duration.
        if rec == "daily":
            return _next_wallclock(alarm.fire_at, 1, now)
        if rec == "weekly":
            return _next_wallclock(alarm.fire_at, 7, now)
        period = _period_seconds(rec)
        if period is None:
            return None
        nxt = alarm.fire_at + period
        while nxt <= now:  # skip any occurrences missed while away
            nxt += period
        return nxt

    def _announcement_text(self, alarm: Alarm, while_away: bool) -> str:
        name = (self._name() or "").strip()
        greet = f"Hey {name} — " if name else ""
        label = alarm.label
        if while_away:
            return f"{greet}while you were away, your {label} went off."
        return f"{greet}your {label} is up."

    async def _catch_up(self) -> None:
        """On start: fire/advance everything that came due while we were down."""
        now = self._now()
        for alarm in await self.list_active():
            if alarm.fire_at > now:
                continue
            is_one_shot = _period_seconds(alarm.recurrence) is None
            if is_one_shot and (now - alarm.fire_at) > self._stale_after:
                await self._deactivate(alarm.id)  # too old to be useful — drop it
            else:
                await self._fire(alarm, while_away=True)

    async def _deactivate(self, alarm_id: int) -> None:
        await self.db.execute("UPDATE alarms SET active=0 WHERE id=?", (alarm_id,))
        await self.db.commit()

    async def _reschedule(self, alarm_id: int, fire_at: float) -> None:
        await self.db.execute(
            "UPDATE alarms SET fire_at=? WHERE id=?", (fire_at, alarm_id)
        )
        await self.db.commit()
