"""Long-term memory in SQLite.

Two stores with DIFFERENT lifetimes:
  * facts  : durable facts and preferences the user explicitly asked to keep
             (user name, "remember I'm vegetarian"). Persist across days.
  * turns  : the conversation transcript — DAILY. Everything from before today
             is purged on open and at the first turn of a new day, so each
             morning starts fresh: no stale topics resurrected into greetings,
             and no transcript quietly accumulating on disk (privacy).

Short-term rolling context lives in the orchestrator; this is the persistent
layer. All access is async via aiosqlite.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import aiosqlite

from aria.config.loader import state_dir


def _start_of_today() -> float:
    now = datetime.now().astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    category   TEXT DEFAULT 'general',
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class Memory:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (state_dir() / "memory.sqlite3")
        self._db: aiosqlite.Connection | None = None
        self._purge_day: date | None = None

    async def open(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        await self.purge_old_turns()

    async def purge_old_turns(self) -> None:
        """Delete conversation turns from before today. Facts are untouched —
        the user explicitly asked for those; the transcript they didn't."""
        await self.db.execute("DELETE FROM turns WHERE created_at < ?", (_start_of_today(),))
        await self.db.commit()
        self._purge_day = date.today()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Memory.open() not called")
        return self._db

    # --- facts ---------------------------------------------------------
    async def remember(self, key: str, value: str, category: str = "general") -> None:
        await self.db.execute(
            "INSERT INTO facts(key, value, category, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "category=excluded.category, updated_at=excluded.updated_at",
            (key, value, category, time.time()),
        )
        await self.db.commit()

    async def recall(self, key: str) -> str | None:
        async with self.db.execute("SELECT value FROM facts WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
            return row[0] if row else None

    async def all_facts(self) -> dict[str, str]:
        async with self.db.execute("SELECT key, value FROM facts ORDER BY updated_at DESC") as cur:
            return {k: v async for k, v in cur}

    async def forget(self, key: str) -> None:
        await self.db.execute("DELETE FROM facts WHERE key=?", (key,))
        await self.db.commit()

    # --- turns ---------------------------------------------------------
    async def log_turn(self, role: str, content: str) -> None:
        if self._purge_day != date.today():
            # Day rolled over while the daemon stayed up: start the day fresh.
            await self.purge_old_turns()
        await self.db.execute(
            "INSERT INTO turns(role, content, created_at) VALUES(?,?,?)",
            (role, content, time.time()),
        )
        await self.db.commit()

    async def recent_turns(self, limit: int = 10) -> list[tuple[str, str]]:
        async with self.db.execute(
            "SELECT role, content FROM turns ORDER BY id DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
        return [(r, c) for r, c in reversed(rows)]
