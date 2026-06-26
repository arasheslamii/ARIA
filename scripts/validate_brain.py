"""Non-interactive real-API smoke test for Aria's brain (Step 2).

Runs four real turns through the orchestrator against the live Groq API — no mic,
no TUI — so we can confirm tool-calling, streaming, the new web search, memory,
timers, and the two-turn spoken confirmation all work end to end, and measure
rough latency.

Run it (uses the key from your keyring or $GROQ_API_KEY):

    uv run python scripts/validate_brain.py
"""

from __future__ import annotations

import asyncio
import time

from aria.app import MissingSecret, build_orchestrator
from aria.config.keyring import SecretStore
from aria.config.loader import load_config
from aria.core.memory import Memory
from aria.core.runtime import friendly_error


async def _say(orch, text: str) -> str:
    print(f"\n\033[1mYou:\033[0m {text}")
    print("\033[36mAria:\033[0m ", end="", flush=True)
    t0 = time.perf_counter()
    first = None
    parts: list[str] = []
    async for delta in orch.respond(text):
        if first is None:
            first = time.perf_counter() - t0
        parts.append(delta)
        print(delta, end="", flush=True)
    total = time.perf_counter() - t0
    print(f"\n  \033[2m[first token {1000 * (first or total):.0f}ms · total {total:.2f}s]\033[0m")
    return "".join(parts)


async def main() -> int:
    config = load_config()
    # Use a throwaway in-memory DB so the test doesn't pollute real memory.
    memory = Memory(db_path=":memory:")
    await memory.open()
    try:
        orch, managers = await build_orchestrator(config, SecretStore(), memory)
    except MissingSecret:
        print(friendly_error(MissingSecret()))
        return 1

    print(f"Models: reasoning={config.llm.reasoning_model}  fast={config.llm.fast_model}")
    await orch.warm_up()

    try:
        # 1) Web search (exercises the new DDG lite scraper + synthesis/citation).
        await _say(orch, "What's the latest news about the James Webb Space Telescope?")
        # 2) Long-term memory write.
        await _say(orch, "Remember that my name is Sam and I prefer Celsius.")
        # 3) Recall (separate turn) to prove persistence within the session.
        await _say(orch, "What's my name and which temperature unit do I like?")
        # 4) Timer.
        await _say(orch, "Set a 10 minute timer for the laundry.")
        # 5) Confirm-risk action -> should ASK first (two-turn spoken confirmation).
        q = await _say(orch, "Lock my screen.")
        asked = "yes or no" in q.lower()
        print(f"\n  \033[2m[confirmation question fired: {asked}]\033[0m")
        # 6) Decline it so nothing actually happens.
        await _say(orch, "No, don't.")
    except BaseException as exc:  # noqa: BLE001
        msg = friendly_error(exc)
        if msg:
            print(f"\n\033[31m{msg}\033[0m")
            return 1
        raise
    finally:
        await memory.close()
        for m in managers:
            await m.aclose()

    print("\n\033[32m✓ Brain validation complete.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
