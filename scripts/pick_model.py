"""Benchmark Groq's LIVE model list and recommend the best reasoning/synthesis model.

Queries the /models endpoint, then tests the strongest tool-capable candidates on:
  * tool-call reliability (does it emit a well-formed function call when asked?),
  * synthesis quality proxy (length/coherence of a grounded-summary answer),
  * latency (time to a small completion).

Run it with your key in the keyring or $GROQ_API_KEY:

    uv run python scripts/pick_model.py

Then set the winner in ~/.config/aria/config.toml:
    [llm]
    synthesis_model = "moonshotai/kimi-k2-instruct"   # or whatever it recommends
"""

from __future__ import annotations

import asyncio
import time

from aria.config.keyring import SecretStore
from aria.llm.base import ToolSpec, system, user
from aria.llm.groq_provider import GroqProvider

# Candidates to consider if present in the live list (strongest first).
_PREFERRED = [
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-instruct-0905",
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen-2.5-72b-instruct",
    "llama-3.1-70b-versatile",
]

_WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


async def _list_models(provider: GroqProvider) -> list[str]:
    # The groq SDK exposes models.list(); fall back to an empty list on error.
    try:
        resp = await provider._client.models.list()  # noqa: SLF001 - simple probe
        return [m.id for m in resp.data]
    except Exception as exc:  # noqa: BLE001
        print(f"  (couldn't list models: {exc})")
        return []


async def _bench(provider: GroqProvider, model: str) -> dict:
    out: dict = {"model": model}
    # 1) tool-call reliability
    t0 = time.perf_counter()
    try:
        res = await provider.chat(
            [user("What's the weather in Paris? Use the tool.")],
            model=model,
            tools=[_WEATHER_TOOL],
        )
        out["tool_ok"] = bool(res.tool_calls and res.tool_calls[0].name == "get_weather")
    except Exception as exc:  # noqa: BLE001
        out["tool_ok"] = False
        out["tool_err"] = str(exc)[:80]
    out["tool_latency_s"] = round(time.perf_counter() - t0, 2)

    # 2) synthesis quality proxy + latency
    t0 = time.perf_counter()
    try:
        res = await provider.chat(
            [
                system("Summarize warmly and concisely for a voice assistant."),
                user(
                    "From this text only: 'The FTSE 100 fell 2% on Tuesday on inflation "
                    "fears, its worst day in months.' — tell me what happened."
                ),
            ],
            model=model,
            max_tokens=120,
        )
        out["synth_chars"] = len(res.content)
        out["grounded"] = "2%" in res.content or "two percent" in res.content.lower()
    except Exception as exc:  # noqa: BLE001
        out["synth_chars"] = 0
        out["synth_err"] = str(exc)[:80]
    out["synth_latency_s"] = round(time.perf_counter() - t0, 2)
    return out


async def main() -> int:
    key = SecretStore().get("groq_api_key")
    if not key:
        print("No Groq API key (keyring or $GROQ_API_KEY). Run `aria setup` first.")
        return 1
    provider = GroqProvider(key)

    print("Listing live Groq models…")
    available = set(await _list_models(provider))
    candidates = [m for m in _PREFERRED if m in available] or sorted(available)[:4]
    print(f"Benchmarking: {candidates}\n")

    rows = [await _bench(provider, m) for m in candidates]
    await provider.aclose()

    print(f"\n{'model':40} tool  tool_s  synth_s  grounded  chars")
    for r in rows:
        print(
            f"{r['model']:40} "
            f"{'✓' if r.get('tool_ok') else '✗':>4}  "
            f"{r.get('tool_latency_s', 0):>5}  "
            f"{r.get('synth_latency_s', 0):>6}  "
            f"{'✓' if r.get('grounded') else '✗':>8}  "
            f"{r.get('synth_chars', 0):>5}"
        )

    # Recommend: best grounded synthesis among the tool-reliable ones.
    reliable = [r for r in rows if r.get("tool_ok")]
    best = max(reliable or rows, key=lambda r: (r.get("grounded", False), r.get("synth_chars", 0)))
    print(f"\nRecommended synthesis_model: {best['model']}")
    print("Set it in ~/.config/aria/config.toml under [llm] synthesis_model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
