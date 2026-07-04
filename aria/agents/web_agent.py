"""Browser-agent engine: drive a real Chromium with an LLM, and ALWAYS stop at
payment.

Built on `browser-use` (Playwright/Chromium in DOM/accessibility mode so a text
model can drive it). Two hard rules baked in:

  * **Stop at payment.** The guardrail system prompt forbids submitting payment /
    clicking "Place order" / confirming a purchase. The agent navigates, searches,
    fills the cart and address, reaches the checkout/payment page, and STOPS —
    handing the live browser to the human. There is NO autonomous-pay code path.
  * **Persistent profile.** A headful Chromium with a user-data-dir under
    ``state_dir()/browser_profile`` so the user logs into delivery sites once and
    cookies persist across runs.

The heavy deps (browser-use, playwright) are optional and imported lazily; if
they're missing — or there's no display — we raise a clear, spoken-friendly error
instead of crashing. The single async seam :func:`_run_agent` is what tests mock.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from aria.config.loader import state_dir
from aria.tools.base import ToolError

# The non-negotiable guardrail, prepended to EVERY browser task. Stage 1/2 both
# rely on this being present (a test asserts it's always injected).
GUARDRAIL = (
    "You are operating a real web browser on the user's behalf. You MAY search, "
    "navigate, sign in (if already logged in), add items to a cart, enter the "
    "delivery address, and proceed to the checkout / payment page. "
    "You MUST NEVER submit payment, click 'Place order', 'Pay', 'Buy now', "
    "'Confirm order', or otherwise complete a purchase. "
    "When you reach the payment or order-confirmation page, STOP immediately and "
    "report a concise summary: the shop name, the items, the order total, and the "
    "estimated delivery time. Do not take any further action after reaching "
    "payment — the human will pay. If you get stuck (a login wall, a captcha, or "
    "an ambiguous choice), STOP and report exactly where you are stuck."
)


class WebAgentError(ToolError):
    """A browser task could not run (no display, deps missing, or it got stuck).
    Spoken-friendly message; the orchestrator surfaces it as-is."""


@dataclass
class WebResult:
    summary: str
    shop: str = ""
    items: str = ""
    total: str = ""
    eta: str = ""
    url: str = ""
    stopped_at_payment: bool = False
    stuck: str | None = None  # reason if the agent stopped without finishing


def _display_available() -> bool:
    """A headful browser needs an X11/Wayland display. The systemd --user daemon
    only has one if the graphical session exported DISPLAY/XAUTHORITY to it."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def browser_profile_dir() -> Path:
    """Persistent Chromium user-data-dir so logins/cookies survive across runs."""
    return state_dir() / "browser_profile"


def chromium_installed() -> bool:
    """Best-effort: is a Playwright Chromium present? (We can't import playwright
    cheaply, so check the standard cache locations.)"""
    candidates = [
        Path.home() / ".cache" / "ms-playwright",
        Path(os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")) if os.environ.get(
            "PLAYWRIGHT_BROWSERS_PATH"
        ) else None,
    ]
    for base in candidates:
        if base and base.exists() and any(base.glob("chromium-*")):
            return True
    return False


def compose_task(task: str, *, profile: str | None = None) -> str:
    """Prepend the guardrail (and any private user profile) to the raw task. The
    guardrail is ALWAYS first so it can never be omitted."""
    parts = [GUARDRAIL]
    if profile:
        parts.append("USER PROFILE (use these details; never read them aloud):\n" + profile)
    parts.append("TASK:\n" + task.strip())
    return "\n\n".join(parts)


def _missing_deps_message() -> str:
    return (
        "The browsing engine isn't installed yet. Set it up once with "
        "`aria install-commerce` (or re-run `aria setup` and say yes when it "
        "offers to install it)."
    )


def _load_browser_use():  # pragma: no cover - exercised only with the real deps
    """Lazily import browser-use + a Chat LLM builder. Raises WebAgentError with a
    clear install hint if the optional deps aren't present."""
    try:
        from browser_use import Agent, Browser  # type: ignore
        from langchain_openai import ChatOpenAI  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise WebAgentError(_missing_deps_message()) from exc
    return Agent, Browser, ChatOpenAI


def _build_llm(ChatOpenAI, engine: dict):  # pragma: no cover - needs real lib
    return ChatOpenAI(
        model=engine["model"],
        base_url=engine["base_url"],
        api_key=engine.get("api_key") or "x",
        temperature=0.2,
    )


async def _run_agent(
    composed_task: str,
    engine: dict,
    *,
    profile_dir: Path,
    headful: bool,
    max_steps: int,
    max_seconds: float,
) -> dict:  # pragma: no cover - real browser path; tests monkeypatch this seam
    """Drive the real browser-use Agent to completion (or the step/time budget) and
    return a raw result dict. This is the ONLY place that touches the heavy deps, so
    tests replace it wholesale."""
    Agent, Browser, ChatOpenAI = _load_browser_use()
    if not chromium_installed():
        raise WebAgentError(_missing_deps_message())
    llm = _build_llm(ChatOpenAI, engine)
    browser = Browser(
        headless=not headful,
        user_data_dir=str(profile_dir),
    )
    agent = Agent(task=composed_task, llm=llm, browser=browser)
    try:
        history = await asyncio.wait_for(agent.run(max_steps=max_steps), timeout=max_seconds)
    except TimeoutError:
        return {"stuck": "timeout", "summary": "I ran out of time on that browse."}
    final = getattr(history, "final_result", lambda: "")() or str(history)
    return {"summary": final, "stopped_at_payment": True}


def _normalize(raw: dict) -> WebResult:
    return WebResult(
        summary=str(raw.get("summary", "")).strip(),
        shop=str(raw.get("shop", "")),
        items=str(raw.get("items", "")),
        total=str(raw.get("total", "")),
        eta=str(raw.get("eta", "")),
        url=str(raw.get("url", "")),
        stopped_at_payment=bool(raw.get("stopped_at_payment", False)),
        stuck=raw.get("stuck"),
    )


async def run_web_task(
    task: str,
    engine: dict,
    *,
    profile: str | None = None,
    profile_dir: Path | None = None,
    headful: bool = True,
    max_steps: int = 40,
    max_seconds: float = 240.0,
) -> WebResult:
    """Run a guard-railed browser task. Raises WebAgentError (spoken-friendly) if
    there's no display or the engine can't run; never pays."""
    if headful and not _display_available():
        raise WebAgentError(
            "I need your screen for that — open a graphical session and try again."
        )
    composed = compose_task(task, profile=profile)
    raw = await _run_agent(
        composed,
        engine,
        profile_dir=profile_dir or browser_profile_dir(),
        headful=headful,
        max_steps=max_steps,
        max_seconds=max_seconds,
    )
    return _normalize(raw)


def bring_browser_to_front() -> None:
    """Best-effort: raise the Chromium window so the human lands on the payment
    page. Silent no-op if no window manager helper is available."""
    for cmd in (["wmctrl", "-a", "Chromium"], ["wmctrl", "-a", "Chrome"]):
        if shutil.which(cmd[0]):
            try:
                import subprocess

                subprocess.run(cmd, timeout=3, check=False)
                return
            except Exception:  # noqa: BLE001 - cosmetic; never fail the order
                return
