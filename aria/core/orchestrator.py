"""The orchestrator: owns the conversation, routes intent, plans, dispatches to
tools/sub-agents, and streams a spoken-friendly reply.

Latency strategy:
  * 8B router picks the cheapest path. ``chitchat`` streams straight from the
    fast model to TTS — no tool round-trip.
  * ``tool``/``agentic`` resolve tool calls (run independent calls in parallel),
    then stream the final answer so sentence 1 is spoken while the rest is still
    being written.
  * Connections are pre-warmed in :meth:`warm_up`.

Risky actions use **two-turn spoken confirmation**: when the model wants to run a
``confirm``-risk tool, Aria asks out loud ("…should I go ahead?") and stashes the
in-flight tool calls. The user's *next* utterance is interpreted as yes/no — on
yes the stashed calls run and the answer streams; on no they're dropped. This
works hands-free over voice and identically in text mode, with no audio
re-entrancy.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from aria.core.debug import _truncate, dlog
from aria.core.executor import ExecConfig, ToolExecutor
from aria.core.memory import Memory
from aria.core.prompts import ORCHESTRATOR_SYSTEM
from aria.llm.base import LLMProvider, Message, ToolCall, assistant, system, tool_result, user
from aria.llm.router import route_intent
from aria.safety.audit import AuditLog
from aria.safety.permissions import classify
from aria.tools.base import ToolRegistry

_MAX_TOOL_STEPS = 5
_HISTORY_TURNS = 8

# Spoken when a SLOW tool round-trip begins, so the voice isn't silent for the
# multi-second cases. Trailing ". " makes the sentencizer flush it at once.
# Stripped from saved history so it never pollutes context. Fast single-tool
# turns (time, timers, system control) answer sub-second and get no filler.
_FILLER = "Let me look into that. "

# Only these warrant a filler — they involve network / multi-step work. Specialist
# sub-agents (``agent_*``) also qualify.
_SLOW_TOOLS = frozenset({"web_search", "read_webpage"})


def _is_slow_call(name: str) -> bool:
    return name in _SLOW_TOOLS or name.startswith("agent_")


# Spoken when a turn would otherwise end with no speech at all.
_NO_RESPONSE_FALLBACK = "Sorry, I'm not sure how to help with that one."

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|affirmative|correct|absolutely|"
    r"go ahead|go for it|do it|please do)\b"
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|don't|do not|cancel|stop|negative|never ?mind|forget it)\b"
)

# Args worth speaking aloud when describing a pending action.
_SALIENT_ARGS = ("to", "recipient", "query", "path", "name", "title", "expression")


@dataclass
class _Pending:
    """A tool step awaiting the user's spoken yes/no."""

    messages: list[Message]
    calls: list[ToolCall]
    question: str = ""
    specs: list = field(default_factory=list)


def interpret_yes_no(text: str) -> bool | None:
    """True/False for a clear yes/no, None if ambiguous (so we can re-ask)."""
    t = text.lower()
    yes, no = bool(_YES_RE.search(t)), bool(_NO_RE.search(t))
    if yes and not no:
        return True
    if no and not yes:
        return False
    return None


class Orchestrator:
    def __init__(
        self,
        *,
        llm: LLMProvider,
        registry: ToolRegistry,
        memory: Memory,
        reasoning_model: str,
        fast_model: str,
        require_confirmation: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        self._require_confirmation = require_confirmation
        self._executor = ToolExecutor(AuditLog(), ExecConfig(require_confirmation=False))
        self._history: list[Message] = []
        self._pending: _Pending | None = None
        # Sources found in the most recent research turn, kept in context so a
        # follow-up ("read the second one", "go deeper") can act on the real URLs.
        self._last_sources: list[dict[str, str]] = []

    async def warm_up(self) -> None:
        """Pre-establish connections AND warm both models so the first real turn
        (including the first tool turn) isn't cold."""

        async def ping(model: str) -> None:
            try:
                await self.llm.chat([user("hi")], model=model, temperature=0.0, max_tokens=1)
            except Exception:  # noqa: BLE001 - warm-up is best-effort
                pass

        # Warm the fast router and the reasoning model concurrently.
        models = {self.fast_model, self.reasoning_model}
        await asyncio.gather(*(ping(m) for m in models))

    async def respond(self, transcript: str) -> AsyncIterator[str]:
        """Main entry from the voice pipeline. Yields spoken-text deltas."""
        await self.memory.log_turn("user", transcript)
        self._history.append(user(transcript))
        self._trim_history()

        if self._pending is not None:
            stream = self._resume_pending(transcript)
        else:
            decision = await route_intent(
                self.llm, self.fast_model, transcript, self.registry.catalog()
            )
            dlog(f"route={decision.route} needs_tools={decision.needs_tools} "
                 f"({decision.reason})")
            stream = (
                self._chitchat(transcript)
                if decision.route == "chitchat"
                else self._agentic(decision)
            )

        spoken_parts: list[str] = []
        async for delta in stream:
            spoken_parts.append(delta)
            yield delta

        # Keep the filler out of saved history so it doesn't accrete across turns.
        full = "".join(spoken_parts).replace(_FILLER, "").strip()
        if not full:
            # Never end a turn silent: if nothing was said (model returned empty,
            # a tool failed without speaking), say so out loud.
            full = _NO_RESPONSE_FALLBACK
            yield full
        self._history.append(assistant(full))
        await self.memory.log_turn("assistant", full)

    # --- fast path -----------------------------------------------------
    async def _chitchat(self, transcript: str) -> AsyncIterator[str]:
        messages = await self._base_messages()
        async for delta in self.llm.stream(
            messages, model=self.fast_model, temperature=0.5, max_tokens=300
        ):
            yield delta

    # --- tool / agentic path ------------------------------------------
    async def _agentic(self, decision) -> AsyncIterator[str]:
        messages = await self._base_messages()
        specs = self._select_specs(decision)
        dlog(f"exposed tools={[s.name for s in specs]}")
        async for delta in self._tool_loop(messages, specs):
            yield delta

    def _select_specs(self, decision) -> list:
        """Shrink the tool surface to what the router thinks the turn needs.

        Fewer tools per request both reduces the malformed-tool-call rate and
        cuts latency. The "agentic"/planning route also gets the specialist
        sub-agents to delegate to; the "tool" route stays minimal.
        """
        include_agents = decision.route == "agentic"
        specs = self.registry.specs_for(decision.needs_tools, include_agents=include_agents)
        # Belt-and-suspenders: if the router named tools but none resolved (so we'd
        # only have core), expose the full native toolset rather than stranding the
        # model — a missing tool is what makes it fabricate success.
        names = {s.name for s in specs}
        if decision.needs_tools and names <= set(self.registry.CORE_TOOLS):
            specs = self.registry.all_specs(include_agents=include_agents)
        return specs or self.registry.all_specs(include_agents=include_agents)

    async def _tool_loop(self, messages: list[Message], specs: list) -> AsyncIterator[str]:
        """Resolve tool calls until the model produces a final spoken answer.

        Returns early (after asking aloud) if a step needs confirmation; the
        turn resumes from the stashed state via :meth:`_resume_pending`.
        """
        filler_spoken = False
        for _ in range(_MAX_TOOL_STEPS):
            result = await self.llm.chat(
                messages, model=self.reasoning_model, tools=specs, temperature=0.3
            )
            if not result.tool_calls:
                async for delta in self._final_answer(messages, result):
                    yield delta
                return

            messages.append(assistant(result.content, result.tool_calls))
            gated = [c for c in result.tool_calls if self._needs_confirmation(c)]
            slow = any(_is_slow_call(c.name) for c in result.tool_calls)
            if slow and not gated and not filler_spoken:
                # A genuinely slow (network / multi-step) round-trip is starting —
                # say a short filler so the voice isn't silent. Fast tools skip it.
                # Stripped from saved history so it doesn't pollute context.
                filler_spoken = True
                yield _FILLER
            if gated:
                self._pending = _Pending(
                    messages=messages,
                    calls=result.tool_calls,
                    question=self._confirm_question(gated),
                    specs=specs,
                )
                yield self._pending.question
                return

            await self._execute_calls(messages, result.tool_calls)

        # Exhausted steps: stream a wrap-up.
        async for delta in self.llm.stream(
            messages, model=self.reasoning_model, temperature=0.4, max_tokens=300
        ):
            yield delta

    async def _resume_pending(self, transcript: str) -> AsyncIterator[str]:
        """Interpret the user's yes/no and continue (or drop) the stashed step."""
        pending = self._pending
        assert pending is not None
        answer = interpret_yes_no(transcript)
        if answer is None:
            yield "Sorry, I didn't catch that — should I go ahead? Please say yes or no."
            return  # keep pending for another try

        self._pending = None
        if answer is False:
            # Declined: acknowledge cleanly and stop. Don't feed "user declined"
            # back to the model — it produced confusing freeform replies.
            yield "Okay, I won't."
            return

        await self._execute_calls(pending.messages, pending.calls)
        async for delta in self._tool_loop(pending.messages, pending.specs):
            yield delta

    async def _final_answer(self, messages: list[Message], result) -> AsyncIterator[str]:
        follow = (
            messages + [user("Now answer me out loud, briefly.")]
            if not result.content
            else messages
        )
        async for delta in self.llm.stream(
            follow, model=self.reasoning_model, temperature=0.4, max_tokens=400
        ):
            yield delta

    async def _execute_calls(self, messages: list[Message], calls: list[ToolCall]) -> None:
        # Run independent tool calls concurrently. Confirmation is handled above
        # (deferred), so the executor runs vetted calls directly.
        outputs = await asyncio.gather(
            *(self._dispatch(call.name, call.arguments) for call in calls)
        )
        for call, out in zip(calls, outputs):
            messages.append(tool_result(call.id, out, name=call.name))

    async def _dispatch(self, name: str, arguments: dict | None) -> str:
        arguments = arguments or {}
        dlog(f"tool call -> {name}({arguments})")
        tool = self.registry.get(name)
        if tool is None:
            dlog(f"tool {name} NOT FOUND")
            return f"error: unknown tool {name}"
        res = await self._executor.execute(tool, arguments, confirm=_auto_approve)
        self._capture_sources(res)
        dlog(f"tool {name} result: {_truncate(res.content)}")
        return res.content

    def _capture_sources(self, res) -> None:
        """Remember URLs from research/search so follow-ups can read them."""
        data = res.data or {}
        found = data.get("sources") or data.get("results")
        if not found:
            if data.get("url"):
                found = [{"title": data.get("title", ""), "url": data["url"]}]
            else:
                return
        collected = [
            {"title": s.get("title") or s.get("url", ""), "url": s["url"]}
            for s in found
            if s.get("url")
        ]
        if collected:
            self._last_sources = collected[:6]

    def _needs_confirmation(self, call: ToolCall) -> bool:
        if not self._require_confirmation:
            return False
        tool = self.registry.get(call.name)
        if tool is None:
            return False
        return classify(tool, call.arguments).risk == "confirm"

    def _confirm_question(self, calls: list[ToolCall]) -> str:
        actions = " and ".join(self._describe(c) for c in calls)
        return f"You want me to {actions}. Should I go ahead? Say yes or no."

    def _describe(self, call: ToolCall) -> str:
        verb = call.name.replace("_", " ")
        for key in _SALIENT_ARGS:
            val = call.arguments.get(key)
            if val:
                return f"{verb} {val}".strip()
        return verb

    # --- context -------------------------------------------------------
    async def _base_messages(self) -> list[Message]:
        facts = await self.memory.all_facts()
        sys = ORCHESTRATOR_SYSTEM
        if facts:
            known = "; ".join(f"{k}={v}" for k, v in list(facts.items())[:20])
            sys += f"\n\nWhat you already know about the user: {known}."
        if self._last_sources:
            srcs = "; ".join(
                f"[{i + 1}] {s['title']} — {s['url']}"
                for i, s in enumerate(self._last_sources)
            )
            sys += (
                f"\n\nSources you just found (use read_webpage on these if the user "
                f"asks to go deeper, read one, or 'the second one'): {srcs}."
            )
        return [system(sys), *self._history]

    def _trim_history(self) -> None:
        if len(self._history) > _HISTORY_TURNS * 2:
            self._history = self._history[-_HISTORY_TURNS * 2 :]


async def _auto_approve(_name: str) -> bool:
    return True
