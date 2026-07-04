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
from datetime import datetime

from aria.core import lines
from aria.core.debug import _truncate, dlog
from aria.core.executor import ExecConfig, ToolExecutor
from aria.core.memory import Memory
from aria.core.prompts import ORCHESTRATOR_SYSTEM, SYNTHESIS_GROUNDING, VOICE_BREVITY
from aria.llm.base import (
    LLMConnectionError,
    LLMProvider,
    LLMRateLimitError,
    Message,
    ToolCall,
    assistant,
    system,
    tool_result,
    user,
)
from aria.llm.router import route_intent
from aria.safety.audit import AuditLog
from aria.safety.permissions import classify
from aria.tools.base import ToolRegistry

_MAX_TOOL_STEPS = 8  # room for multi-step chains (e.g. read email -> add to calendar)
_HISTORY_TURNS = 6  # trimmed to save tokens on the free tier; still enough context

# Spoken when a SLOW tool round-trip begins, so the voice isn't silent for the
# multi-second cases. Trailing ". " makes the sentencizer flush it at once.
# Stripped from saved history so it never pollutes context. Fast single-tool
# turns (time, timers, system control) answer sub-second and get no filler.
# This constant is the default; per-turn the line is varied via lines.FILLERS.
_FILLER = "Let me look into that. "

# A confirmation reply longer than this goes to the fast model instead of the
# yes/no regex — "yes but make it 20 minutes" must be read as an amendment, not
# a plain yes.
_MAX_REGEX_REPLY_WORDS = 4

# Only these warrant a filler — they involve network / multi-step work. Specialist
# sub-agents (``agent_*``) also qualify.
_SLOW_TOOLS = frozenset(
    {
        "web_search", "read_webpage", "get_headlines",
        "list_recent_emails", "search_emails", "read_email", "list_events",
        "browse_web", "order_food",  # real-browser turns are the slowest of all
        "reserve_hotel", "reserve_flight",  # agentic picks drive the browser too
    }
)


def _is_slow_call(name: str) -> bool:
    return name in _SLOW_TOOLS or name.startswith("agent_")


# Kept as stable names for callers/tests; the spoken line itself is varied via
# the pools in aria.core.lines so she never repeats the exact same stock phrase.
_NO_RESPONSE_FALLBACK = lines.NO_RESPONSE[0]
_RATE_LIMIT_MSG = lines.RATE_LIMITED[0]
_OFFLINE_MSG = lines.OFFLINE[0]

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|confirm|affirmative|correct|absolutely|"
    r"go ahead|go for it|do it|please do)\b"
)
_NO_RE = re.compile(
    r"\b(no|nope|nah|don't|do not|cancel|stop|negative|never ?mind|forget it)\b"
)

# Args worth speaking aloud when describing a pending action.
_SALIENT_ARGS = ("to", "recipient", "query", "path", "name", "title", "expression")

# Phrases Whisper INVENTS when a capture is mostly silence or noise (a known
# failure mode of the model family — it was trained on captioned video, so quiet
# audio decodes to outro boilerplate). An open-mic capture whose ENTIRE content
# is one of these is a ghost, not a turn: answering it makes Aria talk to an
# empty room, which re-opens the window and snowballs into her "asking things"
# by herself. Wake-word and confirmation captures are never checked — the user
# explicitly invoked those. Matched on the lowercased, punctuation-free text.
_STT_GHOSTS = frozenset({
    "thank you", "thanks", "thank you for watching", "thanks for watching",
    "thank you so much", "thank you very much", "thank you so much for watching",
    "thank you for watching this video", "please subscribe", "subscribe",
    "see you", "see you next time", "see you in the next video",
    "bye", "bye bye", "goodbye", "the end",
    "you", "so", "yeah", "okay", "uh", "um", "hmm", "oh",
})


def _normalize_utterance(text: str) -> str:
    return " ".join(re.sub(r"[^a-z']+", " ", text.lower()).split())


# Openers that mark an open-mic capture as OBVIOUSLY for the assistant — a
# question word, a connective continuing the last answer, thanks, or a command
# verb Aria's tools cover. Anchored at the start so "John, can you…" (addressed
# to John) still goes to the model for judgement.
_CONTINUATION_CUES = re.compile(
    r"^(and |also |then |no[, ]|yes[,. ]|okay[, ]|ok[, ]|wait |actually |sorry |"
    r"what|how |why |when |where |who |which |can you |could you |please |"
    r"tell |show |set |turn |play |pause |stop |skip |next |volume |"
    r"remind |remember |search |read |open |order |send |check |cancel |snooze |"
    r"add |make |thanks|thank you|never ?mind|go deeper|more |again )",
    re.IGNORECASE,
)

# Validated against local 3B models (llama3.2:3b 9/10, qwen2.5:3b 8/10 on the
# probe set; the misses are covered by _CONTINUATION_CUES above). Lenient by
# design: answering the TV once is awkward, eating the user's turn is worse.
_FOLLOWUP_GATE_PROMPT = """A home voice assistant just spoke, and its microphone \
caught a reply. It is USUALLY the user continuing the conversation with the \
assistant — answer yes unless there is a CLEAR sign the words are for another \
person.

Requests and commands (volume, timers, reminders, weather, music) are FOR the \
assistant: answer yes.
Answer no only when the words are clearly aimed at another person in the room: \
they address someone by name or pet name (honey, love, John), invite someone to \
come/sit/eat, chat about other people, or sound like TV or a phone call.

Examples:
HEARD: and what about tomorrow? -> yes
HEARD: thanks, that was helpful -> yes
HEARD: turn the volume down please -> yes
HEARD: remind me to call the dentist -> yes
HEARD: honey, dinner is ready, come sit down -> no
HEARD: John, can you grab the remote -> no
HEARD: did you see what Maria said at work today -> no

ASSISTANT SAID: {last}
HEARD: {heard}
Answer exactly one word, yes or no:"""


@dataclass
class _Pending:
    """A tool step awaiting the user's spoken yes/no."""

    messages: list[Message]
    calls: list[ToolCall]
    question: str = ""
    specs: list = field(default_factory=list)


def _verdict_word(text: str | None, options: tuple[str, ...]) -> str | None:
    """First of ``options`` appearing as a whole word in a micro-verdict reply.
    Tolerates punctuation and role-echo prefixes some local templates leak."""
    m = re.search(r"\b(" + "|".join(options) + r")\b", (text or "").lower())
    return m.group(1) if m else None


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
        synthesis_model: str | None = None,
        require_confirmation: bool = True,
        voice: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        # Voice turns are spoken aloud, so the final synthesis is told to keep it to
        # one or two sentences (chat mode stays full-length).
        self._voice = voice
        # Stronger model for final spoken synthesis only; verified in warm_up and
        # falls back to reasoning_model if it isn't available on the provider.
        self._synthesis_model = synthesis_model or reasoning_model
        self._require_confirmation = require_confirmation
        self._executor = ToolExecutor(AuditLog(), ExecConfig(require_confirmation=False))
        self._history: list[Message] = []
        self._pending: _Pending | None = None
        # The filler/narration spoken this turn (generic or a tool's slow_filler),
        # tracked so it can be stripped from saved history.
        self._turn_filler: str = _FILLER
        # Sources found in the most recent research turn, kept in context so a
        # follow-up ("read the second one", "go deeper") can act on the real URLs.
        self._last_sources: list[dict[str, str]] = []
        # Conversation memory beyond the trimmed window: turns dropped by
        # _trim_history accumulate here and are folded (in the background) into a
        # compact running summary, so a long conversation never loses its thread.
        self._summary = ""
        self._dropped: list[Message] = []
        self._absorb_task: asyncio.Task | None = None
        # One-line recall of the PREVIOUS session, built at warm-up, so she can
        # pick up where you left off yesterday instead of starting cold.
        self._prev_session_note = ""

    @property
    def awaiting_reply(self) -> bool:
        """True when the last turn ended expecting the user's answer (a pending
        yes/no confirmation). The voice pipeline reads this to re-open the mic right
        after Aria speaks, so the user can just say "yes" with no wake word."""
        return self._pending is not None

    async def warm_up(self) -> None:
        """Pre-establish connections, warm the models, and verify the optional
        synthesis model is actually available (else fall back to reasoning_model)."""

        async def ping(model: str) -> bool:
            try:
                await self.llm.chat([user("hi")], model=model, temperature=0.0, max_tokens=1)
                return True
            except Exception:  # noqa: BLE001 - warm-up is best-effort
                return False

        models = [self.fast_model, self.reasoning_model]
        synth = self._synthesis_model
        probe_synth = synth not in models
        if probe_synth:
            models.append(synth)
        results = await asyncio.gather(*(ping(m) for m in models))
        if probe_synth and not results[-1]:
            # Strong synthesis model isn't available here — degrade gracefully so
            # answers still work (just on the reasoning model) with no per-turn cost.
            dlog(f"synthesis_model {synth!r} unavailable; using {self.reasoning_model}")
            self._synthesis_model = self.reasoning_model
        await self._recall_last_session()

    async def _recall_last_session(self) -> None:
        """Summarize the tail of the previous session (best-effort, one fast call)
        so 'what were we talking about yesterday?' actually works."""
        try:
            turns = await self.memory.recent_turns(12)
            if not turns:
                return
            convo = "\n".join(f"{role}: {content}" for role, content in turns)
            prompt = (
                "Here is the end of a voice assistant's previous conversation with "
                f"its user:\n{convo}\n\n"
                "In under 40 words, note what it was about and anything left open "
                "(tasks, questions, plans). Reply with the note only."
            )
            result = await self.llm.chat(
                [user(prompt)], model=self.fast_model, temperature=0.2, max_tokens=90
            )
            self._prev_session_note = (result.content or "").strip()[:400]
        except Exception:  # noqa: BLE001 - recall is a nicety, never a blocker
            self._prev_session_note = ""

    async def respond(self, transcript: str) -> AsyncIterator[str]:
        """Main entry from the voice pipeline. Yields spoken-text deltas."""
        await self.memory.log_turn("user", transcript)
        self._history.append(user(transcript))
        self._trim_history()

        spoken_parts: list[str] = []
        self._turn_filler = _FILLER
        try:
            # Routing (the 8B call) is also under the guard, so a 429 there can't
            # crash the loop either — the whole turn degrades to a warm line.
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
            async for delta in stream:
                spoken_parts.append(delta)
                yield delta
        except LLMRateLimitError:
            # A transient cap (e.g. Groq's free daily limit) must NOT crash the
            # loop — speak a warm line and let the next turn try again.
            self._pending = None
            msg = lines.pick(lines.RATE_LIMITED)
            spoken_parts.append(msg)
            yield msg
        except LLMConnectionError:
            self._pending = None
            msg = lines.pick(lines.OFFLINE)
            spoken_parts.append(msg)
            yield msg

        # Keep the filler/narration out of saved history so it doesn't accrete.
        full = "".join(spoken_parts).replace(self._turn_filler, "").replace(_FILLER, "").strip()
        if not full:
            # Never end a turn silent: if nothing was said (model returned empty,
            # a tool failed without speaking), say so out loud.
            full = lines.pick(lines.NO_RESPONSE)
            yield full
        self._history.append(assistant(full))
        await self.memory.log_turn("assistant", full)

    def _synthesis_guidance(self, *, grounding: bool) -> list[Message]:
        """Extra system messages appended right before the spoken answer is
        generated: a hard tool-faithfulness rule (after tool calls) and, on voice
        turns, a brevity instruction."""
        extra: list[Message] = []
        if grounding:
            extra.append(system(SYNTHESIS_GROUNDING))
        if self._voice:
            extra.append(system(VOICE_BREVITY))
        return extra

    # --- fast path -----------------------------------------------------
    async def _chitchat(self, transcript: str) -> AsyncIterator[str]:
        # Chitchat is where personality lives, so it gets the STRONG model — the
        # 8B router models produce the flat, dumb-sounding small talk. Streaming
        # keeps first-word latency low even on the big model.
        messages = await self._base_messages()
        messages = messages + self._synthesis_guidance(grounding=False)
        async for delta in self.llm.stream(
            messages, model=self._synthesis_model, temperature=0.6, max_tokens=300
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
            slow_calls = [c for c in result.tool_calls if _is_slow_call(c.name)]
            if slow_calls and not gated and not filler_spoken:
                # A genuinely slow (network / multi-step) round-trip is starting —
                # say a short filler so the voice isn't silent. Fast tools skip it.
                # Stripped from saved history so it doesn't pollute context.
                filler_spoken = True
                yield self._filler_for(slow_calls)
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

        # Exhausted steps: stream a wrap-up (synthesis model), still grounded.
        async for delta in self.llm.stream(
            messages + self._synthesis_guidance(grounding=True),
            model=self._synthesis_model, temperature=0.4, max_tokens=300,
        ):
            yield delta

    async def _resume_pending(self, transcript: str) -> AsyncIterator[str]:
        """Interpret the user's reply to a confirmation and continue (yes), drop
        (no), or ADJUST (anything else) the stashed step.

        A short clear reply is resolved by regex (free, instant). Anything longer
        or unclear goes to the fast model, which may also classify it as a change
        request ("yes but make it 20 minutes", "send it to alice instead") — that
        re-enters the tool loop with the user's words so the model re-plans and
        re-confirms, instead of forcing a robotic yes/no standoff."""
        pending = self._pending
        assert pending is not None
        verdict: str | None = None
        if len(transcript.split()) <= _MAX_REGEX_REPLY_WORDS:
            answer = interpret_yes_no(transcript)
            verdict = {True: "yes", False: "no", None: None}[answer]
        if verdict is None:
            verdict = await self._classify_confirmation(transcript, pending.question)
        if verdict == "unclear":
            yield lines.pick(lines.CONFIRM_REASKS)
            return  # keep pending for another try

        self._pending = None
        if verdict == "no":
            # Declined: acknowledge cleanly and stop. Don't feed "user declined"
            # back to the model — it produced confusing freeform replies.
            yield lines.pick(lines.DECLINED)
            return

        if verdict == "change":
            # The user amended the plan. Drop the un-run tool-call step (its calls
            # never got results, and providers reject dangling tool_calls), give
            # the model the exchange as plain turns, and let it re-plan — the new
            # gated call will ask for confirmation again.
            messages = pending.messages
            if messages and messages[-1].role == "assistant" and messages[-1].tool_calls:
                messages.pop()
            messages.append(assistant(pending.question))
            messages.append(user(transcript))
            async for delta in self._tool_loop(messages, pending.specs):
                yield delta
            return

        # A confirmed slow action (e.g. order_food's browse) runs now with no model
        # round-trip first, so narrate before it starts or the voice goes silent.
        slow_calls = [c for c in pending.calls if _is_slow_call(c.name)]
        if slow_calls:
            yield self._filler_for(slow_calls)
        await self._execute_calls(pending.messages, pending.calls)
        async for delta in self._tool_loop(pending.messages, pending.specs):
            yield delta

    async def _classify_confirmation(self, transcript: str, question: str) -> str:
        """Fast-model read of a confirmation reply: yes | no | change | unclear.
        Fails CLOSED (unclear -> re-ask): a garbled reply must never spend money.

        NOTE: the instruction goes as a USER message — some local chat templates
        (Ollama llama3.x) mangle a system-only conversation into a leaked role
        header instead of an answer."""
        prompt = (
            "The assistant asked the user to confirm an action:\n"
            f"ASSISTANT: {question}\n"
            f"USER REPLIED: {transcript}\n\n"
            "Classify the reply. Answer with exactly one word:\n"
            "yes    - they approve the action as described\n"
            "no     - they decline it\n"
            "change - they want it done differently (an amendment, correction, or "
            "new detail)\n"
            "unclear- you cannot tell"
        )
        try:
            result = await self.llm.chat(
                [user(prompt)], model=self.fast_model, temperature=0.0, max_tokens=8
            )
            verdict = _verdict_word(result.content, ("yes", "no", "change"))
        except Exception:  # noqa: BLE001 - any failure -> re-ask, never guess
            return "unclear"
        return verdict or "unclear"

    async def accept_followup(self, transcript: str) -> bool:
        """Was this open-mic capture actually meant for Aria? Conversation mode
        re-opens the mic after every answer, so the TV or a side conversation can
        land here. Three tiers, each biased towards NOT eating a real user turn:

        1. a pending confirmation reply is always accepted;
        2. a known Whisper silence-hallucination ("Thank you.") is rejected
           outright — it is what the STT says when NOBODY spoke;
        3. an utterance that OPENS like a command/continuation ("and…", "what…",
           "turn…", "remind…", "thanks…") is accepted instantly — no LLM call,
           no latency, and immune to small-model misjudgement;
        4. everything else is judged by the fast model with a lenient, few-shot
           prompt (validated against local 3B models), failing OPEN."""
        if self._pending is not None:
            return True
        if _normalize_utterance(transcript) in _STT_GHOSTS:
            dlog(f"followup dropped as an STT ghost: {transcript!r}")
            return False
        if _CONTINUATION_CUES.match(transcript.strip()):
            return True
        last = next(
            (m.content for m in reversed(self._history) if m.role == "assistant" and m.content),
            "",
        )
        prompt = _FOLLOWUP_GATE_PROMPT.format(last=last[:300], heard=transcript)
        try:
            result = await self.llm.chat(
                [user(prompt)], model=self.fast_model, temperature=0.0, max_tokens=8
            )
        except Exception:  # noqa: BLE001
            return True
        accepted = _verdict_word(result.content, ("yes", "no")) != "no"  # fail open
        if not accepted:
            dlog(f"followup ignored as background speech: {transcript!r}")
        return accepted

    async def _final_answer(self, messages: list[Message], result) -> AsyncIterator[str]:
        # Did this turn run any tools? If so, the answer must be grounded in their
        # real results (no invented success / time / "not connected").
        grounding = any(m.role == "tool" for m in messages)
        follow = (
            messages + [user("Now answer me out loud, briefly.")]
            if not result.content
            else list(messages)
        )
        follow = follow + self._synthesis_guidance(grounding=grounding)
        # Generous budget so a "go deeper / read it fully" turn can be extensive;
        # the persona keeps ordinary answers tight, so this only stretches when the
        # user actually asked for depth. Synthesis uses the stronger model.
        async for delta in self.llm.stream(
            follow, model=self._synthesis_model, temperature=0.4, max_tokens=800
        ):
            yield delta

    async def _execute_calls(self, messages: list[Message], calls: list[ToolCall]) -> None:
        # Run independent tool calls concurrently. Confirmation is handled above
        # (deferred), so the executor runs vetted calls directly.
        outputs = await asyncio.gather(
            *(self._dispatch(call.name, call.arguments) for call in calls)
        )
        for call, out in zip(calls, outputs, strict=True):
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
        shown = "[redacted]" if getattr(tool, "sensitive", False) else _truncate(res.content)
        dlog(f"tool {name} result: {shown}")
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
        # Commerce (spending money / placing an order) is gated just like confirm.
        return classify(tool, call.arguments).risk in ("confirm", "commerce")

    def _filler_for(self, calls: list[ToolCall]) -> str:
        """The narration spoken when a slow round-trip starts. A single slow tool
        may declare its own line (``slow_filler``, e.g. order_food's); otherwise a
        varied generic one. Tracked so respond() can strip it from saved history."""
        text = lines.pick(lines.FILLERS)
        if len(calls) == 1:
            tool = self.registry.get(calls[0].name)
            custom = getattr(tool, "slow_filler", None)
            if custom:
                text = custom if custom.endswith(" ") else custom + " "
        self._turn_filler = text
        return text

    def _confirm_question(self, calls: list[ToolCall]) -> str:
        # The ACTION text is verbatim (it may carry money amounts or recipients);
        # only the conversational frame around it varies.
        actions = " and ".join(self._describe(c) for c in calls)
        return lines.pick(lines.CONFIRM_FRAMES).format(action=actions)

    def _describe(self, call: ToolCall) -> str:
        # Prefer the tool's own read-back (e.g. the full email) for confirm actions.
        tool = self.registry.get(call.name)
        if tool is not None:
            summary = tool.confirm_summary(call.arguments)
            if summary:
                return summary
        verb = call.name.replace("_", " ")
        for key in _SALIENT_ARGS:
            val = call.arguments.get(key)
            if val:
                return f"{verb} {val}".strip()
        return verb

    # --- context -------------------------------------------------------
    async def _base_messages(self) -> list[Message]:
        facts = await self.memory.all_facts()
        # Anchor all relative dates ("tomorrow", "Friday") to the real local date.
        now = datetime.now().astimezone()
        sys = ORCHESTRATOR_SYSTEM + (
            f"\n\nToday is {now.strftime('%A, %-d %B %Y')}, and the local time is "
            f"{now.strftime('%-I:%M %p')}. Use this for any relative dates/times."
        )
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
        if self._summary:
            sys += f"\n\nEarlier in this conversation (summary): {self._summary}"
        if self._prev_session_note:
            sys += (
                f"\n\nFrom your previous conversation with them: "
                f"{self._prev_session_note}"
            )
        return [system(sys), *self._history]

    def _trim_history(self) -> None:
        limit = _HISTORY_TURNS * 2
        if len(self._history) > limit:
            cut = len(self._history) - limit
            # Don't just forget trimmed turns — queue them for the summary.
            self._dropped.extend(m for m in self._history[:cut] if m.content)
            self._history = self._history[cut:]
        if len(self._dropped) >= 6 and (self._absorb_task is None or self._absorb_task.done()):
            batch, self._dropped = self._dropped, []
            # Background fold-in: runs while the reply is being generated, so it
            # adds no latency to the turn. Best-effort — a failure just means the
            # summary lags a little.
            self._absorb_task = asyncio.create_task(self._absorb(batch))

    async def _absorb(self, batch: list[Message]) -> None:
        """Fold dropped turns into the compact running summary."""
        convo = "\n".join(
            f"{m.role}: {m.content}" for m in batch if m.role in ("user", "assistant")
        )
        if not convo:
            return
        prompt = (
            "You maintain a compact running summary of an ongoing voice "
            f"conversation.\nCURRENT SUMMARY: {self._summary or '(empty)'}\n"
            f"OLDER TURNS TO FOLD IN:\n{convo}\n\n"
            "Reply with the updated summary only — under 80 words. Keep names, "
            "facts, decisions, open tasks, and preferences; drop pleasantries."
        )
        try:
            result = await self.llm.chat(
                [user(prompt)], model=self.fast_model, temperature=0.2, max_tokens=160
            )
            text = (result.content or "").strip()
            if text:
                self._summary = text[:800]
        except Exception:  # noqa: BLE001 - summary is best-effort
            pass


async def _auto_approve(_name: str) -> bool:
    return True
