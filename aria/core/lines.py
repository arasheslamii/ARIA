"""Varied spoken lines for the moments Aria can't ask a model to write them.

These cover the paths where an LLM call is unavailable (rate-limited, offline) or
too slow/risky (the filler must be spoken *before* the slow round-trip; the
confirmation read-back must carry the exact action verbatim). Everything else
Aria says comes from the model. Picking randomly from a pool — never the same
line twice in a row — is what keeps her from sounding like a script.
"""

from __future__ import annotations

import random

# Spoken when a slow (network / multi-step) tool round-trip begins. Trailing
# space makes the sentencizer flush it immediately.
FILLERS = (
    "Let me look into that. ",
    "One sec — checking now. ",
    "Alright, let me find out. ",
    "Give me a moment to dig into that. ",
    "Hmm, let me check. ",
    "On it — one moment. ",
    "Let me pull that up. ",
)

# The frame around a confirm-gated action. {action} is the EXACT read-back (never
# paraphrased by a model — it may name money or a recipient). Every frame ends by
# clearly asking, so a yes/no answer stays natural.
CONFIRM_FRAMES = (
    "Just checking — you want me to {action}. Should I go ahead?",
    "So I'll {action} — shall I go ahead?",
    "You'd like me to {action}, right? Say the word and I'll go ahead.",
    "Ready to {action} — should I go ahead?",
)

# Re-ask after an unclear confirmation answer. All variants keep "yes or no" so
# the expected reply is unmistakable even mid-kitchen-noise.
CONFIRM_REASKS = (
    "Sorry, I didn't catch that — yes or no, should I go ahead?",
    "I missed that — yes or no?",
    "Just to be sure: yes or no?",
)

# Acknowledging a declined action.
DECLINED = (
    "Okay, I won't.",
    "No problem — consider it dropped.",
    "Alright, leaving it alone.",
    "Got it, cancelled.",
)

# Giving up on an unanswered/garbled confirmation instead of re-asking forever.
DROPPED = (
    "I'll leave that for now — just ask me again if you still want it.",
    "Let's drop that one for the moment. Say the word if you want it back.",
    "Okay, parking that. Ask again whenever.",
)

# A turn that would otherwise end silent.
NO_RESPONSE = (
    "Sorry, I'm not sure how to help with that one.",
    "Hmm, I don't think I can do that one yet.",
    "That one's beyond me for now, sorry.",
)

# Transient provider failures — warm, human, never a stack trace.
RATE_LIMITED = (
    "I've hit my usage limit for the moment — let's try again in a few minutes.",
    "I'm a bit over my limit right now — give me a few minutes and ask again.",
)
OFFLINE = (
    "I'm having trouble reaching the network right now — let's try again in a moment.",
    "Looks like the internet's not cooperating — try me again in a bit.",
)

_last: dict[int, str] = {}


def pick(pool: tuple[str, ...]) -> str:
    """A random line from ``pool``, avoiding the one used last time."""
    if len(pool) == 1:
        return pool[0]
    prev = _last.get(id(pool))
    choice = random.choice([p for p in pool if p != prev] or list(pool))
    _last[id(pool)] = choice
    return choice
