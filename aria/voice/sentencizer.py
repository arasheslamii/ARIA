"""Turn a stream of LLM token deltas into speakable sentence chunks.

This is what lets Aria start *talking* before the full answer is generated: as
soon as a sentence boundary is seen, that sentence is flushed to TTS while the
model keeps writing the rest.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

# End-of-sentence punctuation followed by space/end. Keeps decimals like 3.14
# and abbreviations mostly intact for natural prosody.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_MIN_CHARS = 12  # don't flush tiny fragments; batch them for smoother prosody

# After the first chunk is out (and playing), later sentences are COALESCED into
# larger chunks up to this size. TTS engines phrase a multi-sentence chunk with far
# more natural prosody than one sentence at a time, and by then playback of chunk 1
# has bought the time — so this trades zero perceived latency for smoothness.
_COALESCE_CHARS = 220

# The FIRST flush sets time-to-first-audio, and Kokoro synthesizes slower than
# real time on CPU — a long opening sentence means seconds of dead air. If the
# first sentence is still unfinished past this size, cut it at a clause boundary
# (comma/semicolon/colon) and start speaking.
_FIRST_CLAUSE_CHARS = 60
_CLAUSE = re.compile(r"(?<=[,;:])\s+")


async def sentence_chunks(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    pending = ""  # completed sentences awaiting coalesced flush (post-first chunk)
    first = True
    async for delta in deltas:
        buffer += delta
        # Consume every complete sentence in the buffer.
        while True:
            match = _BOUNDARY.search(buffer)
            if not match:
                break
            cut = match.end()
            sentence = buffer[:cut].strip()
            buffer = buffer[cut:]
            if first:
                if len(sentence) >= _MIN_CHARS:
                    yield sentence  # speak the first sentence ASAP
                    first = False
                else:
                    # too short — keep it attached to the next sentence
                    buffer = sentence + " " + buffer
                    break
            else:
                pending = f"{pending} {sentence}".strip()
                if len(pending) >= _COALESCE_CHARS:
                    yield pending
                    pending = ""
        # Long opening sentence: don't sit silent waiting for its full stop —
        # flush the first clause so the voice starts, and coalesce the rest.
        if first and len(buffer) >= _FIRST_CLAUSE_CHARS:
            clause = _CLAUSE.search(buffer, _MIN_CHARS)
            if clause:
                yield buffer[: clause.end()].strip()
                buffer = buffer[clause.end() :]
                first = False
    tail = f"{pending} {buffer.strip()}".strip()
    if tail:
        yield tail
