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


async def sentence_chunks(deltas: AsyncIterator[str]) -> AsyncIterator[str]:
    buffer = ""
    async for delta in deltas:
        buffer += delta
        # Flush every complete sentence in the buffer.
        while True:
            match = _BOUNDARY.search(buffer)
            if not match:
                break
            cut = match.end()
            sentence = buffer[:cut].strip()
            buffer = buffer[cut:]
            if len(sentence) >= _MIN_CHARS:
                yield sentence
            else:
                # too short — keep it attached to the next sentence
                buffer = sentence + " " + buffer
                break
    tail = buffer.strip()
    if tail:
        yield tail
