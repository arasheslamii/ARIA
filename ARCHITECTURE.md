# Aria — Architecture

> Aria is a fast, agentic, voice-first AI assistant for Linux. Terminal-native,
> trivial to install, natural to talk to, and genuinely capable of doing things.
> The product name lives in one constant (`aria.APP_NAME`) so it can be renamed
> everywhere at once.

## Design pillars

1. **Fast** — perceived latency is the product. Stream STT → LLM → TTS. Speak
   sentence 1 while the rest is still generating. An 8B router takes the cheapest
   path; only escalate to the 70B model when needed. Independent tool calls run
   in parallel. Connections are pre-warmed.
2. **Natural** — smooth, conversational TTS; sentence-by-sentence playback;
   barge-in (interrupt and it stops); accurate transcription; short,
   spoken-friendly replies by default.
3. **Dead-simple install** — one `.deb`, then `aria` launches a terminal wizard.
   No GUI, ever. No heavy local LLM. Models that must be local (Piper voice) are
   bundled; nothing huge is fetched in `postinst`.
4. **Agentic** — an orchestrator plans, calls tools, delegates to specialist
   sub-agents, self-corrects, and runs multi-step jobs, asking to confirm only
   risky/outward actions.

## Layered structure

```
voice/      audio I/O, VAD, wake word, STT, TTS, streaming pipeline (barge-in)
llm/        LLMProvider interface + Groq impl + 8B fast router
core/       orchestrator, sub-agent base, memory (SQLite), resilient executor
tools/      native tools (search, math, time, timers, system, files, memory)
agents/     specialist sub-agents, each callable by the orchestrator as a tool
mcp/        MCP client — external servers' tools register like native ones
safety/     permission classification (safe/confirm/blocked) + audit trail
config/     TOML schema + loader + keyring secrets
tui/        Textual first-run wizard + live transcript view
packaging/  .deb build, bundled Piper voice, systemd user unit
```

`aria/app.py` is the **composition root**: the only place that maps interfaces
(`LLMProvider`, `STT`, `TTS`, `VAD`, `WakeWord`) to concrete classes. Swapping a
provider is a one-line change there plus a config value.

## The turn lifecycle (latency path)

```
wake word (openWakeWord)            local, always-on, optional
  └─ LISTENING: VAD-endpointed capture (Silero, energy fallback)
       └─ STT (Groq whisper-large-v3-turbo; faster-whisper fallback)
            └─ 8B router classifies: chitchat | tool | agentic
                 ├─ chitchat → stream 8B straight to TTS (no tool round-trip)
                 └─ tool/agentic → 70B resolves tool calls (parallel) →
                       stream final answer
                          └─ sentencizer cuts the token stream into sentences
                               └─ Piper TTS speaks sentence 1 while 2..n generate
                                    └─ barge-in: VAD on the mic stops playback
```

Why this hits the budget: cheap turns never touch the big model; expensive turns
overlap generation with speech; tools fan out concurrently; the network is warm.

## Interfaces (the swappable seams)

- `LLMProvider` — `chat`, `stream`, tool-calling. Groq now; OpenAI-compatible,
  Anthropic, Ollama later (Groq is OpenAI-shaped, so those are small).
- `STT` — `transcribe`. Groq Whisper (cloud) default; faster-whisper local.
- `TTS` — `synthesize` (async frame stream). Piper local; cloud TTS later.
- `VAD` / `WakeWord` — Silero / openWakeWord, each with a graceful fallback.

Models, voices, and provider choices are **config**, never hardcoded
(`~/.config/aria/config.toml`). Secrets live in the **OS keyring**, never in
config or logs.

## Agentic core

- **Orchestrator** (`core/orchestrator.py`) owns the conversation, short-term
  context, and long-term memory recall; routes intent; dispatches tools and
  sub-agents; streams the reply.
- **Sub-agents** (`agents/`) are focused system prompt + tool subset, wrapped as
  a single `Tool` so the orchestrator can delegate and parallelise: Research,
  System-control, Comms (email/calendar), Files, Compute, Travel (scaffold).
- **Tools** (`tools/base.py`) are `name + JSON schema + async run()`. Native and
  MCP tools share one `ToolRegistry` — the model sees a uniform list.
- **Executor** (`core/executor.py`) wraps every call: classify → confirm gate →
  timeout → bounded retry → audit → graceful spoken fallback. One tool failure
  never crashes the loop.
- **Safety** (`safety/`) classifies each call safe/confirm/blocked. confirm =
  send/spend/book/delete/system-change → explicit yes required. All actions are
  appended to an audit log.
- **Memory** (`core/memory.py`) — SQLite: durable facts/preferences recalled
  across sessions + a rolling transcript. Short-term context lives in the
  orchestrator.

## Capability tiers

- **MVP (now):** conversation + barge-in, web search w/ citations, math/units,
  timers/alarms/reminders, system control (volume/brightness/media/screenshot/
  clipboard/lock/open app), long-term memory, general Q&A.
- **Tier 2 (interfaces + MCP seams in place):** email & calendar (MCP), weather +
  morning briefing, file ops, translation, notes/to-dos, music/Spotify.
- **Tier 3 (scaffolded):** travel booking (confirmation-gated), maps/news, smart
  home (Home Assistant via MCP), plugin/skill system, proactive follow-ups.

## Privacy & resilience

Local-first where practical (VAD, wake word, TTS, optional STT). No telemetry.
Secrets only in the keyring. Missing/locked mic, offline voice download, and a
dead MCP server all degrade gracefully instead of crashing.
