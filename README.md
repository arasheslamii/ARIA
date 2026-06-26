# Aria

**A fast, agentic, voice-first AI assistant for Linux.** Talk to it; it talks
back — instantly and naturally. Terminal-native (no GUI), trivial to install,
and genuinely capable of doing things on your machine.

> "Siri, but actually good." The name *Aria* is a placeholder — it lives in one
> constant (`aria.APP_NAME`) and can be renamed everywhere at once.

---

## Install (the one-command path)

Download `aria_<version>_amd64.deb` and:

```bash
sudo apt install ./aria_0.1.1_amd64.deb   # pulls libportaudio2 + libsecret-1-0
aria setup                                # paste your Groq API key (stored securely)
aria                                       # say the wake word and talk
aria enable                                # run her in the background on every login
```

That's it — `aria enable` registers a **systemd user service**, so after the next
login she's listening and firing alarms with **no terminal open**. Manage her
with `aria status | logs | stop | disable`.

The `.deb` is fully self-contained: it bundles its own Python runtime, the whole
voice stack, a local Piper voice, and the on-device wake word — so the install
downloads nothing heavy and works offline. Only `libportaudio2` and
`libsecret-1-0` (and a system `python3 ≥ 3.11`) are pulled from apt; `sudo apt
install ./…deb` resolves these automatically (use `sudo dpkg -i` only if you then
run `sudo apt -f install` to fetch the two libs).

**Uninstall:** `sudo apt remove aria` (run `aria disable` first to stop the
service). Your config, memory, and key under `~/.config/aria` and
`~/.local/share/aria` are left untouched.

---

## Quick start (development)

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/). On Debian/Ubuntu
you'll also want some system packages for audio and desktop control:

```bash
sudo apt install -y libportaudio2 libsecret-1-0 \
    brightnessctl playerctl wl-clipboard grim   # optional, for system control

# create the env and install Aria (editable)
uv venv --python 3.11
uv pip install -e ".[dev]"
```

You'll need a **Groq API key** (free tier works): https://console.groq.com/keys

### First run

```bash
uv run aria          # launches the terminal setup wizard on first run
```

The wizard: validates your Groq key → stores it in your **OS keyring** (never on
disk) → tests your microphone → lets you pick a voice (downloads it if needed) →
saves `~/.config/aria/config.toml`. Then:

```bash
uv run aria          # voice mode: say the wake word, then talk. Ctrl-C to quit.
uv run aria chat     # text mode: same brain, no microphone (great for testing)
uv run aria voice    # force voice mode
```

Don't have a mic handy? `aria chat` exercises the full orchestrator/tool/memory
stack from the keyboard.

---

## Always-on background mode

Run Aria as a background service so she's just *there* — listening for the wake
word and firing your alarms — with no terminal open. She runs as a **systemd
user service**, so she shares your login's PipeWire audio.

```bash
aria enable     # start now AND on every login   (systemctl --user enable --now)
aria status     # is she running?
aria logs       # tail her logs (journald)
aria stop       # stop until next login
aria disable    # stop and don't auto-start anymore
```

`aria` (no args) still runs the interactive foreground loop for dev. Run
`aria setup` once first so her Groq key is stored — the daemon reads it from your
keyring, or from an encrypted-at-rest fallback file when the keyring isn't
available under `systemd --user`.

### Privacy reality

In background mode Aria is **always listening locally** for her wake word
(openWakeWord, on-device — nothing leaves your machine until the wake word
fires). Only after the wake word does audio go to Groq for transcription. There
is no telemetry. To pause her completely: `aria stop` (this session) or
`aria disable` (stop auto-starting). The wake word can be turned off in
`~/.config/aria/config.toml` (`[wakeword] enabled = false`).

---

## Demo script — phrases to try

**Fast path / general:**
- "Hey Jarvis… what time is it?"
- "What's 18 percent of 240?"
- "What's the square root of 2 to four decimals?"

**Web search (cited):**
- "Who won the F1 race last weekend?"
- "What's the latest on the James Webb telescope?"

**Timers & reminders:**
- "Set a 10 minute timer."
- "Remind me to take the pasta off in 8 minutes."

**System control:**
- "Turn the volume down to 30."
- "Pause the music." / "Next track."
- "Take a screenshot."
- "What's on my clipboard?"

**Memory (persists across sessions):**
- "Remember that my name is Sam and I prefer Celsius."
- (next session) "What's my name?"

**Barge-in (opt-in):** off by default so she always finishes her answer on a
laptop's built-in mic+speakers (no echo cancellation). Enable with
`[vad] barge_in = true` in the config if you use a headset — then talking over her
stops her.

---

## How it's built

See [ARCHITECTURE.md](ARCHITECTURE.md). The short version:

- **Streaming everywhere.** Wake → VAD-endpointed capture → Groq Whisper STT →
  8B intent router → (cheap path streams straight to TTS; expensive path runs
  the 70B with parallel tools) → Piper speaks sentence 1 while the rest writes.
- **Swappable providers.** `LLMProvider` / `STT` / `TTS` / `VAD` / `WakeWord`
  interfaces; Groq + Piper + Silero + openWakeWord today, others are a one-line
  swap in `aria/app.py`. Models/voices are config, not code.
- **Agentic.** An orchestrator plans and delegates to specialist sub-agents
  (research, system, comms, files, compute) and tools — native + MCP — through
  one registry. Every call is permission-classified, timeout/retry-wrapped, and
  audit-logged. Risky/outward actions need confirmation.
- **Private by default.** Local VAD/wake-word/TTS, optional local STT, secrets in
  the keyring, no telemetry.

## Build the `.deb`

```bash
aria/packaging/build_deb.sh               # version comes from aria.__version__
# -> aria_0.1.1_amd64.deb  (~144 MB; ~440 MB installed)
```

The build (on a host with network) bundles a venv under `/opt/aria/venv` with the
full voice stack, the Piper voice into `/opt/aria/models`, and the openWakeWord
models into the venv. `/usr/bin/aria` runs the bundled venv's Python directly (no
reliance on the user's system Python or PATH) and points the voice resolver at
the bundled models via `ARIA_MODELS_DIR`. The systemd **user** unit installs to
`/usr/lib/systemd/user/aria.service`; `postinst` downloads nothing and never
enables the service from root — the user runs `aria enable`. `prerm` best-effort
stops the service; removal leaves per-user config/state in `$HOME`.

## Tests

```bash
uv run pytest             # unit tests + a mockable end-to-end voice-loop smoke test
```

## Configuration

- Config: `~/.config/aria/config.toml` (TOML, no secrets).
- State: `~/.local/share/aria/` (SQLite memory, audit log, downloaded voices).
- Secrets: OS keyring via libsecret (env-var fallback `GROQ_API_KEY` for CI).
- **Voice:** several natural local Piper voices ship in the catalog; the setup
  wizard lets you **audition and pick** one (`[tts] voice = …`). All 100% local —
  no cloud TTS. Medium voices keep first-word latency under ~1.2s; `*-high` voices
  sound a touch more natural and are slightly slower.
- **Notifications:** Aria uses your desktop's **native notification system**
  (portable across distros via libnotify/`desktop-notifier`) — branded as "Aria"
  with its own icon. It never drives any one clock/reminder app.

## License

Apache-2.0.
