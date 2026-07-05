# Topol

**A fast, agentic, voice-first AI assistant for Linux.** Talk to it; it talks
back — instantly and naturally. Terminal-native (no GUI), trivial to install,
and genuinely capable of doing things on your machine.

> "Siri, but actually good." The product name is **Topol** (`aria.APP_NAME`);
> the binary, package, and config paths keep the original `aria` slug so
> existing installs and data stay intact.

---

## Install (the one-command path)

Download `aria_<version>_amd64.deb` and:

```bash
sudo apt install ./aria_0.9.2_amd64.deb   # pulls libportaudio2 + libsecret-1-0
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

# create the env and install Topol (editable)
uv venv --python 3.11
uv pip install -e ".[dev]"
```

You'll need a **Groq API key** (free tier works): https://console.groq.com/keys

### First run

```bash
uv run aria          # launches the terminal setup wizard on first run
```

The wizard: picks your **AI provider** → (Groq) validates the key and stores it in
your **OS keyring** (never on disk) → tests your microphone → lets you pick a voice
(downloads it if needed) → saves `~/.config/aria/config.toml`. Then:

```bash
uv run aria          # voice mode: say the wake word, then talk. Ctrl-C to quit.
uv run aria chat     # text mode: same brain, no microphone (great for testing)
uv run aria voice    # force voice mode
```

Don't have a mic handy? `aria chat` exercises the full orchestrator/tool/memory
stack from the keyboard.

### Run it fully local (Ollama) — no key, private, offline

If [Ollama](https://ollama.com) is running, the wizard offers **Local (Ollama)**
alongside Groq. Pick it and Topol runs entirely on your machine — no API key, no
cloud. The model choice is **adaptive to whatever you've pulled**: it inspects
`ollama list`, ranks by real parameter count with a tool-calling boost, and picks
the strongest *tool-capable* model for reasoning (a 32B/70B beats an 8B
automatically) plus the smallest tool-capable one for snappy routing. It tells you
exactly what it chose, and warns plainly if you only have tiny models that can't
reliably do actions (timers, email, calendar) — in which case Groq is recommended.
Speech-to-text also goes local (faster-whisper, bundled in the `.deb`), so the
"local = no Groq key" promise actually holds. TTS (Piper) was already local.

### Automatic local fallback — the daily limit never stops her

If you use **Groq as the main brain** and a local Ollama exists, Topol switches to
it **automatically and mid-conversation** whenever Groq is rate-limited (the free
tier's daily token cap) or the network is down — slightly less brilliant for a
while, but she keeps working instead of apologizing. Zero configuration: it's on
by default (`[llm] local_fallback = true`), costs nothing when Ollama isn't
installed, and picks the local models adaptively (biggest tool-capable model for
reasoning, smallest snappy one for routing).

Don't have a local model yet? One command sizes it to your machine:

```bash
aria install-local    # probes RAM/CPU/GPU/disk → recommends the right Qwen →
                      # installs Ollama + downloads it on your yes
```

### Activation: wake word, hold-a-key, or both

Three ways to start talking, picked in `aria setup` → **Activation**:

- **Wake word** (default): say *"hey jarvis"*.
- **Hold a key**: press and hold a key (default **right ctrl**), talk, release —
  walkie-talkie style. The release ends the capture instantly, so it's the
  fastest and most reliable method (no waiting for silence detection). Needs
  read access to the keyboard: `sudo usermod -aG input $USER`, then log out and
  back in. Works on X11, Wayland, and in the background daemon.
- **Hybrid**: both at once. Bonus: while she's speaking, pressing the key cuts
  her off and listens — guaranteed barge-in.

Either way, a short rising **chime** confirms she's actually listening the moment
she activates (disable with `[activation] chime = false`). Siri-style by default:
she answers and goes back to sleep. Only when SHE asks you something (a
confirmation, a question) does the mic re-open — for **4 seconds**. If you want
the always-flowing mode back (mic re-opens after every answer), set
`[conversation] enabled = true`.

### Custom wake word — "hey topol"

The stock wake models are openWakeWord's pretrained set (`hey_jarvis` is the
default); "hey topol" needs a **one-time custom training run** (~30-60 min on a
free Google Colab GPU — it synthesizes thousands of spoken samples and trains a
small classifier; a laptop CPU is not the right tool for it):

1. Open openWakeWord's **automatic model training notebook** in Colab:
   <https://github.com/dscripka/openWakeWord> → `notebooks/automatic_model_training.ipynb`
   (there's an "Open in Colab" badge in their docs).
2. Set the target phrase to `hey topol`, run all cells, and download the
   resulting `hey_topol.onnx`.
3. Drop it in `~/.local/share/aria/models/` and point the config at it:

   ```toml
   [wakeword]
   model = "/home/YOU/.local/share/aria/models/hey_topol.onnx"
   ```
4. `systemctl --user restart aria` — she now wakes to "hey topol". If the file
   is missing she says so at startup and falls back cleanly.

Custom `.onnx`/`.tflite` paths are fully supported by the config; until you
train one, `hey_jarvis` (or hold-to-talk) keeps working.

### Reconfigure later (without re-entering your key)

Run `aria setup` again any time after the first run and you get a **menu** —
change voice, switch AI provider (Groq ↔ Local), pick the activation method,
set/clear your home city, test the mic, or connect Google. Changing the voice
never asks for your Groq key again.

---

## Always-on background mode

Run Topol as a background service so she's just *there* — listening for the wake
word and firing your alarms — with no terminal open. She runs as a **systemd
user service**, so she shares your login's PipeWire audio.

```bash
aria enable     # start now AND on every BOOT   (systemctl --user enable --now + linger)
aria status     # is she running?
aria logs       # tail her logs (journald)
aria stop       # stop until next login
aria disable    # stop and don't auto-start anymore
```

`aria` (no args) still runs the interactive foreground loop for dev. Run
`aria setup` once first so her Groq key is stored — the daemon reads it from your
keyring, or from an encrypted-at-rest fallback file when the keyring isn't
available under `systemd --user`.

`aria enable` also turns on **linger** (`loginctl enable-linger`) so her user
service starts at **boot** without a graphical login. And because the network
often isn't up the instant WiFi associates after a reboot, the daemon treats a
connection/DNS error as **retryable** — it backs off and stays up rather than
crash-looping, so she's never dead for the first minute after a restart.

### Privacy reality

In background mode Topol is **always listening locally** for her wake word
(openWakeWord, on-device — nothing leaves your machine until the wake word
fires). Only after the wake word does audio go to Groq for transcription. There
is no telemetry. To pause her completely: `aria stop` (this session) or
`aria disable` (stop auto-starting). The wake word can be turned off in
`~/.config/aria/config.toml` (`[wakeword] enabled = false`).

---

## Connect Google (Calendar + Gmail)

Weather needs nothing. Calendar and email need a **one-time** Google setup (~5 min)
because these scopes require *your own* OAuth client:

```bash
aria connect google      # prints the setup steps, then opens your browser
aria disconnect google   # revoke + remove the token
```

`aria connect google` walks you through it:
1. Create a project at <https://console.cloud.google.com/>.
2. **Enable** the *Google Calendar API* and the *Gmail API*.
3. OAuth consent screen → External → add yourself as a **Test user**.
4. Credentials → Create OAuth client ID → **Desktop app** → copy the Client ID + secret.
5. Paste them when prompted; a browser opens to sign in (Calendar + Gmail in one consent).

The token is stored locally in your encrypted secret store (keyring or 0600 file),
refreshed automatically, and only ever sent to Google. Then just ask: *"what's on my
schedule today?"*, *"add a dentist appointment Friday at 3pm"* (she reads it back and
only adds it on your **yes**). When she asks you to confirm, she **holds the floor** —
the mic re-opens right after she speaks, so you just say *"yes"* with no wake word
(she waits a few seconds, then falls back to wake-word mode if you don't answer).

---

## Order food — "buy me a pizza" (agentic, stops at payment)

Topol can drive a **real browser** to find a good shop, build your cart, fill in the
delivery address, and go to the checkout — then **STOP**. She **never pays**: the
live browser window is handed to you at the payment page so you pay yourself. No
card numbers are ever handled or stored (she relies on the site's/browser's saved
payment). Food delivery only, for now.

It's a confirm-gated **commerce** action — she reads back what she'll order and only
starts on your **yes**, then narrates progress so there's no dead air during the
(slow) browse.

**One-time setup:**

```bash
# 1) Install the browser engine (kept out of the .deb to stay lean). This is a
#    single command — no manual pip/playwright. It downloads ~150 MB and asks for
#    your password once (the runtime venv is root-owned), then installs Chromium:
aria install-commerce
#    You can also just open "Delivery profile" in `aria setup` and say yes when it
#    offers to install it — same thing, no terminal needed.

# 2) Get a FREE Gemini key — this is SEPARATE from the Google sign-in above:
#    https://aistudio.google.com/apikey   (drives the browser; not your OAuth token)

# 3) In the reconfigure menu, open "Delivery profile (food ordering)" and set your
#    address, dietary prefs, favourite vendors, preferred app, an optional spend
#    ceiling, and paste the Gemini key:
aria setup
```

If anything goes wrong, the exact manual fallback is `sudo bash
/opt/aria/scripts/install_commerce.sh` (shipped in the package).

The Gemini engine is free and **doesn't touch your Groq budget** (swappable to Groq
or local Ollama via `[commerce] engine`). Your address is **local-only** and
**redacted from the audit log**. Log into your delivery sites once in her browser
(the profile persists), then just say *"order me a large pepperoni"* — she'll get
you to the payment page and say *"Opening the payment page; pay whenever you're
ready."* If she hits a captcha/login or can't find a fit, she stops and tells you
exactly where. She needs your graphical session (X11/Wayland) for the browser; with
no display she says *"I need your screen for that."*

---

## Real-world errands — flights, hotels, shopping (you do the last click)

Beyond delivery, Topol runs errands end to end **except the final approval and
payment, which always happen in *your* own default browser** (whatever `xdg-open`
launches — Firefox, Chrome, anything):

- *"Book me a flight from Edinburgh to London on July 10"* → live Google Flights
  results for that route and date open on your screen; you pick the flight and pay.
- *"Book a hotel in London for the 10th to the 12th"* → live Booking.com results
  open; you choose the room and pay.
- **Give her criteria and she picks FOR you**: *"book a budget hotel in Paris,
  July 10 to 15, under 100 pounds a night"* → `reserve_hotel` drives a real
  browser, filters by your budget, compares ratings, picks the best fit, selects
  a room, and stops at the **final reservation step** with everything set up —
  you review and pay. Same for flights (*"…the best flight under £80, nonstop"*
  → `reserve_flight` stops at the airline's booking page). These are
  confirm-gated, take a couple of minutes (she narrates), and use the same
  browser engine as food ordering (`aria install-commerce`, one-time).
- *"Buy me a 4-port USB-C hub"* → live shopping results open; you check out.
- *"Plan my trip to Rome next weekend"* → the **errands agent** researches, then
  opens each piece (flight + hotel) for you to finish.

No setup, no keys, no browser engine needed — these use smart deep links, so
they're instant. Nothing is ever booked, bought, or paid by Topol; she opens pages,
you do the very last thing.

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
# -> aria_0.9.2_amd64.deb  (~144 MB; ~440 MB installed)
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
- **Voice:** the wizard lets you **audition and pick** a voice. **Kokoro** voices
  (`af_heart`, `am_michael`, …) have near-human prosody — the "doesn't sound like
  a robot" upgrade — and share a one-time ~340 MB download; classic Piper voices
  stay available as the lean/bundled option. All 100% local — no cloud TTS. If a
  Kokoro voice is configured but its files are missing, Topol falls back to the
  bundled Piper voice instead of dying voiceless.
- **Conversation mode** (`[conversation]`, on by default): after Topol answers,
  the mic re-opens for ~6s so you can just keep talking — no wake word between
  turns. A fast-model relevance gate drops background speech (TV, side chatter)
  so she never butts in uninvited; set `enabled = false` for strict wake-word
  turn-taking. Confirmations understand amendments, too: answering "yes but make
  it 20 minutes" to a read-back re-plans and re-confirms instead of demanding a
  yes/no.
- **Conversation memory:** long chats are summarized on the fly (older turns fold
  into a running summary the model keeps seeing), and at startup she recalls what
  your previous session was about — so "what were we talking about yesterday?"
  actually works.
- **Notifications:** Topol uses your desktop's **native notification system**
  (portable across distros via libnotify/`desktop-notifier`) — branded as "Topol"
  with its own icon. It never drives any one clock/reminder app.
- **Free-tier resilience:** if Groq rate-limits you (its free daily cap), Topol says
  *"I've hit my usage limit — let's try again in a few minutes"* and keeps running
  (no crash). For a durable answer, configure a **free fallback provider** — when
  Groq is capped she switches to it automatically:
  ```toml
  [llm]
  fallback_provider = "cerebras"   # or "gemini" (both have free tiers)
  ```
  and provide its key via `ARIA_FALLBACK_API_KEY` (or the keyring). Primary stays
  Groq; the fallback only kicks in on a 429/outage.

## License

Apache-2.0.
