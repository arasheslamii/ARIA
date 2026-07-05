"""Setup wizard (Textual) — first-run setup AND a reconfigure menu.

Two modes, decided from state in :meth:`WizardApp.__init__`:

* **First run** (``not config.setup_complete``): start with a Provider choice when
  a local Ollama is detected — Local (private/free/offline) vs Groq (cloud) — then
  mic → voice → home. Local applies smart, adaptive model picks and skips the key
  step; Groq keeps the existing key step. No Ollama detected → straight to the key
  step exactly as before.
* **Reconfigure** (``setup_complete``): open a MENU (change voice / AI provider /
  home city / mic test / connect Google / exit). Each jumps to one step and saves.
  The Groq key prompt is shown ONLY when the user explicitly changes the provider
  to Groq — changing the voice never asks for a key. That's the core fix.

No secrets ever touch the TOML. Voice download, wake-word prefetch and audition
are preserved and stay best-effort (warnings, not hard failures).
"""

from __future__ import annotations

import asyncio
import importlib.util

from textual.app import App, ComposeResult
from textual.containers import Center, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from aria import APP_NAME, APP_TAGLINE
from aria.config.keyring import SecretStore
from aria.config.loader import load_config, save_config
from aria.config.schema import LLMConfig
from aria.llm.ollama import detect_ollama, pick_models, rank_models
from aria.tui.voices import (
    SAMPLE_TEXT,
    download_voice,
    installed,
    is_kokoro,
    voice_catalog,
)

# Reconfigure menu: (action id, label). Never includes a bare "enter Groq key".
MENU_ACTIONS = [
    ("act_voice", "Change voice"),
    ("act_provider", "Change AI provider (Groq / Local)"),
    ("act_activation", "Activation (wake word / hold-a-key)"),
    ("act_home", "Set / clear home city"),
    ("act_commerce", "Delivery profile (food ordering)"),
    ("act_mic", "Microphone test"),
    ("act_google", "Connect Google (Calendar + Gmail)"),
    ("act_exit", "Exit"),
]

# Which step each menu action jumps to. NOTE: none of these is the "key" step —
# the Groq key prompt is only reachable by choosing Groq on the provider step.
MENU_NEXT = {
    "act_voice": "voice",
    "act_provider": "provider",
    "act_activation": "activation",
    "act_home": "home",
    "act_commerce": "commerce",
    "act_mic": "mictest",
    "act_google": "google",
}


# Plain-language prompt shown in the delivery-profile step when the browser engine
# isn't installed yet. A non-technical user never sees pip/playwright.
COMMERCE_INSTALL_PROMPT = (
    "Food ordering needs a one-time browser setup (~150 MB download, asks for your "
    "password once)."
)


def commerce_install_prompt(ready: bool) -> str | None:
    """The install prompt to show in the delivery-profile step — or None when the
    engine is already installed (so we never nag a set-up user)."""
    return None if ready else COMMERCE_INSTALL_PROMPT


async def _validate_groq_key(key: str) -> bool:
    try:
        from aria.llm.base import user
        from aria.llm.groq_provider import GroqProvider

        provider = GroqProvider(key, timeout=10)
        await provider.chat([user("ping")], model="llama-3.1-8b-instant", max_tokens=1)
        await provider.aclose()
        return True
    except Exception:
        return False


def _mic_ok() -> tuple[bool, str]:
    try:
        import sounddevice as sd

        devices = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
        if not devices:
            return False, "No input devices found."
        return True, f"Default mic: {sd.query_devices(kind='input')['name']}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Audio unavailable: {exc}"


async def _download_wakeword(model: str) -> None:
    """Fetch the openWakeWord model + shared feature models into its cache.
    Custom model PATHS (a trained "hey topol" .onnx) are local files — skip."""
    if model.endswith((".onnx", ".tflite")) or "/" in model:
        return
    from openwakeword.utils import download_models

    await asyncio.to_thread(download_models, [model])


def _local_stt_provider() -> tuple[str, str | None]:
    """Pick the speech-to-text backend for local mode. Prefer faster-whisper so the
    "local = no Groq key" promise actually holds; if it isn't installed, fall back
    to Groq Whisper for STT ONLY and say so plainly."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster_whisper", None
    return (
        "groq",
        "Local speech-to-text needs the faster-whisper package — without it Aria "
        "still needs a Groq key for transcription only (everything else stays local).",
    )


def apply_local_config(config, picks: dict) -> str | None:
    """Point the config at local Ollama with the smart picks. Returns an STT warning
    string (or None). Does NOT touch any secret — local mode needs no API key."""
    config.llm.provider = "ollama"
    config.llm.base_url = "http://localhost:11434/v1"
    config.llm.reasoning_model = picks["reasoning_model"]
    config.llm.fast_model = picks["fast_model"]
    config.llm.synthesis_model = picks["synthesis_model"]
    stt_provider, warning = _local_stt_provider()
    config.stt.provider = stt_provider
    return warning


def apply_groq_config(config) -> None:
    """Restore the LLM fields to their Groq (schema-default) values and set STT back
    to Groq. Used when switching Local → Groq so stale Ollama model names (and the
    localhost base_url) can't linger and cause a 404 on the first cloud call. Reads
    the defaults from the schema rather than hardcoding them."""
    fields = LLMConfig.model_fields
    config.llm.provider = "groq"
    config.llm.reasoning_model = fields["reasoning_model"].default
    config.llm.fast_model = fields["fast_model"].default
    config.llm.synthesis_model = fields["synthesis_model"].default
    config.llm.base_url = fields["base_url"].default
    config.stt.provider = "groq"


def describe_picks(picks: dict) -> str:
    """One honest line about what local mode chose and whether it can act."""
    name = picks.get("reasoning_model")
    if not picks.get("tool_capable"):
        return (
            f"[yellow]Only small local models found ({name}). Aria will run, but it "
            "may NOT reliably perform actions (timers, email, calendar, system "
            "control). Groq is recommended for full capability.[/yellow]"
        )
    params = picks.get("reasoning_params_b") or 0.0
    size = f"~{params:.0f}B " if params else ""
    line = (
        f"[green]Local AI: {name} {size}for reasoning (tool-capable). "
        "Private and offline.[/green]"
    )
    if picks.get("ram_warning"):
        line += f"\n[yellow]{picks['ram_warning']}[/yellow]"
    return line


class WizardApp(App):
    CSS = """
    Screen { align: center middle; }
    #card { width: 76; padding: 1 2; border: round $accent; }
    .status { margin-top: 1; }
    Button { margin-top: 1; }
    Input { margin-top: 1; }
    RadioSet { margin-top: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        self.secrets = SecretStore()
        self.ranked: list = []
        self.picks: dict | None = None
        if self.config.setup_complete:
            # Reconfigure: NEVER probe for the Groq key — open the menu.
            self.mode = "menu"
            self.step = "menu"
        else:
            self.mode = "setup"
            self.ranked = rank_models(detect_ollama())
            if self.ranked:
                self.picks = pick_models(self.ranked)
                self.step = "provider"
            else:
                self.step = "key"

    # --- shell + per-step rendering --------------------------------------
    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="card"):
                yield Static(f"✦ {APP_NAME} — {APP_TAGLINE}", classes="title")
                yield Static("", id="subtitle")
                yield Static("", id="status", classes="status")
                yield Button("Continue", variant="primary", id="next")

    def on_mount(self) -> None:
        self._render_step()

    def _card(self) -> Vertical:
        return self.query_one("#card", Vertical)

    def _status(self) -> Static:
        return self.query_one("#status", Static)

    def _mount(self, widget):  # noqa: ANN001
        """Mount a per-step widget (tagged so we can clear it on the next step)."""
        widget.add_class("dynamic")
        self._card().mount(widget, before=self._status())
        return widget

    def _clear_dynamic(self) -> None:
        for w in self.query(".dynamic"):
            w.remove()

    def _subtitle(self, text: str) -> None:
        self.query_one("#subtitle", Static).update(text)

    def _render_step(self) -> None:
        self._clear_dynamic()
        getattr(self, f"_render_{self.step}", self._render_noop)()

    def _render_noop(self) -> None:  # pragma: no cover - defensive
        pass

    def _render_menu(self) -> None:
        self._subtitle("Reconfigure Aria. Pick one, then Continue.")
        self._mount(
            RadioSet(*[RadioButton(label, id=aid) for aid, label in MENU_ACTIONS], id="menu")
        )
        self._status().update("")

    def _render_provider(self) -> None:
        self._subtitle("Choose your AI provider.")
        local_label = "Local (Ollama) — private, free, offline"
        if self.picks and self.picks.get("reasoning_model"):
            local_label += f"  ·  {self.picks['reasoning_model']}"
        self._mount(
            RadioSet(
                RadioButton(local_label, id="prov_local", value=True),
                RadioButton("Groq (cloud) — fastest, most capable", id="prov_groq"),
                id="provider",
            )
        )
        if self.picks:
            self._status().update(describe_picks(self.picks))

    def _render_key(self) -> None:
        self._subtitle("Paste your Groq API key (stored securely, never on disk).")
        self._mount(Label("Groq API key:", id="key_label"))
        self._mount(Input(password=True, placeholder="gsk_…", id="key_input"))
        self._status().update(
            "[dim]Tip: run `aria install-local` once to add a free offline brain — "
            "Aria switches to it automatically whenever the cloud hits its "
            "daily limit.[/dim]"
        )

    def _render_mic_and_voice(self) -> None:
        """Mic check + voice picker (the shared setup body for first-run + voice menu)."""
        ok, msg = _mic_ok()
        colour = "green" if ok else "yellow"
        self._status().update(
            f"[{colour}]Mic check: {msg}[/{colour}]\n\nPick a voice "
            "(Audition to hear it), then Continue. Kokoro voices sound the most "
            "natural (one-time ~340 MB download, shared by all of them)."
        )
        self._mount(
            RadioSet(
                *[
                    RadioButton(f"{v}  —  {desc}", id=v)
                    for v, desc in voice_catalog().items()
                ],
                id="voices",
            )
        )
        self._mount(Button("Audition", id="audition"))

    def _render_voice(self) -> None:
        self._subtitle("Choose a voice.")
        self._render_mic_and_voice()

    def _render_mictest(self) -> None:
        self._subtitle("Microphone test.")
        ok, msg = _mic_ok()
        colour = "green" if ok else "yellow"
        self._status().update(f"[{colour}]Mic check: {msg}[/{colour}]\n\nContinue to go back.")

    def _render_activation(self) -> None:
        self._subtitle("How should Aria start listening?")
        from aria.voice.hotkey import KEY_CHOICES, access_problem

        a = self.config.activation
        self._mount(
            RadioSet(
                RadioButton(
                    'Wake word — say "hey jarvis"',
                    id="mode_wake_word", value=a.mode == "wake_word",
                ),
                RadioButton(
                    "Hold a key — press and hold, talk, release",
                    id="mode_hotkey", value=a.mode == "hotkey",
                ),
                RadioButton("Both", id="mode_hybrid", value=a.mode == "hybrid"),
                id="act_mode",
            )
        )
        self._mount(
            RadioSet(
                *[
                    RadioButton(k, id=f"key_{k.replace(' ', '_')}", value=k == a.hotkey)
                    for k in KEY_CHOICES
                ],
                id="act_key",
            )
        )
        note = "A short chime confirms she's listening (either way). Continue to save."
        problem = access_problem()
        if problem:
            note = (
                f"[yellow]Hold-to-talk needs one fix first: {problem}.[/yellow]\n" + note
            )
        self._status().update(note)

    def _render_home(self) -> None:
        self._subtitle("Home city for weather (optional).")
        existing = self.config.home_location or ""
        self._mount(Input(value=existing, placeholder="Home city (optional)", id="home_input"))
        self._status().update("Leave blank to clear, then Continue.")

    def _render_google(self) -> None:
        self._subtitle("Connect Google.")
        self._status().update(
            "Calendar + Gmail need a one-time OAuth setup. From a terminal run:\n"
            "  [bold]aria connect google[/bold]\n"
            "It walks you through it and opens your browser. Continue to go back."
        )

    def _render_commerce(self) -> None:
        self._subtitle("Delivery profile (food ordering).")
        c = self.config.commerce
        self._mount(Input(value=c.delivery_address or "",
                          placeholder="Delivery address", id="cm_address"))
        self._mount(Input(value=c.dietary_prefs or "",
                          placeholder="Dietary prefs (e.g. vegetarian, no nuts)", id="cm_diet"))
        self._mount(Input(value=", ".join(c.favorite_vendors),
                          placeholder="Favourite vendors (comma-separated)", id="cm_vendors"))
        self._mount(Input(value=c.default_food_app or "",
                          placeholder="Preferred app (e.g. Uber Eats)", id="cm_app"))
        self._mount(Input(value=("" if c.max_order_value is None else str(c.max_order_value)),
                          placeholder="Max order value (number, optional)", id="cm_max"))
        self._mount(Input(password=True, placeholder="Gemini API key (free, optional)",
                          id="cm_key"))
        note = (
            "Used by [bold]order food[/bold] — she fills the cart and STOPS at payment "
            "(never pays). Address stays local & is redacted from logs. The Gemini key "
            "(free at aistudio.google.com, separate from Google sign-in) drives the "
            "browser. Leave blank to keep current. Continue to save."
        )
        from aria.agents.browser_setup import commerce_engine_ready

        prompt = commerce_install_prompt(commerce_engine_ready())
        if prompt:  # engine not installed yet — offer the one-time setup
            self._mount(Button("Install browser engine now", id="install_commerce"))
            note = f"[yellow]{prompt}[/yellow] Press “Install browser engine now”.\n\n" + note
        self._status().update(note)

    # --- button dispatch -------------------------------------------------
    async def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: ANN001
        if event.button.id == "audition":
            await self._audition()
            return
        if event.button.id == "install_commerce":
            await self._install_commerce()
            return
        handler = getattr(self, f"_on_{self.step}", None)
        if handler is not None:
            await handler()

    def _radio_value(self, rs_id: str, default: str | None = None) -> str | None:
        rs = self.query_one(f"#{rs_id}", RadioSet)
        pressed = rs.pressed_button
        return pressed.id if pressed else default

    # --- step handlers (Continue pressed) --------------------------------
    async def _on_menu(self) -> None:
        action = self._radio_value("menu")
        if action in (None, "act_exit"):
            self.exit()
            return
        nxt = MENU_NEXT[action]
        if nxt == "provider":  # re-detect so the menu reflects current models
            self.ranked = rank_models(detect_ollama())
            self.picks = pick_models(self.ranked) if self.ranked else None
        self.step = nxt
        self._render_step()

    async def _on_activation(self) -> None:
        mode = self._radio_value("act_mode", f"mode_{self.config.activation.mode}")
        key_id = self._radio_value("act_key")
        self.config.activation.mode = (mode or "mode_wake_word").removeprefix("mode_")
        if key_id:
            self.config.activation.hotkey = key_id.removeprefix("key_").replace("_", " ")
        save_config(self.config)
        summary = f"Activation: {self.config.activation.mode.replace('_', ' ')}"
        if self.config.activation.mode != "wake_word":
            summary += f" (key: {self.config.activation.hotkey})"
        await self._back_to_menu(summary + ". Restart Aria to apply.")

    async def _on_provider(self) -> None:
        choice = self._radio_value("provider", "prov_local")
        if choice == "prov_local" and self.picks:
            warning = apply_local_config(self.config, self.picks)
            note = describe_picks(self.picks)
            if warning:
                note += f"\n[yellow]{warning}[/yellow]"
            self._status().update(note)
            if self.mode == "menu":
                save_config(self.config)
                await self._back_to_menu("Switched to local (Ollama).")
            else:
                self._goto_mic_voice()
        else:  # Groq -> reset any stale local model names, then ask for the key
            apply_groq_config(self.config)
            self.step = "key"
            self._render_step()

    async def _on_key(self) -> None:
        status = self._status()
        key = self.query_one("#key_input", Input).value.strip()
        if not key:
            status.update("[red]Please paste your Groq API key.[/red]")
            return
        status.update("Validating…")
        if not await _validate_groq_key(key):
            status.update("[red]That key didn't work. Check it and try again.[/red]")
            return
        backend = self.secrets.set("groq_api_key", key)
        if backend == "none":
            status.update(
                "[red]Couldn't save your key to the keyring or a fallback file.[/red]\n"
                "Add this to your shell profile instead:\n"
                "  [bold]export GROQ_API_KEY=…[/bold]\nthen re-run [bold]aria[/bold]."
            )
            return
        if backend == "file":
            status.update(
                "[yellow]No OS keyring available — saved your key to an encrypted "
                "file (~/.local/share/aria/secrets.enc, 0600).[/yellow]"
            )
        self.config.llm.provider = "groq"
        self.config.stt.provider = "groq"
        if self.mode == "menu":
            save_config(self.config)
            await self._back_to_menu("Switched to Groq.")
        else:
            self._goto_mic_voice()

    def _goto_mic_voice(self) -> None:
        self.step = "voice"
        self._render_step()

    async def _on_voice(self) -> None:
        status = self._status()
        voice = self._selected_voice()
        self.config.tts.voice = voice
        # The engine follows the voice: Kokoro voices need the kokoro provider,
        # classic voices need piper. Set it here so the pick Just Works.
        self.config.tts.provider = "kokoro" if is_kokoro(voice) else "piper"
        if not installed(voice):
            status.update(f"Downloading voice {voice}…")
            try:
                await download_voice(voice)
            except Exception as exc:  # noqa: BLE001
                status.update(
                    f"[yellow]Couldn't download now ({exc}). Will retry at first run.[/yellow]"
                )
        if self.config.wakeword.enabled:
            status.update(f"Fetching wake word “{self.config.wakeword.model}”…")
            try:
                await _download_wakeword(self.config.wakeword.model)
            except Exception:  # best-effort; OpenWakeWord re-fetches on load
                pass
        if self.mode == "menu":
            save_config(self.config)
            await self._back_to_menu(f"Voice set to {voice}.")
        else:
            self.step = "home"
            self._render_step()

    async def _on_home(self) -> None:
        city = self.query_one("#home_input", Input).value.strip()
        self.config.home_location = city or None
        if self.mode == "menu":
            save_config(self.config)
            await self._back_to_menu(
                f"Home city set to {city}." if city else "Home city cleared."
            )
            return
        self.config.user_name = None
        self.config.setup_complete = True
        save_config(self.config)
        where = f" Weather defaults to {city}." if city else ""
        self._status().update(
            f"[green]✓ All set![/green]{where} Run [bold]aria[/bold] and say the wake "
            f"word “{self.config.wakeword.model.replace('_', ' ')}”.\n"
            "[dim]Optional: connect Calendar + Gmail with [bold]aria connect google[/bold].[/dim]"
        )
        self.query_one("#next", Button).label = "Finish"
        self.step = "done"

    async def _on_mictest(self) -> None:
        await self._back_to_menu("")

    async def _on_google(self) -> None:
        await self._back_to_menu("")

    async def _on_commerce(self) -> None:
        c = self.config.commerce
        c.delivery_address = self.query_one("#cm_address", Input).value.strip() or None
        c.dietary_prefs = self.query_one("#cm_diet", Input).value.strip() or None
        vendors = self.query_one("#cm_vendors", Input).value.strip()
        c.favorite_vendors = [v.strip() for v in vendors.split(",") if v.strip()]
        c.default_food_app = self.query_one("#cm_app", Input).value.strip() or None
        max_raw = self.query_one("#cm_max", Input).value.strip()
        try:
            c.max_order_value = float(max_raw) if max_raw else None
        except ValueError:
            c.max_order_value = None
        save_config(self.config)
        # The Gemini key is a SECRET — store it in the keyring, never the TOML.
        key = self.query_one("#cm_key", Input).value.strip()
        note = "Delivery profile saved."
        if key:
            backend = self.secrets.set("commerce_api_key", key, durable=True)
            note += " Gemini key stored." if backend != "none" else (
                " (Couldn't store the key — set ARIA_COMMERCE_API_KEY instead.)"
            )
        await self._back_to_menu(note)

    async def _on_done(self) -> None:
        self.exit()

    async def _back_to_menu(self, note: str) -> None:
        self.step = "menu"
        self._render_step()
        if note:
            self._status().update(f"[green]{note}[/green] Pick another, or Exit.")

    # --- one-time browser-engine install (from the delivery-profile step) ---
    async def _install_commerce(self) -> None:
        """Install the food-ordering browser engine, streaming live progress into the
        status area. The heavy work runs in a worker thread; on failure we show the
        one-line manual fallback. The user never sees raw pip/playwright."""
        from aria.agents.browser_setup import BrowserSetupError, install_commerce_engine

        status = self._status()
        log: list[str] = []
        status.update("Installing the browser engine… this can take a minute.")

        def progress(msg: str) -> None:
            log.append(msg)
            try:  # marshal the UI update back onto the app's event loop
                self.call_from_thread(status.update, "\n".join(log[-8:]))
            except Exception:  # noqa: BLE001 - progress is best-effort
                pass

        try:
            await asyncio.to_thread(install_commerce_engine, progress)
        except BrowserSetupError as exc:
            status.update(f"[yellow]{exc}[/yellow]")
            return
        except Exception as exc:  # noqa: BLE001 - never crash the wizard
            status.update(f"[yellow]Install failed: {exc}[/yellow]")
            return
        for w in self.query("#install_commerce"):
            w.remove()
        status.update("[green]✓ Browser engine installed — food ordering is ready.[/green]")

    # --- audition (unchanged behaviour) ----------------------------------
    async def _audition(self) -> None:
        status = self._status()
        voice = self._selected_voice()
        status.update(f"Loading {voice}…")
        try:
            if not installed(voice):
                status.update(
                    f"Downloading {voice}…"
                    + (" (~340 MB one-time, shared by all Kokoro voices)"
                       if is_kokoro(voice) else "")
                )
                await download_voice(voice)
            from aria.app import build_tts
            from aria.voice.audio import Speaker

            cfg = self.config.model_copy(deep=True)
            cfg.tts.voice = voice
            cfg.tts.provider = "kokoro" if is_kokoro(voice) else "piper"
            cfg.tts.model_path = None
            tts = build_tts(cfg)
            speaker = Speaker(tts.sample_rate)
            status.update(f"[cyan]🔊 Playing {voice}…[/cyan]")
            await speaker.play(tts.synthesize(SAMPLE_TEXT))
            status.update(f"[green]That was {voice}.[/green] Pick one, then Continue.")
        except Exception as exc:  # noqa: BLE001 - audition is best-effort
            status.update(f"[yellow]Couldn't audition ({exc}). Pick one and Continue.[/yellow]")

    def _selected_voice(self) -> str:
        rs = self.query_one("#voices", RadioSet)
        pressed = rs.pressed_button
        return pressed.id if pressed else "en_US-amy-medium"


async def run_wizard() -> int:
    await WizardApp().run_async()
    return 0
