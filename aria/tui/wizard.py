"""First-run setup wizard (Textual).

Steps: welcome → Groq API key (validated, stored in keyring) → microphone test
→ voice pick (download if needed) → save config. No secrets ever touch the TOML.

The wizard is intentionally resilient: missing mic or offline voice download
degrade to warnings, not hard failures, so the user can still finish setup.
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Center, Vertical
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, Static

from aria import APP_NAME, APP_TAGLINE
from aria.config.keyring import SecretStore
from aria.config.loader import load_config, save_config
from aria.tui.voices import SAMPLE_TEXT, VOICES, download_voice, installed


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
    """Fetch the openWakeWord model + shared feature models into its cache."""

    from openwakeword.utils import download_models

    await asyncio.to_thread(download_models, [model])


class WizardApp(App):
    CSS = """
    Screen { align: center middle; }
    #card { width: 72; padding: 1 2; border: round $accent; }
    .title { text-style: bold; color: $accent; }
    .status { margin-top: 1; }
    Button { margin-top: 1; }
    Input { margin-top: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        self.secrets = SecretStore()
        self.step = "key" if not self.secrets.has("groq_api_key") else "mic"

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="card"):
                yield Static(f"✦ {APP_NAME} — {APP_TAGLINE}", classes="title")
                yield Static("Let's get you set up. Takes about a minute.")
                yield Label("Groq API key:", id="key_label")
                yield Input(password=True, placeholder="gsk_…", id="key_input")
                yield Static("", id="status", classes="status")
                yield Button("Continue", variant="primary", id="next")

    async def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: ANN001
        if event.button.id == "audition":
            await self._audition()
            return
        if self.step == "key":
            await self._handle_key()
        elif self.step == "mic":
            self._handle_mic()
        elif self.step == "voice":
            await self._handle_voice()
        elif self.step == "done":
            self.exit()

    async def _audition(self) -> None:
        """Play a short sample of the highlighted voice so the user can hear it."""
        status = self.query_one("#status", Static)
        voice = self._selected_voice()
        status.update(f"Loading {voice}…")
        try:
            if not installed(voice):
                await download_voice(voice)
            from aria.app import resolve_piper_model
            from aria.voice.audio import Speaker
            from aria.voice.tts_piper import PiperTTS

            cfg = self.config.model_copy(deep=True)
            cfg.tts.voice = voice
            tts = PiperTTS(resolve_piper_model(cfg))
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

    async def _handle_key(self) -> None:
        status = self.query_one("#status", Static)
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
            # Don't claim success when nothing persisted — tell the user how to fix.
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
        self.query_one("#key_input", Input).display = False
        self.query_one("#key_label", Label).display = False
        self.step = "mic"
        self._handle_mic()

    def _handle_mic(self) -> None:
        status = self.query_one("#status", Static)
        ok, msg = _mic_ok()
        colour = "green" if ok else "yellow"
        status.update(
            f"[{colour}]Mic check: {msg}[/{colour}]\n\n"
            "Pick a voice (Audition to hear it), then Continue."
        )
        if not self.query(RadioSet):
            rs = RadioSet(
                *[RadioButton(f"{v}  —  {desc}", id=v) for v, desc in VOICES.items()],
                id="voices",
            )
            card = self.query_one("#card", Vertical)
            card.mount(rs, before=status)
            card.mount(Button("Audition", id="audition"), before=status)
        self.step = "voice"

    async def _handle_voice(self) -> None:
        status = self.query_one("#status", Static)
        voice = self._selected_voice()
        self.config.tts.voice = voice
        if not installed(voice):
            status.update(f"Downloading voice {voice}…")
            try:
                await download_voice(voice)
            except Exception as exc:  # noqa: BLE001
                status.update(
                    f"[yellow]Couldn't download now ({exc}). Will retry at first run.[/yellow]"
                )
        # Fetch the local wake-word model too, so the first `aria` run is instant.
        if self.config.wakeword.enabled:
            status.update(f"Fetching wake word “{self.config.wakeword.model}”…")
            try:
                await _download_wakeword(self.config.wakeword.model)
            except Exception:  # best-effort; OpenWakeWord re-fetches on load
                pass
        self.config.user_name = None
        self.config.setup_complete = True
        save_config(self.config)
        status.update(
            f"[green]✓ All set![/green] Run [bold]aria[/bold] and say the wake word "
            f"“{self.config.wakeword.model.replace('_', ' ')}”."
        )
        self.query_one("#next", Button).label = "Finish"
        self.step = "done"


async def run_wizard() -> int:
    await WizardApp().run_async()
    return 0
