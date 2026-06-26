"""Optional live chat/transcript view.

A lightweight Textual widget set the voice runtime can mount to show the rolling
transcript and current pipeline state. Kept minimal for the MVP; the voice loop
also works headless with plain Rich console output (see core.runtime).
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import RichLog, Static

from aria import APP_NAME


class TranscriptView(VerticalScroll):
    """A scrollable transcript pane."""

    def compose(self) -> ComposeResult:
        yield Static(f"{APP_NAME} — live transcript", classes="title")
        yield RichLog(id="log", wrap=True, markup=True)

    def add_user(self, text: str) -> None:
        self.query_one("#log", RichLog).write(f"[bold]You:[/bold] {text}")

    def add_assistant(self, text: str) -> None:
        self.query_one("#log", RichLog).write(f"[cyan]{APP_NAME}:[/cyan] {text}")

    def set_state(self, label: str) -> None:
        self.query_one("#log", RichLog).write(f"[dim]{label}[/dim]")
