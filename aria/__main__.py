"""`aria` CLI entry point.

  aria            -> run the voice loop (or launch setup wizard on first run)
  aria setup      -> first-run TUI wizard (API key, mic test, voice pick)
  aria chat       -> text REPL (same brain, no microphone)
  aria voice      -> force the voice loop
  aria daemon     -> headless background loop (used by the systemd user service)
  aria install-commerce -> install the food-ordering browser engine (one-time)
  aria install-local    -> size-check the machine, install Ollama + the right Qwen
  aria enable     -> start Aria now and on every login (systemctl --user)
  aria disable    -> stop autostart
  aria start/stop -> control the background service
  aria status     -> show the service status
  aria logs       -> tail the daemon logs
  aria --version
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from aria import APP_NAME, __version__
from aria.config.loader import load_config


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aria", description=f"{APP_NAME} voice assistant")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("setup", help="run the first-run setup wizard")
    sub.add_parser("chat", help="text mode (no microphone)")
    sub.add_parser("voice", help="voice mode")
    sub.add_parser("daemon", help="run headless in the background (used by the service)")
    sub.add_parser(
        "install-commerce",
        aliases=["setup-browser"],
        help="install the food-ordering browser engine (one-time)",
    )
    local_p = sub.add_parser(
        "install-local",
        help="check this machine and set up the local offline brain (Ollama + Qwen)",
    )
    local_p.add_argument(
        "model", nargs="?", default=None,
        help="override the auto-recommended model tag (e.g. qwen2.5:7b)",
    )
    sub.add_parser("enable", help="start Aria now and on every login")
    sub.add_parser("disable", help="stop autostart and stop the service")
    sub.add_parser("start", help="start the background service")
    sub.add_parser("stop", help="stop the background service")
    sub.add_parser("status", help="show the background service status")
    sub.add_parser("logs", help="tail the daemon logs")
    connect_p = sub.add_parser("connect", help="connect an integration (e.g. google)")
    connect_p.add_argument("service", choices=["google"])
    disconnect_p = sub.add_parser("disconnect", help="disconnect an integration")
    disconnect_p.add_argument("service", choices=["google"])
    return p


_CONTROL_COMMANDS = {"enable", "disable", "start", "stop", "status", "logs"}


def _install_local(model_override: str | None) -> int:
    """Probe the machine, recommend a right-sized local model, and (on a yes)
    install Ollama + pull it. The download happens here with consent and progress
    — never silently during the .deb install."""
    import subprocess
    from pathlib import Path

    from aria.config.hardware import probe_machine, recommend_local_model
    from aria.llm.ollama import detect_ollama

    profile = probe_machine()
    print(f"Your machine: {profile.describe()}")
    model, note = (model_override, f"Using your choice: {model_override}.") \
        if model_override else recommend_local_model(profile)
    print(note)
    if model is None:
        return 1

    if any(m.name == model for m in detect_ollama()):
        print(f"{model} is already installed and running — you're set. Aria uses it "
              "automatically whenever the cloud is rate-limited or offline.")
        return 0

    try:
        answer = input(f"Install Ollama (if needed) and download {model}? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    if answer.strip().lower() not in ("y", "yes"):
        print("Okay, nothing installed.")
        return 1

    script = Path(__file__).resolve().parents[1] / "scripts" / "install_local.sh"
    if not script.exists():  # installed from the .deb
        script = Path("/opt/aria/scripts/install_local.sh")
    if not script.exists():
        print("Installer script not found — reinstall the aria package.", file=sys.stderr)
        return 1
    result = subprocess.run(["bash", str(script), model], check=False)
    if result.returncode != 0:
        print("The local install didn't finish — see the output above.", file=sys.stderr)
        return result.returncode
    print(
        "\nAll set. Aria now switches to this model automatically whenever the "
        "cloud brain is rate-limited or offline.\n"
        "To make it the MAIN brain (fully offline), run `aria setup` -> "
        "Change AI provider -> Local."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Control commands don't need config or the runtime — just wrap systemctl.
    if args.command in _CONTROL_COMMANDS:
        from aria.core.service import run_control

        return run_control(args.command)

    if args.command == "connect":  # only "google" for now
        from aria.integrations.google_connect import connect_cli

        return connect_cli()
    if args.command == "disconnect":
        from aria.integrations.google_connect import disconnect_cli

        return disconnect_cli()

    if args.command in ("install-commerce", "setup-browser"):
        from aria.agents.browser_setup import (
            BrowserSetupError,
            commerce_engine_ready,
            install_commerce_engine,
        )

        if commerce_engine_ready():
            print("The food-ordering browser engine is already installed. You're set.")
            return 0
        try:
            install_commerce_engine(progress=print)
        except BrowserSetupError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            "Done. If Aria is running in the background, restart her with "
            "`systemctl --user restart aria`."
        )
        return 0

    if args.command == "install-local":
        return _install_local(args.model)

    config = load_config()

    if args.command == "setup" or (args.command is None and not config.setup_complete):
        from aria.tui.wizard import run_wizard

        return asyncio.run(run_wizard())

    if args.command == "chat":
        from aria.core.runtime import run_text

        asyncio.run(run_text(config))
        return 0

    if args.command == "daemon":
        from aria.core.daemon import run_daemon

        return asyncio.run(run_daemon(config))

    # default + "voice"
    from aria.core.runtime import run_voice

    try:
        asyncio.run(run_voice(config))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
