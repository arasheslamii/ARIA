"""`aria` CLI entry point.

  aria            -> run the voice loop (or launch setup wizard on first run)
  aria setup      -> first-run TUI wizard (API key, mic test, voice pick)
  aria chat       -> text REPL (same brain, no microphone)
  aria voice      -> force the voice loop
  aria daemon     -> headless background loop (used by the systemd user service)
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
    sub.add_parser("enable", help="start Aria now and on every login")
    sub.add_parser("disable", help="stop autostart and stop the service")
    sub.add_parser("start", help="start the background service")
    sub.add_parser("stop", help="stop the background service")
    sub.add_parser("status", help="show the background service status")
    sub.add_parser("logs", help="tail the daemon logs")
    return p


_CONTROL_COMMANDS = {"enable", "disable", "start", "stop", "status", "logs"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # Control commands don't need config or the runtime — just wrap systemctl.
    if args.command in _CONTROL_COMMANDS:
        from aria.core.service import run_control

        return run_control(args.command)

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
