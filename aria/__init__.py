"""Aria — a fast, agentic, voice-first AI assistant for Linux.

The product name is intentionally centralised here so it can be renamed
everywhere in one place.
"""

# The single source of truth for the product name. Rename here only.
APP_NAME = "Topol"
# The slug stays "aria": it names config dirs, the binary, the systemd unit and
# the state dir — renaming it would orphan every existing install's data.
APP_SLUG = "aria"
APP_TAGLINE = "your voice, actually heard"

__version__ = "0.9.2"

__all__ = ["APP_NAME", "APP_SLUG", "APP_TAGLINE", "__version__"]
