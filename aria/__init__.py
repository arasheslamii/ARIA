"""Aria — a fast, agentic, voice-first AI assistant for Linux.

The product name is intentionally centralised here so it can be renamed
everywhere in one place.
"""

# The single source of truth for the product name. Rename here only.
APP_NAME = "Aria"
APP_SLUG = "aria"  # used for config dirs, binaries, dbus ids, etc.
APP_TAGLINE = "your voice, actually heard"

__version__ = "0.8.0"

__all__ = ["APP_NAME", "APP_SLUG", "APP_TAGLINE", "__version__"]
