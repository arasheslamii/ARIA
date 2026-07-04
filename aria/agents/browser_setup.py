"""One-command installer for the agentic-ordering browser engine.

The engine (browser-use + Playwright + Chromium) is heavy and optional, so it's
kept OUT of the .deb. Rather than make a non-technical user run pip and playwright
by hand, this module installs it into Aria's *runtime* venv on demand — from the
wizard ("Install now?") or `aria install-commerce`.

Hard facts it copes with (do not assume otherwise):
  * The .deb's runtime venv is ``/opt/aria/venv`` — root-owned and *without pip*.
    We bootstrap pip with ``ensurepip`` and, when the venv isn't writable by the
    current user, escalate each command through ``pkexec`` (a graphical polkit
    prompt — best for the TUI) or, failing that, ``sudo``. We never assume root.
  * The runtime venv is NOT necessarily ``sys.prefix``: the wizard may run from a
    different interpreter. We resolve ``/opt/aria/venv`` explicitly (the path the
    launcher/daemon use) and only fall back to ``sys.prefix`` for dev installs.

Every step mirrors ``scripts/install_commerce.sh`` (the verified reference) and is
idempotent — an already-satisfied step is skipped. On any failure we raise
:class:`BrowserSetupError` naming the exact manual fallback command.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aria.agents.web_agent import chromium_installed

# The bundled runtime venv the .deb launcher execs (`/usr/bin/aria` runs
# /opt/aria/venv/bin/python -m aria). The engine MUST land here so the daemon can
# import it — this is why we don't just trust sys.prefix.
_DEB_VENV = Path("/opt/aria/venv")

# Shipped in the .deb at this path; the documented manual fallback.
FALLBACK_SCRIPT = "/opt/aria/scripts/install_commerce.sh"
FALLBACK_COMMAND = f"sudo bash {FALLBACK_SCRIPT}"

_PACKAGES = ["browser-use>=0.1", "playwright>=1.40", "langchain-openai>=0.1"]


class BrowserSetupError(RuntimeError):
    """Installing the browser engine failed; the message carries the manual
    fallback command so the user always has an escape hatch."""


def _fallback_msg(reason: str) -> str:
    return (
        f"Couldn't set up the browser engine ({reason}). "
        f"Run this once in a terminal, then try again:\n    {FALLBACK_COMMAND}"
    )


def runtime_venv() -> Path:
    """The venv Aria's daemon actually runs from. Prefer the .deb's /opt/aria/venv
    (matching the launcher); fall back to the interpreter we're running under for
    editable/dev installs. Never blindly trust sys.prefix for the .deb case."""
    if (_DEB_VENV / "bin" / "python").exists():
        return _DEB_VENV
    return Path(sys.prefix)


def _venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def _pip_present() -> bool:
    return importlib.util.find_spec("pip") is not None


def _browser_use_importable() -> bool:
    return importlib.util.find_spec("browser_use") is not None


def commerce_engine_ready() -> bool:
    """True only when the browser engine is importable AND a Chromium is present —
    i.e. ``order_food`` can actually run. Reuses web_agent.chromium_installed()."""
    return _browser_use_importable() and chromium_installed()


def _escalation_prefix(venv: Path) -> list[str]:
    """Empty when the venv is writable by us; else ``pkexec`` (graphical polkit
    dialog — best for the TUI) if available, else ``sudo``. Never assumes root."""
    if os.access(venv, os.W_OK):
        return []
    if shutil.which("pkexec"):
        return ["pkexec"]
    if shutil.which("sudo"):
        return ["sudo"]
    return []  # neither available: attempt directly, then fall into the fallback


@dataclass
class _Step:
    desc: str
    argv: list[str]
    satisfied: bool = False


def plan_steps(venv: Path, *, prefix: list[str] | None = None) -> list[_Step]:
    """The idempotent command sequence, mirroring scripts/install_commerce.sh.
    ``prefix`` (pkexec/sudo) is applied to every command when the venv isn't
    writable. Steps whose result already exists are marked satisfied (skipped).

    Chromium download + system-lib install share the chromium check: a prior run
    installs both together, so if Chromium is present we skip both."""
    prefix = _escalation_prefix(venv) if prefix is None else prefix
    py = str(_venv_python(venv))
    have_pip = _pip_present()
    have_pkgs = _browser_use_importable()
    have_chromium = chromium_installed()
    return [
        _Step(
            "Bootstrapping pip in the runtime venv",
            [*prefix, py, "-m", "ensurepip", "--upgrade"],
            satisfied=have_pip,
        ),
        _Step(
            "Installing browser-use + playwright + langchain-openai",
            [*prefix, py, "-m", "pip", "install", *_PACKAGES],
            satisfied=have_pkgs,
        ),
        _Step(
            "Downloading the Chromium browser",
            [*prefix, py, "-m", "playwright", "install", "chromium"],
            satisfied=have_chromium,
        ),
        _Step(
            "Installing Chromium's system libraries",
            [*prefix, py, "-m", "playwright", "install-deps"],
            satisfied=have_chromium,
        ),
    ]


def _run(argv: list[str]) -> None:
    try:
        subprocess.run(argv, check=True)
    except FileNotFoundError as exc:
        raise BrowserSetupError(_fallback_msg(f"{argv[0]} not found")) from exc
    except subprocess.CalledProcessError as exc:
        raise BrowserSetupError(
            _fallback_msg(f"`{' '.join(argv)}` exited {exc.returncode}")
        ) from exc


def install_commerce_engine(progress: Callable[[str], None] | None = None) -> None:
    """Install the browser engine into Aria's runtime venv, idempotently. Streams
    each step through ``progress``. No-ops cleanly if everything is already present.
    Raises :class:`BrowserSetupError` (message includes the manual fallback command)
    on any failure."""
    say = progress or (lambda _m: None)
    venv = runtime_venv()
    if not _venv_python(venv).exists():
        raise BrowserSetupError(_fallback_msg(f"{_venv_python(venv)} not found"))

    prefix = _escalation_prefix(venv)
    if prefix:
        say(f"Some steps need administrator rights (via {prefix[0]}) — you may be "
            "prompted for your password.")

    for step in plan_steps(venv, prefix=prefix):
        if step.satisfied:
            say(f"✓ {step.desc} — already done.")
            continue
        say(f"→ {step.desc}…")
        _run(step.argv)

    say("Verifying the engine…")
    importlib.invalidate_caches()  # so a just-installed package is discoverable
    if not commerce_engine_ready():
        raise BrowserSetupError(_fallback_msg("the engine still isn't importable"))
    say("✓ Browser engine ready — food ordering is set up.")
