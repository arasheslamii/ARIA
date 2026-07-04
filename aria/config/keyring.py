"""Secret storage.

API keys are NEVER written to config files or logs. Preferred storage is the OS
keyring (libsecret via python-keyring). But under some sessions (e.g. headless
`uv run` with no unlocked Secret Service) ``keyring.set_password`` silently
no-ops, so we **verify every write by reading it back** and, if it didn't stick,
fall back to an encrypted-at-rest file in the app state dir (0600).

Lookup precedence on ``get``:
  1. an explicitly-exported env var (lets a dev override a stale stored key);
  2. the OS keyring;
  3. the encrypted file fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import keyring
from keyring.errors import KeyringError

from aria import APP_SLUG

# Logical secret names -> environment-variable overrides.
_ENV_FALLBACK = {
    "groq_api_key": "GROQ_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "fallback_api_key": "ARIA_FALLBACK_API_KEY",
    "commerce_api_key": "ARIA_COMMERCE_API_KEY",  # free Gemini AI Studio key
}


def _machine_seed() -> bytes:
    """A stable per-machine seed for the file fallback's keystream.

    Binds the ciphertext to this host so a copied file is useless elsewhere.
    This is obfuscation-at-rest bound to the machine plus 0600 perms — not a
    substitute for the OS keyring, just a safety net when it's unavailable.
    """
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            mid = Path(p).read_text().strip()
            if mid:
                return hashlib.sha256(f"aria-secret:{mid}".encode()).digest()
        except OSError:
            continue
    import socket

    return hashlib.sha256(f"aria-secret:{socket.gethostname()}".encode()).digest()


def _keystream(n: int) -> bytes:
    seed = _machine_seed()
    out = bytearray()
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()
        counter += 1
    return bytes(out[:n])


def _xor(data: bytes) -> bytes:
    ks = _keystream(len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


class _FileSecretStore:
    """0600 file fallback, used only when the OS keyring won't persist."""

    def __init__(self, path: Path | None = None) -> None:
        self._explicit = path

    def _path(self) -> Path:
        if self._explicit is not None:
            return self._explicit
        from aria.config.loader import state_dir

        return state_dir() / "secrets.enc"

    def _load(self) -> dict[str, str]:
        path = self._path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def get(self, name: str) -> str | None:
        blob = self._load().get(name)
        if not blob:
            return None
        try:
            return _xor(base64.b64decode(blob)).decode()
        except (ValueError, UnicodeDecodeError):
            return None

    def set(self, name: str, value: str) -> None:
        path = self._path()
        data = self._load()
        data[name] = base64.b64encode(_xor(value.encode())).decode()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        path.chmod(0o600)

    def delete(self, name: str) -> None:
        data = self._load()
        if data.pop(name, None) is not None:
            self._path().write_text(json.dumps(data))


class SecretStore:
    """Keyring-first secret store with a verified file fallback."""

    def __init__(self, service: str = APP_SLUG, *, file_store: _FileSecretStore | None = None) -> None:
        self.service = service
        self._file = file_store or _FileSecretStore()

    # --- internal keyring helpers -------------------------------------
    def _keyring_get(self, name: str) -> str | None:
        try:
            return keyring.get_password(self.service, name)
        except KeyringError:
            return None

    # --- public API ----------------------------------------------------
    def get(self, name: str) -> str | None:
        env = _ENV_FALLBACK.get(name)
        if env:
            env_value = os.environ.get(env)
            if env_value:
                return env_value
        kv = self._keyring_get(name)
        if kv:
            return kv
        return self._file.get(name)

    def set(self, name: str, value: str, *, durable: bool = False) -> str:
        """Store ``value`` and return the backend that persisted it ("keyring",
        "file", or "none").

        ``durable=True`` ALSO writes the machine-bound encrypted file even when the
        keyring accepts the write — needed for secrets with no env fallback that
        must be readable from the systemd --user daemon (whose keyring may differ
        from the interactive session that wrote them, e.g. the Google token).
        """
        try:
            keyring.set_password(self.service, name, value)
        except Exception:  # noqa: BLE001 - never raise on set; we verify below
            pass
        keyring_ok = self._keyring_get(name) == value

        file_ok = False
        if durable or not keyring_ok:  # always file on durable, or as fallback
            try:
                self._file.set(name, value)
                file_ok = self._file.get(name) == value
            except OSError:
                pass

        if keyring_ok:
            return "keyring"
        if file_ok:
            return "file"
        return "none"

    def delete(self, name: str) -> None:
        try:
            keyring.delete_password(self.service, name)
        except KeyringError:
            pass
        self._file.delete(name)

    def has(self, name: str) -> bool:
        return self.get(name) is not None
