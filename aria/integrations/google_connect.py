"""Interactive `aria connect google` / `aria disconnect google` CLI."""

from __future__ import annotations

from aria.config.keyring import SecretStore
from aria.integrations.google_auth import (
    SETUP_GUIDE,
    connect,
    disconnect,
    has_client,
    store_client,
)


def connect_cli() -> int:
    secrets = SecretStore()
    if not has_client(secrets):
        print(SETUP_GUIDE)
        client_id = input("\nClient ID: ").strip()
        client_secret = input("Client secret: ").strip()
        if not (client_id and client_secret):
            print("Cancelled — I need both the Client ID and the secret.")
            return 1
        store_client(client_id, client_secret, secrets)

    print("\nOpening your browser to sign in to Google (Calendar + Gmail)…")
    try:
        ok = connect(secrets)
    except Exception as exc:  # noqa: BLE001
        print(f"Sign-in failed: {exc}")
        print("If it keeps failing, double-check the Client ID/secret and that both "
              "the Calendar API and Gmail API are enabled.")
        return 1
    if ok:
        print("✓ Connected to Google. Your calendar and email are ready.")
        return 0
    print("Couldn't connect — no client credentials were available.")
    return 1


def disconnect_cli() -> int:
    disconnect(SecretStore())
    print("Disconnected from Google — the token was revoked and removed.")
    return 0
