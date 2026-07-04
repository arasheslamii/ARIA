"""Google OAuth + credentials, shared by Calendar and Gmail.

One sign-in (`aria connect google`) requests BOTH Calendar and Gmail scopes via
the installed-app loopback flow. The token is stored in the existing encrypted
secret store (keyring or 0600 encrypted file), refreshed automatically. Because
these scopes require the user's OWN OAuth client, the user provides a
client_id/secret once (see SETUP_GUIDE).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aria.config.keyring import SecretStore

# Google's client is SYNCHRONOUS and can block on the network. We run those calls
# in a DEDICATED, bounded thread pool — never the default loop executor — so a hung
# Google call can't starve PiperTTS (which uses the default executor) and freeze
# the whole assistant. Each call is also bounded by asyncio.wait_for.
_GOOGLE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aria-google")
_CALL_TIMEOUT_S = 15.0
_HTTP_TIMEOUT_S = 12.0


class GoogleTimeout(RuntimeError):
    """A Google call didn't return within the timeout."""


async def run_blocking(
    fn: Callable[[], Any],
    *,
    timeout: float = _CALL_TIMEOUT_S,  # noqa: ASYNC109 - a timeout arg is the right API here
) -> Any:
    """Run a blocking Google call in the isolated pool, bounded by a timeout."""
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(_GOOGLE_EXECUTOR, fn), timeout)
    except TimeoutError as exc:
        raise GoogleTimeout("timed out") from exc


def friendly_google_error(exc: BaseException) -> str:
    """Map a Google/network failure to a short spoken-friendly phrase."""
    if isinstance(exc, GoogleTimeout):
        return "it timed out"
    try:
        from googleapiclient.errors import HttpError

        if isinstance(exc, HttpError):
            status = getattr(exc.resp, "status", 0)
            if status in (401, 403):
                return "I'm not authorized — try reconnecting with `aria connect google`"
            if status == 429:
                return "Google is rate-limiting me right now — try again in a moment"
            return f"Google returned an error ({status})"
    except Exception:  # noqa: BLE001
        pass
    return "I couldn't reach Google"

# Calendar read/write + Gmail read/modify/send/draft (one consent for both).
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
]

_TOKEN_KEY = "google_token"
_CLIENT_KEY = "google_oauth_client"

SETUP_GUIDE = """\
One-time Google setup (~5 min, needs a Google account):
  1. Go to  https://console.cloud.google.com/  and create a project (any name).
  2. APIs & Services → Library → enable BOTH:
        • Google Calendar API
        • Gmail API
  3. APIs & Services → OAuth consent screen → External → fill the basics →
     add yourself as a Test user.
  4. APIs & Services → Credentials → Create Credentials → OAuth client ID →
     Application type: Desktop app → Create.
  5. Copy the Client ID and Client secret and paste them here.
Your credentials are stored locally (encrypted) and only ever sent to Google."""


class GoogleNotConnected(RuntimeError):
    """Raised when a Google tool is used before `aria connect google`."""


def _client_config(client_id: str, client_secret: str) -> dict:
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def is_connected(secrets: SecretStore | None = None) -> bool:
    secrets = secrets or SecretStore()
    return secrets.has(_TOKEN_KEY)


def load_credentials(secrets: SecretStore | None = None):
    """Return google Credentials (refreshing if needed) or None if not connected."""
    secrets = secrets or SecretStore()
    raw = secrets.get(_TOKEN_KEY)
    if not raw:
        return None
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    creds = Credentials.from_authorized_user_info(json.loads(raw), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        secrets.set(_TOKEN_KEY, creds.to_json(), durable=True)  # persist refreshed token
    return creds


def build_service(api: str, version: str, secrets: SecretStore | None = None):
    """Build a googleapiclient service (sync). Raises GoogleNotConnected if no token.

    Uses static (bundled) discovery and an HTTP timeout so a network blip can never
    hang a worker thread indefinitely."""
    creds = load_credentials(secrets)
    if creds is None:
        raise GoogleNotConnected("Not connected to Google — run `aria connect google`.")
    from googleapiclient.discovery import build

    try:
        import httplib2
        from google_auth_httplib2 import AuthorizedHttp

        authed = AuthorizedHttp(creds, http=httplib2.Http(timeout=_HTTP_TIMEOUT_S))
        return build(api, version, http=authed, cache_discovery=False, static_discovery=True)
    except Exception:  # noqa: BLE001 - fall back if static discovery isn't bundled
        return build(api, version, credentials=creds, cache_discovery=False)


def has_client(secrets: SecretStore | None = None) -> bool:
    secrets = secrets or SecretStore()
    return secrets.has(_CLIENT_KEY)


def store_client(client_id: str, client_secret: str, secrets: SecretStore | None = None) -> None:
    secrets = secrets or SecretStore()
    payload = json.dumps({"client_id": client_id, "client_secret": client_secret})
    secrets.set(_CLIENT_KEY, payload, durable=True)


def connect(
    secrets: SecretStore | None = None,
    *,
    client_id: str | None = None,
    client_secret: str | None = None,
    open_browser: bool = True,
) -> bool:
    """Run the loopback OAuth flow and store the token. Returns False if no client
    id/secret is available (caller should prompt for them first)."""
    secrets = secrets or SecretStore()
    if not (client_id and client_secret):
        stored = secrets.get(_CLIENT_KEY)
        if stored:
            d = json.loads(stored)
            client_id, client_secret = d["client_id"], d["client_secret"]
    if not (client_id and client_secret):
        return False
    store_client(client_id, client_secret, secrets)
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_config(
        _client_config(client_id, client_secret), SCOPES
    )
    creds = flow.run_local_server(port=0, open_browser=open_browser)
    # Durable: also write the machine-bound encrypted file so the systemd --user
    # daemon can read the token even if its keyring differs from this session's.
    secrets.set(_TOKEN_KEY, creds.to_json(), durable=True)
    return True


def disconnect(secrets: SecretStore | None = None) -> None:
    """Revoke the token at Google (best-effort) and delete it locally."""
    secrets = secrets or SecretStore()
    raw = secrets.get(_TOKEN_KEY)
    if raw:
        try:
            import httpx

            info = json.loads(raw)
            token = info.get("refresh_token") or info.get("token")
            if token:
                httpx.post(
                    "https://oauth2.googleapis.com/revoke", params={"token": token}, timeout=8
                )
        except Exception:  # noqa: BLE001 - revoke is best-effort
            pass
    secrets.delete(_TOKEN_KEY)
