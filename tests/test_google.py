"""Google OAuth/credentials + Calendar tools (Milestone 4 Part B), all mocked."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from aria.config.keyring import SecretStore, _FileSecretStore
from aria.integrations import google_auth
from aria.tools.base import ToolError


@pytest.fixture
def secrets(tmp_path, monkeypatch):
    # Force the encrypted-file fallback so tests never touch the real keyring.
    monkeypatch.setattr("aria.config.keyring.keyring.set_password", lambda *a, **k: None)
    monkeypatch.setattr("aria.config.keyring.keyring.get_password", lambda *a, **k: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return SecretStore(file_store=_FileSecretStore(tmp_path / "secrets.enc"))


def _token_json() -> str:
    # A future (naive-UTC) expiry so load_credentials doesn't try to refresh.
    future = (datetime.now(UTC) + timedelta(hours=1)).replace(tzinfo=None).isoformat()
    return json.dumps(
        {
            "token": "access-x",
            "refresh_token": "refresh-y",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "cid",
            "client_secret": "csec",
            "scopes": google_auth.SCOPES,
            "expiry": future,
        }
    )


# --- OAuth / credentials --------------------------------------------------
def test_not_connected_by_default(secrets):
    assert google_auth.is_connected(secrets) is False
    assert google_auth.load_credentials(secrets) is None


def test_build_service_without_token_raises(secrets):
    with pytest.raises(google_auth.GoogleNotConnected):
        google_auth.build_service("calendar", "v3", secrets)


def test_store_and_load_client_and_token(secrets):
    google_auth.store_client("cid", "csec", secrets)
    assert google_auth.has_client(secrets)

    secrets.set("google_token", _token_json())
    assert google_auth.is_connected(secrets)
    creds = google_auth.load_credentials(secrets)  # builds Credentials, no refresh
    assert creds is not None and creds.token == "access-x"


def test_durable_set_also_writes_file_when_keyring_works(tmp_path, monkeypatch):
    # BUG 1: the Google token has no env fallback, and the systemd --user daemon's
    # keyring may differ from the interactive session that wrote it. A durable write
    # must ALSO land in the machine-bound encrypted file so the daemon can read it
    # even when the keyring "accepted" the write in another session.
    store: dict[str, str] = {}
    monkeypatch.setattr(
        "aria.config.keyring.keyring.set_password",
        lambda s, n, v: store.__setitem__(n, v),
    )
    monkeypatch.setattr(
        "aria.config.keyring.keyring.get_password", lambda s, n: store.get(n)
    )
    file_store = _FileSecretStore(tmp_path / "secrets.enc")
    secrets = SecretStore(file_store=file_store)

    token = _token_json()
    backend = secrets.set("google_token", token, durable=True)
    assert backend == "keyring"  # the keyring did accept it
    assert file_store.get("google_token") == token  # ...AND the file has it too


def test_store_client_and_connect_token_are_durable(secrets):
    # store_client + the connect() token write must reach the file fallback so the
    # daemon stays connected even if the keyring drops the entries.
    google_auth.store_client("cid", "csec", secrets)
    assert secrets._file.get(google_auth._CLIENT_KEY)  # client persisted to file


def test_disconnect_removes_token(secrets, monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "post", lambda *a, **k: None)  # stub the revoke call
    secrets.set("google_token", _token_json())
    google_auth.disconnect(secrets)
    assert not google_auth.is_connected(secrets)


# --- fake Google Calendar service ----------------------------------------
class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Events:
    def __init__(self, list_result, insert_result, captured):
        self._list = list_result
        self._insert = insert_result
        self._cap = captured

    def list(self, **kwargs):
        self._cap["list"] = kwargs
        return _Req(self._list)

    def insert(self, **kwargs):
        self._cap["insert"] = kwargs
        return _Req(self._insert)


class _Service:
    def __init__(self, list_result=None, insert_result=None, captured=None):
        self._events = _Events(
            list_result or {"items": []}, insert_result or {"id": "evt1"}, captured
        )

    def events(self):
        return self._events


def _provider(**kwargs):
    return lambda: _Service(**kwargs)


# --- list_events ----------------------------------------------------------
async def test_list_events_formats_agenda():
    from aria.tools.calendar_tool import ListEventsTool

    events = {
        "items": [
            {"summary": "Standup", "start": {"dateTime": "2026-06-26T10:00:00+01:00"}},
            {"summary": "Lunch with Sam", "start": {"dateTime": "2026-06-26T12:30:00+01:00"}},
        ]
    }
    res = await ListEventsTool(_provider(list_result=events, captured={})).run(range="today")
    assert "2 things" in res.spoken
    assert "Standup at 10:00 AM" in res.spoken and "Lunch with Sam" in res.spoken


async def test_list_events_empty_is_clear():
    from aria.tools.calendar_tool import ListEventsTool

    res = await ListEventsTool(_provider(captured={})).run(range="today")
    assert "nothing on" in res.spoken.lower()


# --- create_event: confirm-gated + creates -------------------------------
def test_create_event_is_confirm_gated():
    from aria.safety.permissions import classify
    from aria.tools.calendar_tool import CreateEventTool

    tool = CreateEventTool(_provider(captured={}))
    assert tool.risk == "confirm"
    assert classify(tool, {"title": "x", "start": "friday 3pm"}).risk == "confirm"


async def test_create_event_inserts_with_parsed_time():
    from aria.tools.calendar_tool import CreateEventTool

    cap: dict = {}
    res = await CreateEventTool(_provider(captured=cap)).run(
        title="Dentist", start="tomorrow at 3pm"
    )
    body = cap["insert"]["body"]
    assert body["summary"] == "Dentist"
    assert "dateTime" in body["start"] and "dateTime" in body["end"]
    assert "added dentist" in res.spoken.lower()


async def test_create_event_echoes_api_returned_time_and_id():
    # The model must echo the time the API ACTUALLY scheduled, not a guessed one.
    from aria.tools.calendar_tool import CreateEventTool

    insert_result = {"id": "evtX", "start": {"dateTime": "2026-07-01T09:00:00+01:00"}}
    res = await CreateEventTool(
        _provider(insert_result=insert_result, captured={})
    ).run(title="Concert", start="tomorrow at 8pm")  # parsed 8pm, API says 9 AM
    assert res.data["id"] == "evtX"  # the verified, API-returned id
    assert "9:00 AM" in res.spoken  # the API time wins over the parsed 8pm
    assert "9:00 AM" in res.data["when"]


async def test_calendar_tool_friendly_when_not_connected():
    from aria.tools.calendar_tool import ListEventsTool

    def boom():
        raise google_auth.GoogleNotConnected("no token")

    with pytest.raises(ToolError, match="connect google"):
        await ListEventsTool(boom).run(range="today")


# --- BUG 1: current date in context + no false-success on create ----------
async def test_current_date_injected_into_context():
    from datetime import datetime

    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.tools.base import ToolRegistry

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=object(), registry=ToolRegistry(), memory=mem,
        reasoning_model="r", fast_model="f",
    )
    msgs = await orch._base_messages()
    sys_text = msgs[0].content
    await mem.close()
    assert "Today is" in sys_text
    assert datetime.now().strftime("%A") in sys_text  # the real weekday


async def test_create_event_reports_failure_when_no_id():
    from aria.tools.calendar_tool import CreateEventTool

    # Insert "succeeds" but returns no id -> must NOT claim success.
    res_provider = _provider(insert_result={"status": "weird"}, captured={})  # no "id"
    with pytest.raises(ToolError, match="didn't go through"):
        await CreateEventTool(res_provider).run(title="X", start="tomorrow at 3pm")
