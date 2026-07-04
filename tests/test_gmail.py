"""Gmail tools (Milestone 4 Part C), mocked — incl. confirm-gated send + privacy."""

from __future__ import annotations

import base64

import pytest

from aria.integrations.google_auth import GoogleNotConnected
from aria.tools.base import ToolError


# --- fake Gmail service ---------------------------------------------------
class _Req:
    def __init__(self, result):
        self._result = result

    def execute(self):
        return self._result


class _Messages:
    def __init__(self, parent):
        self._p = parent

    def list(self, **kwargs):
        self._p.captured["list"] = kwargs
        return _Req({"messages": [{"id": "m1"}, {"id": "m2"}]})

    def get(self, **kwargs):
        self._p.captured.setdefault("get", []).append(kwargs)
        mid = kwargs["id"]
        if kwargs.get("format") == "full":
            body = base64.urlsafe_b64encode(b"The full body text.").decode()
            return _Req(
                {
                    "id": mid,
                    "snippet": "snippet",
                    "payload": {
                        "mimeType": "text/plain",
                        "body": {"data": body},
                        "headers": [
                            {"name": "From", "value": "bob@x.com"},
                            {"name": "Subject", "value": "Lunch?"},
                        ],
                    },
                }
            )
        return _Req(
            {
                "id": mid,
                "snippet": f"snippet for {mid}",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "bob@x.com"},
                        {"name": "Subject", "value": "Lunch?"},
                    ]
                },
            }
        )

    def send(self, **kwargs):
        self._p.captured["send"] = kwargs
        return _Req({"id": "sent1"})


class _Drafts:
    def __init__(self, parent):
        self._p = parent

    def create(self, **kwargs):
        self._p.captured["draft"] = kwargs
        return _Req({"id": "draft1"})


class _Users:
    def __init__(self, parent):
        self._p = parent

    def messages(self):
        return _Messages(self._p)

    def drafts(self):
        return _Drafts(self._p)


class _Gmail:
    def __init__(self):
        self.captured: dict = {}

    def users(self):
        return _Users(self)


def _provider(svc):
    return lambda: svc


# --- list / search / read -------------------------------------------------
async def test_list_recent_emails_summary():
    from aria.tools.gmail_tool import ListEmailsTool

    svc = _Gmail()
    res = await ListEmailsTool(_provider(svc)).run(query="is:unread")
    assert svc.captured["list"]["q"] == "is:unread"
    assert len(res.data["emails"]) == 2
    assert res.data["emails"][0]["from"] == "bob@x.com"
    assert res.data["emails"][0]["subject"] == "Lunch?"


# --- BUG 2 regression: per-message fetch must NOT race on the shared socket --
class _ConcurrencyDetectingMessages:
    """Mimics httplib2's non-thread-safety: if two ``get()`` calls overlap on the
    one shared TLS socket, the socket is corrupted (the real symptom is
    ``ssl.SSLError WRONG_VERSION_NUMBER``). We raise that here on any overlap."""

    def __init__(self, parent):
        self._p = parent

    def list(self, **kwargs):
        ids = [{"id": f"m{i}"} for i in range(8)]
        return _Req({"messages": ids})

    def get(self, **kwargs):
        import threading
        import time as _time

        self._p.threads.add(threading.get_ident())
        if self._p.in_flight:  # another get() is already on the socket
            raise OSError("ssl WRONG_VERSION_NUMBER: concurrent use of shared Http")
        self._p.in_flight = True
        try:
            _time.sleep(0.01)  # widen the window so a real race would be caught
        finally:
            self._p.in_flight = False
        mid = kwargs["id"]
        return _Req(
            {
                "id": mid,
                "snippet": f"snippet for {mid}",
                "payload": {
                    "headers": [
                        {"name": "From", "value": "bob@x.com"},
                        {"name": "Subject", "value": "Lunch?"},
                    ]
                },
            }
        )


class _RacyGmail:
    def __init__(self):
        self.in_flight = False
        self.threads: set[int] = set()

    def users(self):
        class _U:
            def __init__(self, p):
                self._p = p

            def messages(self):
                return _ConcurrencyDetectingMessages(self._p)

        return _U(self)


async def test_list_emails_fetches_sequentially_no_socket_race():
    # The service shares one httplib2.Http (a single TLS socket); httplib2 is not
    # thread-safe. The list path must fetch messages sequentially so concurrent
    # get()s can never interleave on the socket and corrupt the TLS stream.
    from aria.tools.gmail_tool import ListEmailsTool

    svc = _RacyGmail()
    res = await ListEmailsTool(_provider(svc)).run()  # would raise OSError if racing
    assert len(res.data["emails"]) == 8
    assert len(svc.threads) == 1  # all gets ran on ONE worker thread (one socket)


async def test_read_email_returns_full_body():
    from aria.tools.gmail_tool import ReadEmailTool

    res = await ReadEmailTool(_provider(_Gmail())).run(id="m1")
    assert "The full body text." in res.content
    assert "Lunch?" in res.content


async def test_search_emails_requires_query():
    from aria.tools.gmail_tool import SearchEmailsTool

    svc = _Gmail()
    await SearchEmailsTool(_provider(svc)).run(query="from:bank")
    assert svc.captured["list"]["q"] == "from:bank"


# --- send is confirm-gated and reads the draft back -----------------------
def test_send_email_is_confirm_gated_with_full_readback():
    from aria.tools.gmail_tool import SendEmailTool

    tool = SendEmailTool(_provider(_Gmail()))
    assert tool.risk == "confirm"
    summary = tool.confirm_summary({"to": "bob@x.com", "subject": "Hi", "body": "See you at 5."})
    assert "bob@x.com" in summary and "Hi" in summary and "See you at 5." in summary


async def test_send_email_sends_via_api():
    from aria.tools.gmail_tool import SendEmailTool

    svc = _Gmail()
    res = await SendEmailTool(_provider(svc)).run(to="bob@x.com", subject="Hi", body="Yo")
    assert "send" in svc.captured  # actually called the send endpoint
    assert "on its way to bob@x.com" in res.spoken


async def test_draft_email_creates_draft_not_send():
    from aria.tools.gmail_tool import DraftEmailTool

    svc = _Gmail()
    res = await DraftEmailTool(_provider(svc)).run(to="a@b.com", subject="Hi", body="Hey")
    assert "draft" in svc.captured and "send" not in svc.captured
    assert "want me to send it" in res.spoken.lower()


async def test_gmail_friendly_when_not_connected():
    from aria.tools.gmail_tool import ListEmailsTool

    def boom():
        raise GoogleNotConnected("no token")

    with pytest.raises(ToolError, match="connect google"):
        await ListEmailsTool(boom).run()


# --- privacy: email tools are sensitive (kept out of logs/audit) ----------
def test_email_tools_are_marked_sensitive():
    from aria.tools.gmail_tool import (
        ListEmailsTool,
        ReadEmailTool,
        SendEmailTool,
    )

    for cls in (ListEmailsTool, ReadEmailTool, SendEmailTool):
        assert cls.sensitive is True


async def test_audit_redacts_sensitive_arguments(tmp_path):
    # send_email body must NOT appear in the audit trail.
    from aria.core.executor import ExecConfig, ToolExecutor
    from aria.safety.audit import AuditLog
    from aria.tools.gmail_tool import SendEmailTool

    audit_path = tmp_path / "audit.log"
    execu = ToolExecutor(AuditLog(audit_path), ExecConfig(require_confirmation=False))

    async def yes(_name):
        return True

    await execu.execute(
        SendEmailTool(_provider(_Gmail())),
        {"to": "bob@x.com", "subject": "Secret", "body": "TOP SECRET BODY"},
        confirm=yes,
    )
    logged = audit_path.read_text()
    assert "TOP SECRET BODY" not in logged  # body never written
    assert "Secret" not in logged  # subject never written
    assert "_redacted_fields" in logged  # but the action is still recorded


# --- BUG 2: timeouts/errors never hang; the turn completes -----------------
async def test_run_blocking_times_out_fast():
    import time as _time

    from aria.integrations.google_auth import GoogleTimeout, run_blocking

    with pytest.raises(GoogleTimeout):
        await run_blocking(lambda: _time.sleep(5), timeout=0.1)


def test_google_uses_dedicated_executor():
    # Google calls must NOT run on the default loop executor (which PiperTTS uses),
    # so a hung Google thread can't starve TTS and freeze later turns.
    from concurrent.futures import ThreadPoolExecutor

    from aria.integrations.google_auth import _GOOGLE_EXECUTOR

    assert isinstance(_GOOGLE_EXECUTOR, ThreadPoolExecutor)


def test_friendly_google_error_maps_status():
    import httplib2
    from googleapiclient.errors import HttpError

    from aria.integrations.google_auth import GoogleTimeout, friendly_google_error

    assert "timed out" in friendly_google_error(GoogleTimeout("x"))
    auth = HttpError(httplib2.Response({"status": 401}), b"{}")
    assert "reconnect" in friendly_google_error(auth).lower()
    quota = HttpError(httplib2.Response({"status": 429}), b"{}")
    assert "rate-limit" in friendly_google_error(quota).lower()


async def test_gmail_timeout_is_friendly_not_a_hang():
    from aria.integrations.google_auth import GoogleTimeout
    from aria.tools.gmail_tool import ListEmailsTool

    def hangs():
        raise GoogleTimeout("timed out")  # as run_blocking would surface

    with pytest.raises(ToolError, match="timed out"):
        await ListEmailsTool(hangs).run()


async def test_pipeline_recovers_after_a_tool_error():
    # A failing/hung tool must complete the turn with a spoken error AND the NEXT
    # turn must still work (not frozen).
    from aria.core.memory import Memory
    from aria.core.orchestrator import Orchestrator
    from aria.llm.base import ChatResult, ToolCall
    from aria.tools.base import Tool, ToolRegistry

    class _HangTool(Tool):
        name = "list_recent_emails"
        description = "email"
        parameters = {"type": "object", "properties": {}}
        risk = "safe"

        async def run(self, **k):
            raise ToolError("I couldn't reach your email (it timed out).")

    reg = ToolRegistry()
    reg.register(_HangTool())
    turn = {"n": 0}

    class LLM:
        async def chat(self, messages, *, model, tools=None, temperature=None, max_tokens=None):
            if model == "small":  # router
                route = "tool" if turn["n"] == 0 else "chitchat"
                return ChatResult(content=f'{{"route":"{route}","needs_tools":[],"reason":"x"}}')
            if turn["n"] == 0 and not getattr(self, "_called", False):
                self._called = True
                return ChatResult(content="", tool_calls=[ToolCall("e", "list_recent_emails", {})])
            return ChatResult(content="No worries.")

        async def stream(self, messages, *, model, temperature=None, max_tokens=None):
            yield "All good."

    mem = Memory(":memory:")
    await mem.open()
    orch = Orchestrator(
        llm=LLM(), registry=reg, memory=mem, reasoning_model="big", fast_model="small"
    )

    out1 = "".join([d async for d in orch.respond("any unread emails?")])
    assert out1.strip()  # turn COMPLETED with some spoken text (not frozen)
    turn["n"] = 1
    out2 = "".join([d async for d in orch.respond("how are you?")])
    assert out2.strip()  # the NEXT turn still works
    await mem.close()
