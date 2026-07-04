"""Gmail tools: read/search/summarize inbox, draft, and (confirm-gated) send.

The Google client is sync, so every API call runs in a worker thread. All tools
that touch message content are marked ``sensitive`` so bodies/snippets never reach
debug logs or the audit trail. Sending is ``risk=confirm`` — Aria reads the
recipient, subject, and body back and only sends on an explicit yes.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from email.message import EmailMessage
from typing import Any

from aria.integrations.google_auth import (
    GoogleNotConnected,
    friendly_google_error,
    run_blocking,
)
from aria.tools.base import Tool, ToolError, ToolResult

ServiceProvider = Callable[[], Any]

_NOT_CONNECTED = "You're not connected to Google yet — run `aria connect google` first."


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _decode_part(data: str) -> str:
    return base64.urlsafe_b64decode(data.encode()).decode("utf-8", "replace")


def _extract_body(payload: dict) -> str:
    """Pull the text/plain body out of a Gmail message payload."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return _decode_part(payload["body"]["data"])
    for part in payload.get("parts", []) or []:
        text = _extract_body(part)
        if text:
            return text
    # last resort: any body data
    if payload.get("body", {}).get("data"):
        return _decode_part(payload["body"]["data"])
    return ""


class _GmailTool(Tool):
    def __init__(self, service_provider: ServiceProvider) -> None:
        self._provider = service_provider

    async def _service(self):
        try:
            return await run_blocking(self._provider)
        except GoogleNotConnected as exc:
            raise ToolError(_NOT_CONNECTED) from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"I couldn't reach your email ({friendly_google_error(exc)}).") from exc

    async def _call(self, fn):
        # Bounded in the dedicated Google pool — a hung call times out and the turn
        # COMPLETES with a friendly error; it never blocks the pipeline/next turn.
        try:
            return await run_blocking(fn)
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"I couldn't reach your email ({friendly_google_error(exc)}).") from exc


class ListEmailsTool(_GmailTool):
    name = "list_recent_emails"
    description = (
        "List recent emails (sender, subject, snippet). Use for 'read my latest "
        "emails', 'any unread?'. Optional Gmail query like 'is:unread' or 'from:bob'."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional Gmail search query."},
            "max_results": {"type": "integer", "description": "Default 8."},
        },
    }
    risk = "safe"
    sensitive = True

    async def run(self, **kwargs: Any) -> ToolResult:
        service = await self._service()
        query = str(kwargs.get("query") or "")
        limit = max(1, min(int(kwargs.get("max_results", 8)), 20))

        def _list():
            return (
                service.users()
                .messages()
                .list(userId="me", q=query, maxResults=limit)
                .execute()
            )

        listing = await self._call(_list)
        ids = [m["id"] for m in listing.get("messages", [])]
        if not ids:
            return ToolResult(content="no emails", spoken="Your inbox is clear — nothing there.")

        def _get_all():
            # SEQUENTIAL on purpose: the service shares one httplib2.Http (a single
            # TLS socket) and httplib2 is NOT thread-safe — concurrent execute()s
            # interleave on the socket and corrupt it (ssl WRONG_VERSION_NUMBER).
            # One thread, one socket, in order. For <=8 messages latency is fine.
            out = []
            for mid in ids:
                out.append(
                    service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=mid,
                        format="metadata",
                        metadataHeaders=["From", "Subject"],
                    )
                    .execute()
                )
            return out

        msgs = await self._call(_get_all)
        rows = []
        data = []
        for m in msgs:
            headers = m.get("payload", {}).get("headers", [])
            sender = _header(headers, "From")
            subject = _header(headers, "Subject") or "(no subject)"
            snippet = m.get("snippet", "")
            rows.append(f"[{m['id']}] from {sender}: {subject} — {snippet}")
            data.append({"id": m["id"], "from": sender, "subject": subject, "snippet": snippet})
        return ToolResult(content="\n".join(rows), data={"emails": data})


class SearchEmailsTool(ListEmailsTool):
    name = "search_emails"
    description = "Search the user's email with a Gmail query (e.g. 'from:bank invoice')."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Gmail search query."}},
        "required": ["query"],
    }


class ReadEmailTool(_GmailTool):
    name = "read_email"
    description = "Read the full body of one email by its id (from list_recent_emails)."
    parameters = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "The message id."}},
        "required": ["id"],
    }
    risk = "safe"
    sensitive = True

    async def run(self, **kwargs: Any) -> ToolResult:
        mid = str(kwargs.get("id") or "").strip()
        if not mid:
            raise ToolError("Which email? I need its id.")
        service = await self._service()

        def _get():
            return service.users().messages().get(userId="me", id=mid, format="full").execute()

        msg = await self._call(_get)
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        sender = _header(headers, "From")
        subject = _header(headers, "Subject") or "(no subject)"
        body = _extract_body(payload) or msg.get("snippet", "")
        return ToolResult(
            content=f"From: {sender}\nSubject: {subject}\n\n{body[:3500]}",
            data={"id": mid, "from": sender, "subject": subject},
        )


class DraftEmailTool(_GmailTool):
    name = "draft_email"
    description = (
        "Create a DRAFT email (does not send). Use to compose or draft a reply; the "
        "user can review or ask you to send it."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email."},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    }
    risk = "safe"
    sensitive = True

    async def run(self, **kwargs: Any) -> ToolResult:
        to, subject, body = (str(kwargs.get(k, "")).strip() for k in ("to", "subject", "body"))
        if not to:
            raise ToolError("Who should the draft go to?")
        raw = _build_raw(to, subject, body)
        service = await self._service()

        def _draft():
            return (
                service.users()
                .drafts()
                .create(userId="me", body={"message": {"raw": raw}})
                .execute()
            )

        await self._call(_draft)
        return ToolResult(
            content=f"drafted to {to}",
            data={"to": to, "subject": subject},
            spoken=f"I've drafted that email to {to}. Want me to send it?",
        )


class SendEmailTool(_GmailTool):
    name = "send_email"
    description = (
        "Send an email. Provide to, subject, body. This actually SENDS — it is "
        "confirm-gated, so it's read back and only sent on an explicit yes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email."},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    }
    risk = "confirm"  # never auto-send
    sensitive = True

    def confirm_summary(self, arguments: dict[str, Any]) -> str:
        to = arguments.get("to", "someone")
        subject = arguments.get("subject", "")
        body = arguments.get("body", "")
        return f"send an email to {to}, subject '{subject}', that says: {body}"

    async def run(self, **kwargs: Any) -> ToolResult:
        to, subject, body = (str(kwargs.get(k, "")).strip() for k in ("to", "subject", "body"))
        if not to:
            raise ToolError("Who should I send it to?")
        raw = _build_raw(to, subject, body)
        service = await self._service()

        def _send():
            return service.users().messages().send(userId="me", body={"raw": raw}).execute()

        await self._call(_send)
        return ToolResult(
            content=f"sent to {to}", spoken=f"Sent — your email's on its way to {to}."
        )


def _build_raw(to: str, subject: str, body: str) -> str:
    msg = EmailMessage()
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode()


def gmail_tools(service_provider: ServiceProvider) -> list[Tool]:
    return [
        ListEmailsTool(service_provider),
        SearchEmailsTool(service_provider),
        ReadEmailTool(service_provider),
        DraftEmailTool(service_provider),
        SendEmailTool(service_provider),
    ]
