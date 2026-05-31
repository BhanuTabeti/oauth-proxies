"""Thin HTTP/SSE transport to the Codex (ChatGPT-subscription) Responses backend.

Small and auditable on purpose: it attaches the OAuth credential + Codex client
identity headers and POSTs to ``chatgpt.com/backend-api/codex/responses``. The
backend streams Server-Sent Events; we always request a stream upstream and let
callers either forward the events (streaming client) or collect the final
``response.completed`` object (non-streaming client).
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, Optional

import httpx

from oauth_proxy.codex_auth import (
    CLIENT_VERSION,
    MODELS_ENDPOINT,
    ORIGINATOR,
    RESPONSES_ENDPOINT,
)

# A Codex-CLI-style User-Agent; the subscription backend expects requests that
# look like the official client. Not a secret; version is cosmetic.
_USER_AGENT = "codex_cli_rs/0.0.0 (oauth-proxy)"


class CodexHTTPError(RuntimeError):
    """Non-2xx from the Codex backend. Carries the HTTP status for classification."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _transport_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": _USER_AGENT,
        "originator": ORIGINATOR,
        "session_id": str(uuid.uuid4()),
    }


def _parse_sse(resp: httpx.Response) -> Iterator[Dict[str, Any]]:
    """Yield JSON event objects from an SSE response body.

    Responses-API events carry their kind in a ``type`` field on the ``data:``
    payload, so we ignore ``event:`` lines and parse each ``data:`` JSON object.
    """
    for raw in resp.iter_lines():
        line = (raw.decode() if isinstance(raw, (bytes, bytearray)) else raw).strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def stream_events(
    body: Dict[str, Any],
    *,
    auth_headers: Dict[str, str],
    timeout: float,
    url: str = RESPONSES_ENDPOINT,
) -> Iterator[Dict[str, Any]]:
    """POST the Responses body (forcing ``stream:true``) and yield event dicts.

    ``auth_headers`` come from ``CodexTokenProvider.headers()`` (Authorization +
    ChatGPT-Account-ID + originator). Raises ``CodexHTTPError`` on a non-2xx
    status, reading the error body for a useful message.
    """
    payload = {**body, "stream": True}
    headers = {**_transport_headers(), **auth_headers}
    with httpx.stream("POST", url, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status_code >= 400:
            detail = resp.read().decode(errors="replace")
            raise CodexHTTPError(resp.status_code, detail[:500] or f"HTTP {resp.status_code}")
        yield from _parse_sse(resp)


def collect_final(events: Iterator[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drain an event stream and return a complete Responses object.

    The Codex backend streams finalized items via ``response.output_item.done``
    and sends usage in ``response.completed`` with an EMPTY ``output``. So we
    assemble ``output`` from the per-item ``done`` events and merge usage/status
    from ``completed`` (falling back to the completed snapshot's own ``output``
    if a backend ever populates it). Raises on ``response.failed`` / ``error``.
    """
    from oauth_proxy.codex_stream_mapping import CodexStreamError

    completed: Optional[Dict[str, Any]] = None
    items: list = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        etype = ev.get("type")
        if etype == "response.output_item.done":
            item = ev.get("item")
            if item:
                items.append(item)
        elif etype == "response.completed":
            completed = ev.get("response") or {}
        elif etype in {"response.failed", "error"}:
            resp = ev.get("response") or {}
            err = resp.get("error") or ev.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise CodexStreamError(msg or "responses stream failed")

    if completed is None and not items:
        return None
    final = dict(completed or {})
    if not final.get("output"):
        final["output"] = items
    return final


def list_models(
    auth_headers: Dict[str, str],
    *,
    timeout: float = 15.0,
    url: str = MODELS_ENDPOINT,
    client_version: str = CLIENT_VERSION,
) -> list:
    """Fetch the live model allowlist for the logged-in ChatGPT account.

    Returns the accepted model slugs. Raises ``CodexHTTPError`` on a non-2xx.
    """
    headers = {**_transport_headers(), **auth_headers}
    headers["Accept"] = "application/json"
    resp = httpx.get(
        url, params={"client_version": client_version}, headers=headers, timeout=timeout
    )
    if resp.status_code >= 400:
        raise CodexHTTPError(resp.status_code, resp.text[:300] or f"HTTP {resp.status_code}")
    data = resp.json()
    # Skip models the backend marks hidden (e.g. internal "codex-auto-review").
    return [
        m["slug"]
        for m in data.get("models", [])
        if isinstance(m, dict) and m.get("slug") and m.get("visibility") != "hide"
    ]
