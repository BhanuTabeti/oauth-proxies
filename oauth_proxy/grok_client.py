"""Thin OpenAI-compatible passthrough to the xAI (SuperGrok) backend.

xAI's ``api.x.ai/v1`` natively speaks the OpenAI Chat Completions AND Responses
APIs, so the Grok provider needs no request/response translation — it injects
the OAuth bearer and forwards. Non-stream calls return the upstream JSON
verbatim; streaming calls forward the upstream SSE bytes unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator

import httpx

from oauth_proxy.grok_auth import BASE_URL


class GrokHTTPError(RuntimeError):
    """Non-2xx from the xAI backend. Carries the HTTP status for classification."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _headers(auth_headers: Dict[str, str]) -> Dict[str, str]:
    return {**auth_headers, "Content-Type": "application/json"}


def post_json(path: str, body: Dict[str, Any], *, auth_headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    """POST a non-streaming request; return the upstream JSON (already OpenAI shape)."""
    resp = httpx.post(BASE_URL + path, json=body, headers=_headers(auth_headers), timeout=timeout)
    if resp.status_code >= 400:
        raise GrokHTTPError(resp.status_code, resp.text[:500] or f"HTTP {resp.status_code}")
    return resp.json()


def stream_raw(path: str, body: Dict[str, Any], *, auth_headers: Dict[str, str], timeout: float) -> Iterator[bytes]:
    """POST a streaming request; yield the upstream SSE bytes verbatim."""
    headers = {**_headers(auth_headers), "Accept": "text/event-stream"}
    payload = {**body, "stream": True}
    with httpx.stream("POST", BASE_URL + path, json=payload, headers=headers, timeout=timeout) as resp:
        if resp.status_code >= 400:
            detail = resp.read().decode(errors="replace")
            raise GrokHTTPError(resp.status_code, detail[:500] or f"HTTP {resp.status_code}")
        for chunk in resp.iter_bytes():
            if chunk:
                yield chunk


def list_models(auth_headers: Dict[str, str], *, timeout: float = 15.0) -> list:
    """Fetch the live model list from xAI (``GET /models``); return model ids."""
    resp = httpx.get(BASE_URL + "/models", headers=_headers(auth_headers), timeout=timeout)
    if resp.status_code >= 400:
        raise GrokHTTPError(resp.status_code, resp.text[:300] or f"HTTP {resp.status_code}")
    data = resp.json()
    return [m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
