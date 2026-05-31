"""Codex endpoint wiring tests: routing -> request map -> transport -> response.

The transport (``codex_client.stream_events``) is monkeypatched to emit canned
Responses events, so no token or network is involved.
"""
import json

import pytest
from fastapi.testclient import TestClient

from oauth_proxy import codex_auth, codex_client
from oauth_proxy.app import build_app
from oauth_proxy.config import Config


class _FakeCodexTokens:
    def __init__(self, *, error=None, logged_in=True):
        self._error = error
        self._logged_in = logged_in

    def headers(self):
        if self._error is not None:
            raise self._error
        return {"Authorization": "Bearer t", "ChatGPT-Account-ID": "acc", "originator": "codex_cli_rs"}

    def is_logged_in(self):
        return self._logged_in


def _text_events():
    """Mirrors the real Codex backend: text via deltas, the finalized message in
    output_item.done, and an EMPTY output in response.completed (usage only)."""
    mid = "msg_1"
    return [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": mid, "type": "message", "role": "assistant", "content": []}},
        {"type": "response.output_text.delta", "item_id": mid, "delta": "Hi"},
        {"type": "response.output_text.delta", "item_id": mid, "delta": " there"},
        {"type": "response.output_item.done", "output_index": 0,
         "item": {"id": mid, "type": "message", "role": "assistant",
                  "content": [{"type": "output_text", "text": "Hi there"}]}},
        {"type": "response.completed", "response": {
            "id": "resp_1", "status": "completed", "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}}},
    ]


def _make_client(monkeypatch, events=None, *, tokens=None, stream_fn=None, models=None):
    if stream_fn is not None:
        monkeypatch.setattr(codex_client, "stream_events", stream_fn)
    else:
        monkeypatch.setattr(codex_client, "stream_events", lambda body, **kw: iter(events or []))
    # Keep /v1/models hermetic: stub the live allowlist fetch (no network).
    monkeypatch.setattr(codex_client, "list_models", lambda headers, **kw: list(models or ["gpt-5.2"]))
    app = build_app(Config(), codex_token_provider=tokens or _FakeCodexTokens())
    return TestClient(app)


def test_codex_chat_non_stream(monkeypatch):
    client = _make_client(monkeypatch, _text_events())
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-5-codex", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-5-codex"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "Hi there"
    assert choice["finish_reason"] == "stop"
    assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_codex_chat_stream(monkeypatch):
    client = _make_client(monkeypatch, _text_events())
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "gpt-5-codex", "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        chunks = []
        for line in resp.iter_lines():
            if line and line.startswith("data: ") and "[DONE]" not in line:
                chunks.append(json.loads(line[len("data: "):]))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "Hi there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_codex_responses_passthrough_non_stream(monkeypatch):
    client = _make_client(monkeypatch, _text_events())
    resp = client.post("/v1/responses", json={
        "model": "gpt-5-codex",
        "input": [{"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "hi"}]}],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "resp_1"
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"] == "Hi there"


def test_responses_rejects_non_responses_provider(monkeypatch):
    client = _make_client(monkeypatch, _text_events())
    resp = client.post("/v1/responses", json={"model": "claude-opus-4-8", "input": []})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unsupported_model"


def test_codex_token_error_returns_401(monkeypatch):
    client = _make_client(
        monkeypatch, _text_events(),
        tokens=_FakeCodexTokens(error=codex_auth.TokenError("run login codex")),
    )
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-5-codex", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "oauth_token_unavailable"


def test_codex_backend_403_maps_to_permission_error(monkeypatch):
    def boom(body, **kw):
        raise codex_client.CodexHTTPError(403, "subscription not entitled")
        yield  # pragma: no cover - makes this a generator function

    client = _make_client(monkeypatch, stream_fn=boom)
    resp = client.post("/v1/chat/completions", json={
        "model": "gpt-5-codex", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "subscription_not_entitled"


def test_models_catalog_lists_live_codex_models_when_logged_in(monkeypatch):
    # Logged-in Codex provider -> live-fetched codex models advertised.
    client = _make_client(monkeypatch, _text_events(),
                          tokens=_FakeCodexTokens(logged_in=True), models=["gpt-5.2"])
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert "gpt-5.2" in ids
    # Grok is not wired into this server yet, so it must NOT be advertised.
    assert "grok-4.3" not in ids


def test_list_models_filters_hidden_models(monkeypatch):
    class _Resp:
        status_code = 200

        def json(self):
            return {"models": [
                {"slug": "gpt-5.5", "visibility": "list"},
                {"slug": "codex-auto-review", "visibility": "hide"},
                {"slug": "gpt-5.2"},  # no visibility -> kept
                {"display_name": "no slug"},
            ]}

    monkeypatch.setattr(codex_client.httpx, "get", lambda *a, **k: _Resp())
    assert codex_client.list_models({"Authorization": "Bearer t"}) == ["gpt-5.5", "gpt-5.2"]


def test_models_catalog_omits_codex_when_not_logged_in(monkeypatch):
    client = _make_client(monkeypatch, _text_events(), tokens=_FakeCodexTokens(logged_in=False))
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert "gpt-5-codex" not in ids
