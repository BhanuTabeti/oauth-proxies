"""Grok (xAI) tests: OAuth helpers + endpoint passthrough (mocked transport)."""
from __future__ import annotations

import json
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from oauth_proxy import grok_auth, grok_client
from oauth_proxy.app import build_app
from oauth_proxy.config import Config
from oauth_proxy.routing import route_provider


# ── Routing ───────────────────────────────────────────────────────────────

def test_routing_grok():
    assert route_provider("grok-4.3") == "grok"
    assert route_provider("grok-4.20-multi-agent-0309") == "grok"


# ── OAuth helpers ─────────────────────────────────────────────────────────

def test_authorize_url_has_xai_params_incl_plan_generic():
    url = grok_auth._build_authorize_url(
        "https://auth.x.ai/oauth2/authorize",
        redirect_uri="http://127.0.0.1:56121/callback",
        code_challenge="CH", state="ST", nonce="NO",
    )
    base, _, qs = url.partition("?")
    assert base == "https://auth.x.ai/oauth2/authorize"
    p = dict(urllib.parse.parse_qsl(qs))
    assert p["client_id"] == "b1a00492-073a-47ea-816f-4c329264a828"
    assert p["code_challenge_method"] == "S256"
    assert p["plan"] == "generic"          # load-bearing
    assert p["redirect_uri"] == "http://127.0.0.1:56121/callback"
    assert p["state"] == "ST" and p["nonce"] == "NO"
    assert "grok-cli:access" in p["scope"]


def test_record_carries_forward_and_keeps_token_endpoint():
    prev = {"refresh_token": "r_old", "token_endpoint": "https://auth.x.ai/oauth2/token"}
    rec = grok_auth._record_from_token_response(
        {"access_token": "at", "expires_in": 100}, prev=prev,
        token_endpoint="https://auth.x.ai/oauth2/token", now_ms=0,
    )
    assert rec["access_token"] == "at"
    assert rec["refresh_token"] == "r_old"
    assert rec["expires_at"] == 100_000
    assert rec["token_endpoint"] == "https://auth.x.ai/oauth2/token"


def test_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    assert grok_auth.read_credentials() is None
    grok_auth.write_credentials({"access_token": "x"})
    assert grok_auth.read_credentials() == {"access_token": "x"}


def _future_ms(s=3600):
    import time
    return int(time.time() * 1000) + s * 1000


def test_get_token_refreshes_via_stored_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    grok_auth.write_credentials(
        {"access_token": "stale", "refresh_token": "r1", "expires_at": 1,
         "token_endpoint": "https://auth.x.ai/oauth2/token"}
    )
    seen = {}

    def fake_refresh(endpoint, refresh_token, *, timeout):
        seen["endpoint"] = endpoint
        return {"access_token": "new", "refresh_token": "r2", "expires_in": 3600}

    monkeypatch.setattr(grok_auth, "_refresh", fake_refresh)
    assert grok_auth.GrokTokenProvider().get_token() == "new"
    assert seen["endpoint"] == "https://auth.x.ai/oauth2/token"


def test_get_token_no_creds_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    with pytest.raises(grok_auth.TokenError, match="login grok"):
        grok_auth.GrokTokenProvider().get_token()


# ── Endpoint passthrough ──────────────────────────────────────────────────

class _FakeGrokTokens:
    def __init__(self, *, error=None, logged_in=True):
        self._error, self._logged_in = error, logged_in

    def headers(self):
        if self._error is not None:
            raise self._error
        return {"Authorization": "Bearer gtok"}

    def is_logged_in(self):
        return self._logged_in


def _client(monkeypatch, *, tokens=None, post=None, stream=None, models=None):
    if post is not None:
        monkeypatch.setattr(grok_client, "post_json", post)
    if stream is not None:
        monkeypatch.setattr(grok_client, "stream_raw", stream)
    monkeypatch.setattr(grok_client, "list_models", lambda h, **kw: list(models or ["grok-4.3"]))
    app = build_app(Config(), grok_token_provider=tokens or _FakeGrokTokens())
    return TestClient(app)


def test_grok_chat_non_stream_passthrough(monkeypatch):
    completion = {"id": "x", "object": "chat.completion",
                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "yo"},
                               "finish_reason": "stop"}]}

    def fake_post(path, body, *, auth_headers, timeout):
        assert path == "/chat/completions"
        assert body["model"].startswith("grok")
        assert auth_headers["Authorization"] == "Bearer gtok"
        return completion

    client = _client(monkeypatch, post=fake_post)
    r = client.post("/v1/chat/completions", json={
        "model": "grok-4.3", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "yo"


def test_grok_chat_stream_passthrough(monkeypatch):
    def fake_stream(path, body, *, auth_headers, timeout):
        assert path == "/chat/completions"
        yield b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        yield b'data: [DONE]\n\n'

    client = _client(monkeypatch, stream=fake_stream)
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "grok-4.3", "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        body = b"".join(r.iter_bytes()).decode()
    assert '"content":"hel"' in body and '"content":"lo"' in body and "[DONE]" in body


def test_grok_403_maps_to_permission_error(monkeypatch):
    def fake_post(path, body, *, auth_headers, timeout):
        raise grok_client.GrokHTTPError(403, "tier denied")

    client = _client(monkeypatch, post=fake_post)
    r = client.post("/v1/chat/completions", json={
        "model": "grok-4.3", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "subscription_not_entitled"


def test_grok_token_error_returns_401(monkeypatch):
    client = _client(monkeypatch, tokens=_FakeGrokTokens(error=grok_auth.TokenError("login grok")))
    r = client.post("/v1/chat/completions", json={
        "model": "grok-4.3", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "oauth_token_unavailable"


def test_grok_models_listed_live_when_logged_in(monkeypatch):
    client = _client(monkeypatch, tokens=_FakeGrokTokens(logged_in=True), models=["grok-4.3", "grok-4.20-0309-reasoning"])
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert {"grok-4.3", "grok-4.20-0309-reasoning"} <= ids
    assert any(m["owned_by"] == "xai" for m in client.get("/v1/models").json()["data"])
