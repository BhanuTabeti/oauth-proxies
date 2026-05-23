"""Tests for oauth_proxy.auth.TokenProvider / TokenError.

All adapter access is monkeypatched on ``oauth_proxy.auth.adapter`` so no real
network / keychain / files are touched. Time-based behaviour is exercised by
monkeypatching the adapter helpers (``is_claude_code_token_valid``) for
determinism rather than fighting the wall clock.
"""
from __future__ import annotations

import time

import pytest

from oauth_proxy import auth
from oauth_proxy.auth import TokenError, TokenProvider


# An OAuth-shaped subscription access token (sk-ant-oat*) and a JWT both pass
# adapter._is_oauth_token; a plain API key (sk-ant-api*) does not.
OAUTH_TOKEN = "sk-ant-oat-abc123"
API_KEY = "sk-ant-api-plainkey"
FAR_FUTURE_MS = int(time.time() * 1000) + 10 * 60 * 1000  # +10 minutes


def _install_adapter_stub(monkeypatch, **overrides):
    """Replace every adapter fn used by auth with a controllable stub.

    Returns a ``calls`` dict counting invocations so tests can assert caching
    behaviour. Each adapter fn defaults to a "nothing found" return value;
    pass overrides to set specific return values (or callables).
    """
    calls = {
        "read_claude_code_credentials": 0,
        "is_claude_code_token_valid": 0,
        "_refresh_oauth_token": 0,
        "resolve_anthropic_token": 0,
        "_is_oauth_token": 0,
    }

    def make(name, default):
        def fn(*args, **kwargs):
            calls[name] += 1
            val = overrides.get(name, default)
            if callable(val):
                return val(*args, **kwargs)
            return val
        return fn

    monkeypatch.setattr(
        auth.adapter, "read_claude_code_credentials",
        make("read_claude_code_credentials", None),
    )
    monkeypatch.setattr(
        auth.adapter, "is_claude_code_token_valid",
        make("is_claude_code_token_valid", False),
    )
    monkeypatch.setattr(
        auth.adapter, "_refresh_oauth_token",
        make("_refresh_oauth_token", None),
    )
    monkeypatch.setattr(
        auth.adapter, "resolve_anthropic_token",
        make("resolve_anthropic_token", None),
    )
    # Real _is_oauth_token logic is cheap & deterministic; use it directly but
    # count calls.
    monkeypatch.setattr(
        auth.adapter, "_is_oauth_token",
        make("_is_oauth_token", lambda key: bool(key) and not key.startswith("sk-ant-api")),
    )
    return calls


def test_valid_cred_file_token_returned(monkeypatch):
    creds = {"accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": FAR_FUTURE_MS}
    _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=True,
    )
    tp = TokenProvider()
    assert tp.get_token() == OAUTH_TOKEN


def test_expired_creds_refresh_success(monkeypatch):
    creds = {"accessToken": "old", "refreshToken": "r", "expiresAt": 1}
    refreshed = "sk-ant-oat-refreshed"
    _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=False,  # expired -> go to refresh
        _refresh_oauth_token=refreshed,
    )
    tp = TokenProvider()
    assert tp.get_token() == refreshed


def test_expired_creds_refresh_fails_then_env_fallback(monkeypatch):
    creds = {"accessToken": "old", "refreshToken": "r", "expiresAt": 1}
    env_token = "sk-ant-oat-fromenv"
    _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=False,
        _refresh_oauth_token=None,  # refresh failed
        resolve_anthropic_token=env_token,
    )
    tp = TokenProvider()
    assert tp.get_token() == env_token


def test_no_creds_env_oauth_token_returned(monkeypatch):
    env_token = "eyJhbGciOiJ"  # JWT-shaped OAuth token
    calls = _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=None,
        resolve_anthropic_token=env_token,
    )
    tp = TokenProvider()
    assert tp.get_token() == env_token
    # refresh should never be attempted when there are no creds
    assert calls["_refresh_oauth_token"] == 0


def test_resolved_token_is_plain_api_key_raises(monkeypatch):
    _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=None,
        resolve_anthropic_token=API_KEY,  # plain API key, not OAuth
    )
    tp = TokenProvider()
    with pytest.raises(TokenError) as excinfo:
        tp.get_token()
    msg = str(excinfo.value).lower()
    assert "api key" in msg or "oauth" in msg


def test_nothing_resolvable_raises_setup_token(monkeypatch):
    _install_adapter_stub(monkeypatch)  # everything returns None / False
    tp = TokenProvider()
    with pytest.raises(TokenError) as excinfo:
        tp.get_token()
    assert "setup-token" in str(excinfo.value)


def test_caching_avoids_second_credential_read(monkeypatch):
    creds = {"accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": FAR_FUTURE_MS}
    calls = _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=True,
    )
    tp = TokenProvider()
    assert tp.get_token() == OAUTH_TOKEN
    assert tp.get_token() == OAUTH_TOKEN
    # Resolution (and thus credential read) ran only once while still valid.
    assert calls["read_claude_code_credentials"] == 1


def test_cache_re_resolves_after_expiry(monkeypatch):
    """A cached token within ~60s of expiry forces re-resolution next call."""
    now_ms = int(time.time() * 1000)
    creds = {
        "accessToken": OAUTH_TOKEN,
        "refreshToken": "r",
        # expires very soon -> inside the ~60s skew window -> not cacheable
        "expiresAt": now_ms + 5_000,
    }
    calls = _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=True,
    )
    tp = TokenProvider()
    assert tp.get_token() == OAUTH_TOKEN
    assert tp.get_token() == OAUTH_TOKEN
    # Because the cached token sits inside the expiry skew window, each call
    # re-runs resolution rather than serving a stale token.
    assert calls["read_claude_code_credentials"] == 2


def test_build_client_passes_token_and_timeout(monkeypatch):
    creds = {"accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": FAR_FUTURE_MS}
    _install_adapter_stub(
        monkeypatch,
        read_claude_code_credentials=creds,
        is_claude_code_token_valid=True,
    )

    sentinel = object()
    recorded = {}

    def fake_build(api_key, base_url=None, timeout=None, **kwargs):
        recorded["api_key"] = api_key
        recorded["base_url"] = base_url
        recorded["timeout"] = timeout
        recorded["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(auth.adapter, "build_anthropic_client", fake_build)

    tp = TokenProvider(timeout=123.0)
    client = tp.build_client()

    assert client is sentinel
    assert recorded["api_key"] == OAUTH_TOKEN
    assert recorded["timeout"] == 123.0
    assert recorded["base_url"] is None
