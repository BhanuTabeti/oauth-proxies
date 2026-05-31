"""Unit tests for Codex OAuth: pure helpers + token provider (mocked I/O)."""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.parse

import pytest

from oauth_proxy import codex_auth


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _make_id_token(account_id: str | None) -> str:
    auth = {"chatgpt_account_id": account_id} if account_id is not None else {}
    payload = {"https://api.openai.com/auth": auth, "sub": "user_1"}
    seg = _b64url(json.dumps(payload).encode("utf-8"))
    return f"{_b64url(b'{}')}.{seg}.sig"


# ── PKCE ─────────────────────────────────────────────────────────────────────

def test_pkce_challenge_is_sha256_of_verifier():
    verifier, challenge = codex_auth._generate_pkce()
    assert 43 <= len(verifier) <= 128
    expected = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    assert challenge == expected
    assert "=" not in verifier and "=" not in challenge  # base64url, unpadded


def test_pkce_pairs_are_unique():
    assert codex_auth._generate_pkce()[0] != codex_auth._generate_pkce()[0]


# ── Authorize URL ─────────────────────────────────────────────────────────────

def test_authorize_url_has_exact_codex_params():
    url = codex_auth._build_authorize_url(
        redirect_uri="http://localhost:1455/auth/callback",
        code_challenge="CHAL",
        state="STATE",
    )
    base, _, qs = url.partition("?")
    assert base == "https://auth.openai.com/oauth/authorize"
    params = dict(urllib.parse.parse_qsl(qs))
    assert params["response_type"] == "code"
    assert params["client_id"] == "app_EMoamEEZ73f0CkXaXp7hrann"
    assert params["redirect_uri"] == "http://localhost:1455/auth/callback"
    assert params["code_challenge"] == "CHAL"
    assert params["code_challenge_method"] == "S256"
    assert params["state"] == "STATE"
    assert params["id_token_add_organizations"] == "true"
    assert params["codex_cli_simplified_flow"] == "true"
    assert params["originator"] == "codex_cli_rs"
    assert "offline_access" in params["scope"]


# ── JWT / account id ──────────────────────────────────────────────────────────

def test_account_id_extracted_from_id_token():
    assert codex_auth._account_id_from_id_token(_make_id_token("acc_42")) == "acc_42"


@pytest.mark.parametrize("bad", [None, "", "not-a-jwt", "a.b", _make_id_token(None)])
def test_account_id_missing_or_garbage_returns_none(bad):
    assert codex_auth._account_id_from_id_token(bad) is None


# ── Token-response mapping ─────────────────────────────────────────────────────

def test_record_from_token_response_maps_and_computes_expiry():
    rec = codex_auth._record_from_token_response(
        {
            "access_token": "at1",
            "refresh_token": "rt1",
            "id_token": _make_id_token("acc_7"),
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        now_ms=1_000_000,
    )
    assert rec["access_token"] == "at1"
    assert rec["refresh_token"] == "rt1"
    assert rec["account_id"] == "acc_7"
    assert rec["expires_at"] == 1_000_000 + 3600 * 1000


def test_record_from_refresh_carries_forward_missing_fields():
    prev = {"refresh_token": "rt_old", "id_token": _make_id_token("acc_old"), "account_id": "acc_old"}
    rec = codex_auth._record_from_token_response(
        {"access_token": "at2", "expires_in": 60}, prev=prev, now_ms=0
    )
    assert rec["access_token"] == "at2"
    assert rec["refresh_token"] == "rt_old"   # reused
    assert rec["account_id"] == "acc_old"     # reused
    assert rec["expires_at"] == 60_000


# ── Storage round-trip ──────────────────────────────────────────────────────

def test_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    assert codex_auth.read_credentials() is None
    codex_auth.write_credentials({"access_token": "x", "account_id": "acc"})
    assert codex_auth.read_credentials() == {"access_token": "x", "account_id": "acc"}
    # 0600 perms
    mode = (codex_auth._store_path().stat().st_mode) & 0o777
    assert mode == 0o600


# ── Token provider ────────────────────────────────────────────────────────────

def _future_ms(seconds: int = 3600) -> int:
    import time
    return int(time.time() * 1000) + seconds * 1000


def test_get_token_serves_fresh_stored_token(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    codex_auth.write_credentials(
        {"access_token": "live", "refresh_token": "r", "expires_at": _future_ms()}
    )
    tp = codex_auth.CodexTokenProvider()
    assert tp.get_token() == "live"


def test_get_token_refreshes_when_expired(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    codex_auth.write_credentials(
        {"access_token": "stale", "refresh_token": "r1", "expires_at": 1}  # long expired
    )

    def fake_refresh(refresh_token, *, timeout):
        assert refresh_token == "r1"
        return {"access_token": "new", "refresh_token": "r2", "expires_in": 3600}

    monkeypatch.setattr(codex_auth, "_refresh", fake_refresh)
    tp = codex_auth.CodexTokenProvider()
    assert tp.get_token() == "new"
    # persisted
    assert codex_auth.read_credentials()["access_token"] == "new"


def test_get_token_no_creds_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    tp = codex_auth.CodexTokenProvider()
    with pytest.raises(codex_auth.TokenError, match="login codex"):
        tp.get_token()


def test_get_token_expired_no_refresh_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    codex_auth.write_credentials({"access_token": "stale", "expires_at": 1})
    tp = codex_auth.CodexTokenProvider()
    with pytest.raises(codex_auth.TokenError, match="login codex"):
        tp.get_token()


def test_headers_include_account_and_originator(tmp_path, monkeypatch):
    monkeypatch.setenv("OAUTH_PROXY_HOME", str(tmp_path))
    codex_auth.write_credentials(
        {"access_token": "tok", "refresh_token": "r", "account_id": "acc_9", "expires_at": _future_ms()}
    )
    h = codex_auth.CodexTokenProvider().headers()
    assert h["Authorization"] == "Bearer tok"
    assert h["ChatGPT-Account-ID"] == "acc_9"
    assert h["originator"] == "codex_cli_rs"
