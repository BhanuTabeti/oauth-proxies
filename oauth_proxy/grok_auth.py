"""Grok (xAI / SuperGrok subscription) OAuth: PKCE login, token store, refresh.

Mirrors the Codex provider but for xAI's subscription OAuth, using the official
public Grok-CLI client identity (the subscription backend only honors it). Key
xAI specifics, all source-verified from the Hermes ``auth.py`` + xAI live OIDC
discovery: endpoints are discovered at runtime; the authorize request needs
``plan=generic`` (else accounts.x.ai rejects a non-allowlisted loopback client);
the token exchange echoes the PKCE challenge; no account-id is needed (xAI
resolves the tenant from the bearer). Inference goes to ``api.x.ai/v1`` with a
plain ``Authorization: Bearer`` — xAI is natively OpenAI-compatible.

CONTRACT (app.py and tests depend on these):

    class TokenError(RuntimeError): ...
    class GrokTokenProvider:
        def __init__(self, *, timeout: float = 900.0) -> None: ...
        def get_token(self) -> str: ...
        def is_logged_in(self) -> bool: ...
        def headers(self) -> Dict[str, str]: ...
    def login(*, open_browser: bool = True, timeout: float = 180.0) -> dict: ...
"""
from __future__ import annotations

import os
import secrets
import time
import urllib.parse
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

from oauth_proxy.oauth_pkce import (
    OAuthLoopbackError,
    capture_redirect,
    generate_pkce,
)

# ── Verified public Grok-CLI OAuth constants (Hermes auth.py + xAI OIDC) ─────
OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
ISSUER = "https://auth.x.ai"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
# Used only if OIDC discovery is unreachable; discovery is preferred at runtime.
_AUTHORIZE_FALLBACK = f"{ISSUER}/oauth2/authorize"
_TOKEN_FALLBACK = f"{ISSUER}/oauth2/token"
SCOPE = "openid profile email offline_access grok-cli:access api:access"
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORTS = (56121, 0)  # preferred port, then any OS-assigned (wildcard loopback)
REDIRECT_PATH = "/callback"
# Non-standard but load-bearing authorize params.
_PLAN = "generic"            # REQUIRED — else accounts.x.ai rejects the loopback client
_REFERRER = "oauth-proxy"    # best-effort attribution

# xAI inference backend (OpenAI-compatible: native /chat/completions + /responses).
BASE_URL = "https://api.x.ai/v1"

_EXPIRY_SKEW_MS = 120_000  # xAI refreshes 120s before expiry


class TokenError(RuntimeError):
    """Raised when no usable Grok OAuth subscription token can be resolved."""


# ── Pure helpers ──────────────────────────────────────────────────────────

def _expires_at_ms(expires_in: Optional[float], *, now_ms: Optional[int] = None) -> Optional[int]:
    if not expires_in:
        return None
    base = now_ms if now_ms is not None else int(time.time() * 1000)
    return base + int(float(expires_in) * 1000)


def _build_authorize_url(
    authorize_endpoint: str, *, redirect_uri: str, code_challenge: str, state: str, nonce: str
) -> str:
    params = {
        "response_type": "code",
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
        "plan": _PLAN,
        "referrer": _REFERRER,
    }
    return authorize_endpoint + "?" + urllib.parse.urlencode(params)


def _record_from_token_response(
    data: Dict, *, prev: Optional[Dict] = None, token_endpoint: Optional[str] = None,
    now_ms: Optional[int] = None,
) -> Dict:
    prev = prev or {}
    return {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token") or prev.get("refresh_token"),
        "id_token": data.get("id_token") or prev.get("id_token"),
        "expires_at": _expires_at_ms(data.get("expires_in"), now_ms=now_ms),
        "token_endpoint": token_endpoint or prev.get("token_endpoint") or _TOKEN_FALLBACK,
        "token_type": data.get("token_type") or "Bearer",
    }


# ── Storage ─────────────────────────────────────────────────────────────────

def _app_home() -> Path:
    return Path(os.environ.get("OAUTH_PROXY_HOME") or (Path.home() / ".oauth-proxy"))


def _store_path() -> Path:
    return _app_home() / ".grok_oauth.json"


def read_credentials() -> Optional[Dict]:
    import json
    try:
        return json.loads(_store_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None


def write_credentials(record: Dict) -> None:
    import json
    home = _app_home()
    home.mkdir(parents=True, exist_ok=True)
    p = _store_path()
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


# ── Token-endpoint I/O ───────────────────────────────────────────────────────

def _discover(*, timeout: float) -> Tuple[str, str]:
    """Resolve (authorize, token) endpoints from OIDC discovery, with fallback."""
    try:
        r = httpx.get(DISCOVERY_URL, headers={"Accept": "application/json"}, timeout=timeout)
        if r.status_code < 400:
            d = r.json()
            return (
                d.get("authorization_endpoint") or _AUTHORIZE_FALLBACK,
                d.get("token_endpoint") or _TOKEN_FALLBACK,
            )
    except Exception:  # pragma: no cover - network/parse failure
        pass
    return _AUTHORIZE_FALLBACK, _TOKEN_FALLBACK


def _post_token(token_endpoint: str, form: Dict[str, str], *, timeout: float) -> Dict:
    resp = httpx.post(
        token_endpoint,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise TokenError(f"xAI token endpoint returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _exchange_code(
    token_endpoint: str, *, code: str, verifier: str, challenge: str, redirect_uri: str, timeout: float
) -> Dict:
    # xAI re-validates the PKCE challenge at the token step, so echo it back.
    return _post_token(
        token_endpoint,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OAUTH_CLIENT_ID,
            "code_verifier": verifier,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        },
        timeout=timeout,
    )


def _refresh(token_endpoint: str, refresh_token: str, *, timeout: float) -> Dict:
    return _post_token(
        token_endpoint,
        {"grant_type": "refresh_token", "client_id": OAUTH_CLIENT_ID, "refresh_token": refresh_token},
        timeout=timeout,
    )


# ── Loopback PKCE login ──────────────────────────────────────────────────────

def login(*, open_browser: bool = True, timeout: float = 180.0) -> Dict:
    """Run the interactive PKCE loopback login against xAI and persist tokens."""
    authorize_endpoint, token_endpoint = _discover(timeout=30.0)
    verifier, challenge = generate_pkce()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(16)

    try:
        code, redirect_uri = capture_redirect(
            redirect_host=REDIRECT_HOST,
            ports=REDIRECT_PORTS,
            path=REDIRECT_PATH,
            build_authorize_url=lambda ru: _build_authorize_url(
                authorize_endpoint, redirect_uri=ru, code_challenge=challenge, state=state, nonce=nonce
            ),
            expected_state=state,
            open_browser=open_browser,
            timeout=timeout,
        )
    except OAuthLoopbackError as exc:
        raise TokenError(str(exc))

    data = _exchange_code(
        token_endpoint, code=code, verifier=verifier, challenge=challenge,
        redirect_uri=redirect_uri, timeout=timeout,
    )
    record = _record_from_token_response(data, token_endpoint=token_endpoint)
    if not record.get("access_token"):
        raise TokenError("token exchange did not return an access_token")
    write_credentials(record)
    return record


# ── Token provider ────────────────────────────────────────────────────────────

class GrokTokenProvider:
    def __init__(self, *, timeout: float = 900.0) -> None:
        self.timeout = timeout
        self._record: Optional[Dict] = None

    def _fresh(self, record: Optional[Dict]) -> bool:
        if not record or not record.get("access_token"):
            return False
        exp = record.get("expires_at")
        if exp is None:
            return False
        return int(time.time() * 1000) < (int(exp) - _EXPIRY_SKEW_MS)

    def get_token(self) -> str:
        if self._fresh(self._record):
            return self._record["access_token"]  # type: ignore[index]
        record = self._record or read_credentials()
        if not record:
            raise TokenError(
                "No Grok OAuth token found. Run `oauth-proxy login grok` to "
                "authorize with your SuperGrok subscription."
            )
        if self._fresh(record):
            self._record = record
            return record["access_token"]
        refresh_token = record.get("refresh_token")
        if not refresh_token:
            raise TokenError(
                "Grok OAuth token is expired and no refresh token is stored. "
                "Run `oauth-proxy login grok` again."
            )
        refreshed = _record_from_token_response(
            _refresh(record.get("token_endpoint") or _TOKEN_FALLBACK, refresh_token, timeout=self.timeout),
            prev=record,
            token_endpoint=record.get("token_endpoint"),
        )
        if not refreshed.get("access_token"):
            raise TokenError("Grok token refresh failed. Run `oauth-proxy login grok` again.")
        write_credentials(refreshed)
        self._record = refreshed
        return refreshed["access_token"]

    def is_logged_in(self) -> bool:
        record = self._record or read_credentials()
        return bool(record and record.get("access_token"))

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}
