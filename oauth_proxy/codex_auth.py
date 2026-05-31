"""Codex (ChatGPT subscription) OAuth: PKCE login, token store, refresh.

This proxy runs its OWN OAuth flow against OpenAI's auth server using the
*official public Codex CLI client identity* — the ChatGPT subscription backend
only honors that client, so we reuse it (we do not, and cannot, register a
private app). We do NOT read the Codex CLI's stored credentials; this module
mints and persists its own token bundle under ``~/.oauth-proxy/.codex_oauth.json``.

All constants below were read from the official ``openai/codex`` Rust source
(``codex-rs/login`` + ``codex-rs/model-provider-info``) and adversarially
verified; see the design spec for provenance. They are public client values,
not secrets.

CONTRACT (app.py and tests depend on these):

    class TokenError(RuntimeError): ...

    class CodexTokenProvider:
        def __init__(self, *, timeout: float = 900.0) -> None: ...
        def get_token(self) -> str: ...        # resolves + refreshes; raises TokenError
        def account_id(self) -> Optional[str]: ...
        def headers(self) -> Dict[str, str]: ...  # Authorization + ChatGPT-Account-ID + originator

    def login(*, open_browser: bool = True, timeout: float = 180.0) -> dict: ...

Pure helpers (``_generate_pkce``, ``_build_authorize_url``,
``_account_id_from_id_token``, ``_decode_jwt_segment``) take no I/O and are unit
tested directly.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

import httpx

# ── Verified public Codex CLI OAuth constants (openai/codex, main) ──────────
# codex-rs/login/src/auth/manager.rs: CLIENT_ID
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
# codex-rs/login/src/server.rs: DEFAULT_ISSUER
_ISSUER = "https://auth.openai.com"
AUTHORIZE_ENDPOINT = f"{_ISSUER}/oauth/authorize"
TOKEN_ENDPOINT = f"{_ISSUER}/oauth/token"
# codex-rs/login/src/server.rs: DEFAULT_PORT (1455) + FALLBACK_PORT (1457),
# redirect path "/auth/callback". Both ports are in OpenAI's redirect allowlist.
_REDIRECT_PORTS = (1455, 1457)
_REDIRECT_PATH = "/auth/callback"
# codex-rs/login/src/server.rs authorize query: scope + flow flags.
_SCOPE = "openid profile email offline_access api.connectors.read api.connectors.invoke"
# codex-rs/login/src/auth/default_client.rs: DEFAULT_ORIGINATOR
ORIGINATOR = "codex_cli_rs"

# ── Verified Codex Responses backend (ChatGPT-subscription mode) ────────────
# codex-rs/model-provider-info: CHATGPT_CODEX_BASE_URL + "/responses".
RESPONSES_BASE_URL = "https://chatgpt.com/backend-api/codex"
RESPONSES_ENDPOINT = f"{RESPONSES_BASE_URL}/responses"

# id_token JWT claim namespace -> chatgpt_account_id (codex-rs/login/token_data.rs)
_AUTH_CLAIM = "https://api.openai.com/auth"

# Re-resolve a cached token this many ms before its known expiry.
_EXPIRY_SKEW_MS = 60_000


class TokenError(RuntimeError):
    """Raised when no usable Codex OAuth subscription token can be resolved."""


# ── Pure helpers (no I/O — unit tested directly) ────────────────────────────

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _generate_pkce() -> Tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE S256."""
    verifier = _b64url(os.urandom(64))[:128]
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _build_authorize_url(*, redirect_uri: str, code_challenge: str, state: str) -> str:
    """Build the OpenAI authorize URL with the exact Codex CLI query params."""
    params = {
        "response_type": "code",
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": _SCOPE,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": ORIGINATOR,
    }
    return AUTHORIZE_ENDPOINT + "?" + urllib.parse.urlencode(params)


def _decode_jwt_segment(token: str) -> Dict:
    """Decode (without verifying) the payload segment of a JWT into a dict."""
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("not a JWT (missing payload segment)")
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)  # restore base64 padding
    return json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))


def _account_id_from_id_token(id_token: Optional[str]) -> Optional[str]:
    """Extract ``chatgpt_account_id`` from the id_token's auth claim, if present."""
    if not id_token:
        return None
    try:
        claims = _decode_jwt_segment(id_token)
    except (ValueError, json.JSONDecodeError):
        return None
    auth = claims.get(_AUTH_CLAIM)
    if isinstance(auth, dict):
        acc = auth.get("chatgpt_account_id")
        return acc if isinstance(acc, str) and acc else None
    return None


def _expires_at_ms(expires_in: Optional[float], *, now_ms: Optional[int] = None) -> Optional[int]:
    """Convert an OAuth ``expires_in`` (seconds) into an absolute epoch-ms expiry."""
    if not expires_in:
        return None
    base = now_ms if now_ms is not None else int(time.time() * 1000)
    return base + int(float(expires_in) * 1000)


def _record_from_token_response(
    data: Dict, *, prev: Optional[Dict] = None, now_ms: Optional[int] = None
) -> Dict:
    """Build a stored credential record from a token-endpoint JSON response.

    Refresh responses may omit ``refresh_token`` (reuse the old one) and
    ``id_token`` (reuse the old account id). ``prev`` carries those forward.
    """
    prev = prev or {}
    id_token = data.get("id_token") or prev.get("id_token")
    refresh_token = data.get("refresh_token") or prev.get("refresh_token")
    account_id = _account_id_from_id_token(id_token) or prev.get("account_id")
    return {
        "access_token": data.get("access_token"),
        "refresh_token": refresh_token,
        "id_token": id_token,
        "account_id": account_id,
        "expires_at": _expires_at_ms(data.get("expires_in"), now_ms=now_ms),
        "token_type": data.get("token_type") or "Bearer",
    }


# ── Storage ─────────────────────────────────────────────────────────────────

def _app_home() -> Path:
    return Path(os.environ.get("OAUTH_PROXY_HOME") or (Path.home() / ".oauth-proxy"))


def _store_path() -> Path:
    return _app_home() / ".codex_oauth.json"


def read_credentials() -> Optional[Dict]:
    """Read the persisted Codex OAuth record, or None if absent/unreadable."""
    p = _store_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def write_credentials(record: Dict) -> None:
    """Persist the Codex OAuth record with owner-only permissions (0600)."""
    home = _app_home()
    home.mkdir(parents=True, exist_ok=True)
    p = _store_path()
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


# ── Token-endpoint I/O ───────────────────────────────────────────────────────

def _post_token(form: Dict[str, str], *, timeout: float) -> Dict:
    resp = httpx.post(
        TOKEN_ENDPOINT,
        data=form,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        timeout=timeout,
    )
    if resp.status_code >= 400:
        raise TokenError(f"token endpoint returned {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _exchange_code(*, code: str, verifier: str, redirect_uri: str, timeout: float) -> Dict:
    return _post_token(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OAUTH_CLIENT_ID,
            "code_verifier": verifier,
        },
        timeout=timeout,
    )


def _refresh(refresh_token: str, *, timeout: float) -> Dict:
    return _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": OAUTH_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        timeout=timeout,
    )


# ── Loopback PKCE login ──────────────────────────────────────────────────────

class _CallbackHandler(BaseHTTPRequestHandler):
    # Set by login() before serving.
    captured: Dict[str, str] = {}

    def do_GET(self) -> None:  # noqa: N802 (stdlib name)
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != _REDIRECT_PATH:
            self.send_response(404)
            self.end_headers()
            return
        query = urllib.parse.parse_qs(parsed.query)
        type(self).captured = {k: v[0] for k, v in query.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h2>Codex login complete.</h2>"
            b"<p>You can close this tab and return to the terminal.</p></body></html>"
        )

    def log_message(self, *args) -> None:  # silence stdlib request logging
        return


def _bind_callback_server() -> Tuple[HTTPServer, str]:
    """Bind the loopback callback server, trying each allowlisted port in turn."""
    last_err: Optional[OSError] = None
    for port in _REDIRECT_PORTS:
        try:
            server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        except OSError as exc:  # port busy
            last_err = exc
            continue
        return server, f"http://localhost:{port}{_REDIRECT_PATH}"
    raise TokenError(
        f"could not bind the OAuth callback server on ports {_REDIRECT_PORTS}: {last_err}"
    )


def login(*, open_browser: bool = True, timeout: float = 180.0) -> Dict:
    """Run the interactive PKCE loopback login and persist the token bundle.

    Opens the browser to OpenAI's consent screen, captures the redirect on a
    local loopback server, exchanges the code, and writes the credential record.
    Returns the stored record. Raises ``TokenError`` on failure/timeout.
    """
    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(32)
    server, redirect_uri = _bind_callback_server()
    _CallbackHandler.captured = {}

    url = _build_authorize_url(redirect_uri=redirect_uri, code_challenge=challenge, state=state)
    print(f"Opening browser for Codex (ChatGPT) login:\n  {url}\n")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - headless/no browser
            print("(could not open a browser automatically — open the URL above manually)")

    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    thread.join(timeout)
    server.server_close()

    captured = _CallbackHandler.captured
    if not captured:
        raise TokenError("login timed out waiting for the OAuth redirect")
    if captured.get("state") != state:
        raise TokenError("OAuth state mismatch (possible CSRF) — aborting")
    if "error" in captured:
        raise TokenError(f"authorization failed: {captured.get('error')}")
    code = captured.get("code")
    if not code:
        raise TokenError("no authorization code in the OAuth redirect")

    token_resp = _exchange_code(
        code=code, verifier=verifier, redirect_uri=redirect_uri, timeout=timeout
    )
    record = _record_from_token_response(token_resp)
    if not record.get("access_token"):
        raise TokenError("token exchange did not return an access_token")
    write_credentials(record)
    return record


# ── Token provider (resolve + refresh + cache) ───────────────────────────────

class CodexTokenProvider:
    def __init__(self, *, timeout: float = 900.0) -> None:
        self.timeout = timeout
        self._record: Optional[Dict] = None

    def _fresh(self, record: Optional[Dict]) -> bool:
        if not record or not record.get("access_token"):
            return False
        exp = record.get("expires_at")
        if exp is None:
            return False  # unknown expiry — re-resolve to be safe
        return int(time.time() * 1000) < (int(exp) - _EXPIRY_SKEW_MS)

    def get_token(self) -> str:
        """Resolve a valid Codex access token, refreshing if needed.

        Raises ``TokenError`` (advising ``oauth-proxy login codex``) when no
        usable credential exists.
        """
        if self._fresh(self._record):
            return self._record["access_token"]  # type: ignore[index]

        record = self._record or read_credentials()
        if not record:
            raise TokenError(
                "No Codex OAuth token found. Run `oauth-proxy login codex` to "
                "authorize with your ChatGPT subscription."
            )

        if self._fresh(record):
            self._record = record
            return record["access_token"]

        refresh_token = record.get("refresh_token")
        if not refresh_token:
            raise TokenError(
                "Codex OAuth token is expired and no refresh token is stored. "
                "Run `oauth-proxy login codex` again."
            )
        refreshed = _record_from_token_response(
            _refresh(refresh_token, timeout=self.timeout), prev=record
        )
        if not refreshed.get("access_token"):
            raise TokenError(
                "Codex token refresh failed. Run `oauth-proxy login codex` again."
            )
        write_credentials(refreshed)
        self._record = refreshed
        return refreshed["access_token"]

    def account_id(self) -> Optional[str]:
        record = self._record or read_credentials() or {}
        return record.get("account_id") or _account_id_from_id_token(record.get("id_token"))

    def headers(self) -> Dict[str, str]:
        """Build the request headers for the Codex Responses backend."""
        h = {
            "Authorization": f"Bearer {self.get_token()}",
            "originator": ORIGINATOR,
        }
        acc = self.account_id()
        if acc:
            h["ChatGPT-Account-ID"] = acc
        return h
