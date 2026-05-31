"""OAuth-subscription token resolution + Anthropic client construction.

Thin wrapper over the vendored Anthropic-Messages adapter, specialised for the
OAuth-subscription path (the only auth mode this proxy supports).

CONTRACT (do not change these signatures — app.py and tests depend on them):

    class TokenError(RuntimeError): ...

    class TokenProvider:
        def __init__(self, *, timeout: float = 900.0) -> None: ...
        def get_token(self) -> str: ...        # resolves + refreshes; raises TokenError
        def build_client(self): ...            # returns anthropic.Anthropic for OAuth

All access to the vendored adapter goes through ``from oauth_proxy._vendor
import adapter``. Tests monkeypatch ``adapter.<fn>`` to avoid network/keychain.
"""
from __future__ import annotations

import time
from typing import Optional

from oauth_proxy._vendor import adapter

# Re-resolve when a cached token is within this many ms of its known expiry.
# Mirrors the 60s skew the adapter's own validity check uses.
_EXPIRY_SKEW_MS = 60_000


class TokenError(RuntimeError):
    """Raised when no usable OAuth subscription token can be resolved."""


class TokenProvider:
    def __init__(self, *, timeout: float = 900.0) -> None:
        self.timeout = timeout
        # In-process cache so repeated calls within the validity window don't
        # re-hit the keychain / trigger a refresh on every request.
        self._token: Optional[str] = None
        self._expires_at_ms: Optional[int] = None

    def _cache_is_fresh(self) -> bool:
        """True if the cached token can still be served without re-resolving."""
        if self._token is None:
            return False
        # Tokens with no known expiry (e.g. env-resolved) are not cached as
        # "fresh"; we re-resolve on the next call. This keeps caching behaviour
        # deterministic and explicit.
        if self._expires_at_ms is None:
            return False
        now_ms = int(time.time() * 1000)
        return now_ms < (self._expires_at_ms - _EXPIRY_SKEW_MS)

    def get_token(self) -> str:
        """Resolve (and refresh if needed) a valid OAuth subscription token.

        Caches the resolved token in-process; serves the cached value while it
        remains comfortably inside its validity window. Raises ``TokenError``
        when nothing usable can be resolved or the resolved credential is a
        plain API key.
        """
        if self._cache_is_fresh():
            return self._token  # type: ignore[return-value]

        candidate: Optional[str] = None
        candidate_expires_at_ms: Optional[int] = None

        creds = adapter.read_claude_code_credentials()
        had_expired_creds = False
        if creds and adapter.is_claude_code_token_valid(creds):
            candidate = creds.get("accessToken")
            candidate_expires_at_ms = creds.get("expiresAt") or None
        elif creds:
            # Present but expired/invalid — try a refresh (writes back to disk).
            had_expired_creds = True
            candidate = adapter._refresh_oauth_token(creds)
            # A refresh rotates the token; we don't know the new expiry here,
            # so leave it unknown and let the next call re-validate.

        if not candidate:
            # Env-var / credential-file fallback resolver.
            candidate = adapter.resolve_anthropic_token()

        if not candidate:
            if had_expired_creds:
                # Distinguish the common case (Claude Code IS installed, but its
                # persisted token expired and the stored refresh token is stale)
                # from "no credentials at all" — the generic message misled here.
                raise TokenError(
                    "Found a Claude Code OAuth token in the credential store, but it "
                    "is expired and automatic refresh failed (the stored refresh "
                    "token is likely stale). Mint a fresh one with `claude "
                    "setup-token` and set CLAUDE_CODE_OAUTH_TOKEN, or re-login with "
                    "the `claude` CLI."
                )
            raise TokenError(
                "No Claude Code OAuth subscription token found. Log in with the "
                "`claude` CLI and run `claude setup-token`, or set "
                "CLAUDE_CODE_OAUTH_TOKEN."
            )

        if not adapter._is_oauth_token(candidate):
            raise TokenError(
                "Resolved credential looks like a plain Anthropic API key, but "
                "this proxy is OAuth-only and requires a Claude Code OAuth "
                "subscription token. Run `claude setup-token` or set "
                "CLAUDE_CODE_OAUTH_TOKEN."
            )

        self._token = candidate
        self._expires_at_ms = candidate_expires_at_ms
        return candidate

    def build_client(self):
        return adapter.build_anthropic_client(
            self.get_token(), base_url=None, timeout=self.timeout
        )
