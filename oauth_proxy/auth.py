"""OAuth-subscription token resolution + Anthropic client construction.

Thin wrapper over the vendored hermes adapter, specialised for the
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

from oauth_proxy._vendor import adapter


class TokenError(RuntimeError):
    """Raised when no usable OAuth subscription token can be resolved."""


class TokenProvider:
    def __init__(self, *, timeout: float = 900.0) -> None:
        raise NotImplementedError("Agent C implements this.")

    def get_token(self) -> str:
        raise NotImplementedError("Agent C implements this.")

    def build_client(self):
        raise NotImplementedError("Agent C implements this.")
