"""Shims for the two hermes ``utils`` helpers the adapter imports at module top.

For the OAuth-subscription path ``base_url`` is ``None`` (native
``api.anthropic.com``), so ``base_url_host_matches`` is effectively unused; we
still implement it faithfully because the adapter imports it at load time and
calls it from third-party-endpoint detection.
"""
from urllib.parse import urlparse


def base_url_host_matches(base_url: str, host: str) -> bool:
    """Return True if ``base_url``'s host equals ``host`` or is a subdomain of it."""
    if not base_url or not host:
        return False
    try:
        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
        netloc_host = (parsed.hostname or "").lower().rstrip(".")
    except Exception:
        return False
    host = host.lower().rstrip(".")
    return netloc_host == host or netloc_host.endswith("." + host)


def normalize_proxy_env_vars() -> None:
    """No-op: httpx/the Anthropic SDK already honor HTTP(S)_PROXY env vars.

    hermes normalizes mixed-case proxy variables here; nothing in this proxy's
    OAuth path depends on that, so the shim does nothing.
    """
    return None
