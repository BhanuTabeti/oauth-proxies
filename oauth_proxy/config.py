"""Runtime configuration, loaded from environment variables.

All knobs are optional; defaults target a single-user, localhost deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8787
    # Optional shared secret. When set, clients must send
    # ``Authorization: Bearer <proxy_api_key>``. When None, the server is open
    # (intended for localhost-only use).
    proxy_api_key: Optional[str] = None
    # Fallback Claude model when a client requests a non-Claude model name
    # (e.g. "gpt-4o"). When None and a non-Claude model is requested, the name
    # is passed through unchanged (the adapter will normalize it).
    default_model: str = "claude-opus-4-7"
    # Default reasoning effort: one of {off, low, medium, high, xhigh, max}.
    # "off" disables extended thinking unless the client explicitly requests it.
    default_reasoning_effort: str = "off"
    # Surface Claude "thinking" text in responses as a non-standard
    # ``reasoning_content`` field (OpenAI clients ignore unknown fields).
    include_reasoning: bool = False
    # Read timeout (seconds) for upstream Anthropic calls.
    request_timeout_seconds: float = 900.0


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    """Build a Config from environment variables."""
    return Config(
        host=os.environ.get("PROXY_HOST", "127.0.0.1"),
        port=int(os.environ.get("PROXY_PORT", "8787")),
        proxy_api_key=(os.environ.get("PROXY_API_KEY") or None),
        default_model=os.environ.get("DEFAULT_MODEL", "claude-opus-4-7"),
        default_reasoning_effort=os.environ.get("DEFAULT_REASONING_EFFORT", "off").strip().lower(),
        include_reasoning=_get_bool("PROXY_INCLUDE_REASONING", False),
        request_timeout_seconds=float(os.environ.get("PROXY_REQUEST_TIMEOUT", "900")),
    )
