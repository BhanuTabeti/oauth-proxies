"""OpenAI-compatible request schemas (pydantic) and the served model catalog.

Responses are emitted as plain dicts by the converter modules
(``response_mapping`` / ``stream_mapping``), so only *requests* are modeled
here. Request models are intentionally permissive (``extra="allow"``) so that
unknown OpenAI fields don't 422 a client.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class ChatMessage(_Lenient):
    role: str
    # content may be a string, a list of content parts (multimodal), or null
    # (assistant tool-call messages).
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    # assistant tool calls
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # tool result messages
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(_Lenient):
    model: str
    messages: List[ChatMessage]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # newer OpenAI alias
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    # OpenAI o-series reasoning control; mapped to Anthropic thinking effort.
    reasoning_effort: Optional[str] = None

    def resolved_max_tokens(self) -> Optional[int]:
        return self.max_tokens if self.max_tokens is not None else self.max_completion_tokens

    def wants_usage(self) -> bool:
        opts = self.stream_options or {}
        return bool(opts.get("include_usage"))


# ── Served model catalog ──────────────────────────────────────────────────
# Surfaced at GET /v1/models. Grouped by the subscription backend that serves
# them; the proxy routes a request to a backend by the model-name prefix.
KNOWN_MODELS: List[str] = [
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-7-sonnet",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
]

# Codex (ChatGPT subscription) models. This is only a FALLBACK — /v1/models
# fetches the real allowlist live from the backend per logged-in account. The
# accepted set is account/client-specific and drifts over time.
CODEX_MODELS: List[str] = [
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.3-codex",
    "gpt-5.2",
]

# Grok (SuperGrok subscription) models, current as of 2026-05; retired slugs
# (grok-4, grok-code-fast, ...) are remapped to grok-4.3 server-side by xAI.
GROK_MODELS: List[str] = [
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
]

# provider id -> (OpenAI ``owned_by`` label, curated model ids)
_CATALOG = {
    "anthropic": ("anthropic", KNOWN_MODELS),
    "codex": ("openai", CODEX_MODELS),
    "grok": ("xai", GROK_MODELS),
}


def model_catalog(
    available: Optional[Any] = None,
    live: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Return the OpenAI ``GET /v1/models`` payload.

    ``available`` is an iterable of logged-in provider ids
    (``{"anthropic", "codex", "grok"}``); only those providers' models are
    listed, so the catalog reflects which subscriptions you can actually use.
    When ``None``, every provider is listed (back-compat default).

    ``live`` maps a provider id to model ids fetched live from that provider's
    backend; when present for a provider, it replaces the curated fallback so
    the catalog reflects the upstream's real allowlist.
    """
    selected = set(_CATALOG) if available is None else set(available)
    live = live or {}
    created = int(time.time())
    data = []
    for pid, (owner, curated) in _CATALOG.items():
        if pid not in selected:
            continue
        for m in live.get(pid) or curated:
            data.append({"id": m, "object": "model", "created": created, "owned_by": owner})
    return {"object": "list", "data": data}
