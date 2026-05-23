"""OpenAI ChatCompletion request -> Anthropic Messages kwargs.

Thin glue over the vendored adapter's ``build_anthropic_kwargs``. Always runs in
OAuth mode (``is_oauth=True``), targeting native Anthropic (``base_url=None``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from oauth_proxy._vendor import adapter
from oauth_proxy.models import ChatCompletionRequest

_OFF_EFFORTS = {None, "", "off", "none"}


def _looks_like_claude(name: str) -> bool:
    n = (name or "").lower()
    return n.startswith("claude") or n.startswith("anthropic/")


def _resolve_model(requested: str, default_model: str) -> str:
    """Pass Claude model names through; substitute the default for non-Claude
    names (e.g. a client hardcoded to ``gpt-4o``)."""
    return requested if _looks_like_claude(requested) else default_model


def _reasoning_config(
    req: ChatCompletionRequest, default_effort: str
) -> Optional[Dict[str, Any]]:
    effort = req.reasoning_effort or default_effort
    if effort in _OFF_EFFORTS:
        return None
    return {"enabled": True, "effort": str(effort).lower()}


def _tool_choice(req: ChatCompletionRequest) -> Optional[str]:
    tc = req.tool_choice
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        return fn.get("name") or "auto"
    if isinstance(tc, str):
        return tc  # "auto" | "none" | "required"
    return None


def build_kwargs(
    req: ChatCompletionRequest,
    *,
    default_model: str,
    default_reasoning_effort: str,
) -> Dict[str, Any]:
    """Return kwargs ready for ``anthropic.messages.create()``.

    The returned dict includes the resolved (normalized) Claude model under
    ``"model"`` — callers echo that back in the OpenAI response.
    """
    model = _resolve_model(req.model, default_model)
    messages: List[Dict[str, Any]] = [m.model_dump(exclude_none=True) for m in req.messages]

    return adapter.build_anthropic_kwargs(
        model=model,
        messages=messages,
        tools=req.tools,
        max_tokens=req.resolved_max_tokens(),
        reasoning_config=_reasoning_config(req, default_reasoning_effort),
        tool_choice=_tool_choice(req),
        is_oauth=True,
        base_url=None,
    )
