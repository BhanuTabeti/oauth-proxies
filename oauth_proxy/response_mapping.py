"""Anthropic Messages `Message` -> OpenAI ChatCompletion (non-streaming).

CONTRACT (do not change the signature — app.py and tests depend on it):

    anthropic_message_to_openai(message, *, model, completion_id, created,
                                include_reasoning=False, strip_mcp_prefix=True) -> dict

`message` is an Anthropic Messages response as produced by
``anthropic.types.Message.model_dump()`` — a plain dict, e.g.::

    {
      "id": "msg_01...", "type": "message", "role": "assistant",
      "model": "claude-opus-4-7",
      "content": [
        {"type": "thinking", "thinking": "...", "signature": "..."},
        {"type": "text", "text": "Hello"},
        {"type": "tool_use", "id": "toolu_01...", "name": "mcp_get_weather",
         "input": {"city": "SF"}},
      ],
      "stop_reason": "tool_use",        # end_turn|max_tokens|tool_use|stop_sequence|pause_turn|refusal
      "stop_sequence": null,
      "usage": {"input_tokens": 12, "output_tokens": 34,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
    }

Returns an OpenAI ``chat.completion`` dict.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# Anthropic stop_reason -> OpenAI finish_reason. Anything not listed (missing,
# None, or an unknown future value) maps to "stop".
_FINISH_REASON: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

_MCP_PREFIX = "mcp_"


def anthropic_message_to_openai(
    message: Dict[str, Any],
    *,
    model: str,
    completion_id: str,
    created: int,
    include_reasoning: bool = False,
    strip_mcp_prefix: bool = True,
) -> Dict[str, Any]:
    """Convert an Anthropic ``Message.model_dump()`` dict into an OpenAI
    ``chat.completion`` response dict. Pure function; tolerant of malformed
    input (never raises on bad shapes — defaults to empty content)."""
    if not isinstance(message, dict):
        message = {}

    raw_content = message.get("content")
    blocks: List[Dict[str, Any]] = raw_content if isinstance(raw_content, list) else []

    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            if include_reasoning:
                thinking_parts.append(str(block.get("thinking", "")))
        elif btype == "redacted_thinking":
            # No plaintext to surface — intentionally skipped.
            continue
        elif btype == "tool_use":
            tool_calls.append(_to_tool_call(block, strip_mcp_prefix))

    # content: concatenated text; None when no text but tool_use present;
    # "" when neither text nor tool_use.
    if text_parts:
        content: Any = "".join(text_parts)
    elif tool_calls:
        content = None
    else:
        content = ""

    msg: Dict[str, Any] = {"role": "assistant", "content": content}

    if tool_calls:
        msg["tool_calls"] = tool_calls

    if include_reasoning:
        reasoning = "".join(thinking_parts)
        if reasoning:
            msg["reasoning_content"] = reasoning

    # finish_reason: mapped from stop_reason; a tool_use stop always yields
    # "tool_calls" (already in the table, but be explicit if tools are present).
    stop_reason = message.get("stop_reason")
    finish_reason = _FINISH_REASON.get(stop_reason, "stop") if stop_reason else "stop"
    if tool_calls and stop_reason == "tool_use":
        finish_reason = "tool_calls"

    usage = message.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = usage.get("input_tokens", 0) or 0
    completion_tokens = usage.get("output_tokens", 0) or 0

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _to_tool_call(block: Dict[str, Any], strip_mcp_prefix: bool) -> Dict[str, Any]:
    """Build one OpenAI tool_call entry from an Anthropic ``tool_use`` block."""
    name = block.get("name", "")
    if not isinstance(name, str):
        name = str(name)
    if strip_mcp_prefix and name.startswith(_MCP_PREFIX):
        name = name[len(_MCP_PREFIX):]  # strip exactly one leading prefix

    tool_input = block.get("input")
    if tool_input is None:
        arguments = "{}"
    else:
        arguments = json.dumps(tool_input, ensure_ascii=False)

    return {
        "id": block.get("id", ""),
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
