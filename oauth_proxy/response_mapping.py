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

from typing import Any, Dict


def anthropic_message_to_openai(
    message: Dict[str, Any],
    *,
    model: str,
    completion_id: str,
    created: int,
    include_reasoning: bool = False,
    strip_mcp_prefix: bool = True,
) -> Dict[str, Any]:
    raise NotImplementedError("Agent A implements this.")
