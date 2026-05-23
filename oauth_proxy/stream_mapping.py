"""Anthropic streaming events -> OpenAI ``chat.completion.chunk`` (streaming).

CONTRACT (do not change the signature — app.py and tests depend on it):

    anthropic_events_to_openai_chunks(events, *, model, completion_id, created,
                                      include_reasoning=False,
                                      strip_mcp_prefix=True,
                                      include_usage=False) -> Iterator[dict]

`events` is an iterable of Anthropic raw streaming events, each as
``event.model_dump()`` (plain dicts). Event types and shapes::

    {"type": "message_start", "message": {"id": "msg_..", "usage": {"input_tokens": 12, "output_tokens": 1}}}
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_..", "name": "mcp_calc", "input": {}}}
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"x\":"}}
    {"type": "content_block_delta", "index": 2, "delta": {"type": "thinking_delta", "thinking": ".."}}
    {"type": "content_block_delta", "index": 2, "delta": {"type": "signature_delta", "signature": ".."}}
    {"type": "content_block_stop", "index": 0}
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null}, "usage": {"output_tokens": 34}}
    {"type": "message_stop"}

Yields OpenAI ``chat.completion.chunk`` dicts. The first yielded chunk carries
``delta={"role": "assistant"}``; the final yielded chunk carries the mapped
``finish_reason``. Do NOT yield the terminal ``"[DONE]"`` sentinel — app.py
appends it to the SSE stream.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator


def anthropic_events_to_openai_chunks(
    events: Iterable[Dict[str, Any]],
    *,
    model: str,
    completion_id: str,
    created: int,
    include_reasoning: bool = False,
    strip_mcp_prefix: bool = True,
    include_usage: bool = False,
) -> Iterator[Dict[str, Any]]:
    raise NotImplementedError("Agent B implements this.")
