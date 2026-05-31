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

_MCP_TOOL_PREFIX = "mcp_"

# Anthropic ``message_delta.delta.stop_reason`` -> OpenAI ``finish_reason``.
_FINISH_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _map_finish_reason(stop_reason: Any) -> str:
    """Map an Anthropic stop_reason to an OpenAI finish_reason.

    Missing / ``None`` / unknown values fall back to ``"stop"``.
    """
    if not isinstance(stop_reason, str):
        return "stop"
    return _FINISH_REASON_MAP.get(stop_reason, "stop")


def _strip_prefix(name: str, *, strip: bool) -> str:
    if strip and name.startswith(_MCP_TOOL_PREFIX):
        return name[len(_MCP_TOOL_PREFIX):]
    return name


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
    """Convert a stream of Anthropic raw streaming events into a stream of
    OpenAI ``chat.completion.chunk`` dicts.

    See the module docstring for the full contract. This is a pure generator;
    it never performs I/O and is defensive against malformed events.
    """

    def _chunk(delta: Dict[str, Any], finish_reason: Any = None) -> Dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    role_emitted = False

    # Anthropic content-block index -> OpenAI tool-call index ``k``, assigned in
    # arrival order, ONLY for tool_use blocks.
    tool_index_by_block: Dict[Any, int] = {}
    next_tool_index = 0

    # Usage accounting, harvested from message_start / message_delta.
    # Anthropic reports input tokens split three ways (uncached / cache_read /
    # cache_creation); OpenAI's ``prompt_tokens`` is their sum, with the cached
    # portion surfaced separately under ``prompt_tokens_details``.
    input_tokens = 0
    cache_read = 0
    cache_creation = 0
    completion_tokens = 0

    # finish_reason, mapped from message_delta; defaults to "stop".
    finish_reason = "stop"

    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")

        if etype == "message_start":
            if not role_emitted:
                role_emitted = True
                yield _chunk({"role": "assistant"})
            message = event.get("message")
            if isinstance(message, dict):
                usage = message.get("usage")
                if isinstance(usage, dict):
                    val = usage.get("input_tokens")
                    if isinstance(val, int):
                        input_tokens = val
                    val = usage.get("cache_read_input_tokens")
                    if isinstance(val, int):
                        cache_read = val
                    val = usage.get("cache_creation_input_tokens")
                    if isinstance(val, int):
                        cache_creation = val
            continue

        # Any event that produces output must be preceded by the role chunk.
        if not role_emitted:
            role_emitted = True
            yield _chunk({"role": "assistant"})

        if etype == "content_block_start":
            block = event.get("content_block")
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                # text / thinking blocks do not consume a tool-call index.
                continue
            block_index = event.get("index")
            k = next_tool_index
            next_tool_index += 1
            tool_index_by_block[block_index] = k
            name = block.get("name")
            if not isinstance(name, str):
                name = ""
            name = _strip_prefix(name, strip=strip_mcp_prefix)
            yield _chunk(
                {
                    "tool_calls": [
                        {
                            "index": k,
                            "id": block.get("id"),
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                }
            )
            continue

        if etype == "content_block_delta":
            delta = event.get("delta")
            if not isinstance(delta, dict):
                continue
            dtype = delta.get("type")

            if dtype == "text_delta":
                text = delta.get("text")
                if isinstance(text, str):
                    yield _chunk({"content": text})
                continue

            if dtype == "input_json_delta":
                block_index = event.get("index")
                if block_index not in tool_index_by_block:
                    # arg delta for an unknown / non-tool block — skip.
                    continue
                partial = delta.get("partial_json")
                if not isinstance(partial, str):
                    continue
                k = tool_index_by_block[block_index]
                yield _chunk(
                    {
                        "tool_calls": [
                            {"index": k, "function": {"arguments": partial}}
                        ]
                    }
                )
                continue

            if dtype == "thinking_delta":
                if include_reasoning:
                    thinking = delta.get("thinking")
                    if isinstance(thinking, str):
                        yield _chunk({"reasoning_content": thinking})
                continue

            # signature_delta and any unknown delta type: ignore.
            continue

        if etype == "content_block_stop":
            # Nothing to emit; the OpenAI stream has no per-block stop event.
            continue

        if etype == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, dict):
                finish_reason = _map_finish_reason(delta.get("stop_reason"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                val = usage.get("output_tokens")
                if isinstance(val, int):
                    completion_tokens = val
                # Some models / paths report cache fields on the message_delta
                # as well; capture them if present.
                val = usage.get("cache_read_input_tokens")
                if isinstance(val, int):
                    cache_read = val
                val = usage.get("cache_creation_input_tokens")
                if isinstance(val, int):
                    cache_creation = val
            continue

        if etype == "message_stop":
            # Terminal Anthropic marker; the final OpenAI chunk is emitted after
            # the loop. Nothing to do here.
            continue

        # Unknown event type: ignore defensively.
        continue

    # The role chunk must be emitted exactly once, even for an empty stream.
    if not role_emitted:
        role_emitted = True
        yield _chunk({"role": "assistant"})

    # Final chunk carries the mapped finish_reason and an empty delta.
    yield _chunk({}, finish_reason=finish_reason)

    # Optional trailing usage chunk (after the finish_reason chunk).
    if include_usage:
        prompt_tokens = input_tokens + cache_read + cache_creation
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cache_read},
            },
        }
