"""Tests for ``oauth_proxy.stream_mapping.anthropic_events_to_openai_chunks``.

Written test-first against the module contract (see stream_mapping.py docstring
and the implementation brief). The converter is a pure generator: an iterable of
Anthropic raw streaming-event dicts (``event.model_dump()``) in -> an iterator of
OpenAI ``chat.completion.chunk`` dicts out.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

from oauth_proxy.stream_mapping import anthropic_events_to_openai_chunks

# ── Common kwargs ────────────────────────────────────────────────────────────
COMMON = dict(model="claude-opus-4-7", completion_id="chatcmpl-xyz", created=1700000000)


def convert(events: List[Dict[str, Any]], **overrides: Any) -> List[Dict[str, Any]]:
    kwargs = {**COMMON, **overrides}
    return list(anthropic_events_to_openai_chunks(iter(events), **kwargs))


# ── Event builders ───────────────────────────────────────────────────────────
def message_start(
    input_tokens: int = 12,
    output_tokens: int = 1,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> Dict[str, Any]:
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage["cache_creation_input_tokens"] = cache_creation
    return {
        "type": "message_start",
        "message": {"id": "msg_abc", "role": "assistant", "usage": usage},
    }


def text_start(index: int = 0) -> Dict[str, Any]:
    return {"type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}


def text_delta(text: str, index: int = 0) -> Dict[str, Any]:
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "text_delta", "text": text}}


def tool_start(index: int, tool_id: str, name: str) -> Dict[str, Any]:
    return {"type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": tool_id, "name": name, "input": {}}}


def tool_delta(index: int, partial_json: str) -> Dict[str, Any]:
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json}}


def thinking_start(index: int) -> Dict[str, Any]:
    return {"type": "content_block_start", "index": index,
            "content_block": {"type": "thinking", "thinking": ""}}


def thinking_delta(index: int, thinking: str) -> Dict[str, Any]:
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking}}


def signature_delta(index: int, signature: str = "sig") -> Dict[str, Any]:
    return {"type": "content_block_delta", "index": index,
            "delta": {"type": "signature_delta", "signature": signature}}


def block_stop(index: int) -> Dict[str, Any]:
    return {"type": "content_block_stop", "index": index}


def message_delta(stop_reason: Any, output_tokens: int = 34) -> Dict[str, Any]:
    return {"type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens}}


def message_stop() -> Dict[str, Any]:
    return {"type": "message_stop"}


# ── Skeleton / shape helpers ─────────────────────────────────────────────────
def assert_skeleton(chunk: Dict[str, Any]) -> None:
    assert chunk["id"] == COMMON["completion_id"]
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["created"] == COMMON["created"]
    assert chunk["model"] == COMMON["model"]
    assert "choices" in chunk


def delta_of(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return chunk["choices"][0]["delta"]


def finish_of(chunk: Dict[str, Any]) -> Any:
    return chunk["choices"][0]["finish_reason"]


# ── Tests: pure text ─────────────────────────────────────────────────────────
def test_pure_text_stream():
    events = [
        message_start(),
        text_start(0),
        text_delta("Hel", 0),
        text_delta("lo", 0),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events)

    for c in chunks:
        assert_skeleton(c)

    # First chunk is the role chunk.
    assert delta_of(chunks[0]) == {"role": "assistant"}
    assert finish_of(chunks[0]) is None

    # Content chunks in order.
    contents = [delta_of(c)["content"] for c in chunks if "content" in delta_of(c)]
    assert contents == ["Hel", "lo"]
    assert "".join(contents) == "Hello"

    # Final chunk carries finish_reason and empty delta.
    last = chunks[-1]
    assert finish_of(last) == "stop"
    assert delta_of(last) == {}

    # Content chunks have finish_reason None.
    for c in chunks:
        if "content" in delta_of(c):
            assert finish_of(c) is None


def test_role_and_finish_emitted_exactly_once():
    events = [
        message_start(),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events)

    role_chunks = [c for c in chunks if delta_of(c).get("role") == "assistant"]
    finish_chunks = [c for c in chunks if finish_of(c) is not None]
    assert len(role_chunks) == 1
    assert len(finish_chunks) == 1
    # Role chunk is first; finish chunk is the last non-usage chunk.
    assert chunks[0] is role_chunks[0]


def test_role_chunk_emitted_even_without_message_start():
    events = [
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events)
    # Role chunk must precede the first content chunk.
    assert delta_of(chunks[0]) == {"role": "assistant"}
    first_content_idx = next(i for i, c in enumerate(chunks) if "content" in delta_of(c))
    assert first_content_idx > 0


# ── Tests: single tool call ──────────────────────────────────────────────────
def test_single_tool_call():
    events = [
        message_start(),
        tool_start(1, "toolu_1", "mcp_calc"),
        tool_delta(1, '{"x":'),
        tool_delta(1, "2}"),
        block_stop(1),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events)

    assert delta_of(chunks[0]) == {"role": "assistant"}

    # Tool start chunk.
    start_chunks = [
        c for c in chunks
        if "tool_calls" in delta_of(c) and delta_of(c)["tool_calls"][0].get("id")
    ]
    assert len(start_chunks) == 1
    tc = delta_of(start_chunks[0])["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["id"] == "toolu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "calc"  # mcp_ stripped by default
    assert tc["function"]["arguments"] == ""

    # Arg-delta chunks forwarded verbatim, concatenation reassembles JSON.
    arg_pieces = []
    for c in chunks:
        d = delta_of(c)
        if "tool_calls" in d:
            fn = d["tool_calls"][0].get("function", {})
            if "arguments" in fn and not d["tool_calls"][0].get("id"):
                arg_pieces.append(fn["arguments"])
    assert arg_pieces == ['{"x":', "2}"]
    assert "".join(arg_pieces) == '{"x":2}'

    # All tool-arg delta chunks reference index 0.
    for c in chunks:
        d = delta_of(c)
        if "tool_calls" in d:
            assert d["tool_calls"][0]["index"] == 0

    # Final finish reason.
    assert finish_of(chunks[-1]) == "tool_calls"


def test_two_tool_calls_index_mapping():
    # text at anthropic index 0, tools at anthropic indices 1 and 2.
    events = [
        message_start(),
        text_start(0),
        text_delta("ok", 0),
        block_stop(0),
        tool_start(1, "toolu_a", "mcp_first"),
        tool_delta(1, "{}"),
        block_stop(1),
        tool_start(2, "toolu_b", "mcp_second"),
        tool_delta(2, "{}"),
        block_stop(2),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events)

    # Collect tool start chunks in order with their ids and openai indices.
    starts = []
    for c in chunks:
        d = delta_of(c)
        if "tool_calls" in d and d["tool_calls"][0].get("id"):
            starts.append((d["tool_calls"][0]["id"], d["tool_calls"][0]["index"]))
    assert starts == [("toolu_a", 0), ("toolu_b", 1)]

    # Arg deltas map to correct openai indices (anthropic 1->0, 2->1).
    arg_indices_by_value = {}
    for c in chunks:
        d = delta_of(c)
        if "tool_calls" in d and not d["tool_calls"][0].get("id"):
            tcd = d["tool_calls"][0]
            arg_indices_by_value.setdefault(tcd["index"], []).append(tcd["function"]["arguments"])
    # Both tools emitted arg deltas at openai indices 0 and 1.
    assert set(arg_indices_by_value.keys()) == {0, 1}


def test_text_does_not_consume_tool_index():
    # Tool at anthropic index 1 with text before it must still be openai index 0.
    events = [
        message_start(),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        tool_start(1, "toolu_only", "mcp_solo"),
        tool_delta(1, "{}"),
        block_stop(1),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events)
    start = next(
        delta_of(c)["tool_calls"][0]
        for c in chunks
        if "tool_calls" in delta_of(c) and delta_of(c)["tool_calls"][0].get("id")
    )
    assert start["index"] == 0


# ── Tests: mcp_ stripping ────────────────────────────────────────────────────
def test_mcp_prefix_stripped_by_default():
    events = [
        message_start(),
        tool_start(1, "toolu_1", "mcp_get_weather"),
        block_stop(1),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events)
    start = next(
        delta_of(c)["tool_calls"][0]
        for c in chunks
        if "tool_calls" in delta_of(c) and delta_of(c)["tool_calls"][0].get("id")
    )
    assert start["function"]["name"] == "get_weather"


def test_mcp_prefix_kept_when_disabled():
    events = [
        message_start(),
        tool_start(1, "toolu_1", "mcp_get_weather"),
        block_stop(1),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events, strip_mcp_prefix=False)
    start = next(
        delta_of(c)["tool_calls"][0]
        for c in chunks
        if "tool_calls" in delta_of(c) and delta_of(c)["tool_calls"][0].get("id")
    )
    assert start["function"]["name"] == "mcp_get_weather"


def test_non_mcp_name_unchanged():
    events = [
        message_start(),
        tool_start(1, "toolu_1", "plain_tool"),
        block_stop(1),
        message_delta("tool_use"),
        message_stop(),
    ]
    chunks = convert(events)
    start = next(
        delta_of(c)["tool_calls"][0]
        for c in chunks
        if "tool_calls" in delta_of(c) and delta_of(c)["tool_calls"][0].get("id")
    )
    assert start["function"]["name"] == "plain_tool"


# ── Tests: thinking / reasoning ──────────────────────────────────────────────
def test_reasoning_surfaced_when_enabled():
    events = [
        message_start(),
        thinking_start(0),
        thinking_delta(0, "hmm"),
        thinking_delta(0, " ok"),
        signature_delta(0, "xx"),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events, include_reasoning=True)
    reasoning = [delta_of(c)["reasoning_content"] for c in chunks if "reasoning_content" in delta_of(c)]
    assert reasoning == ["hmm", " ok"]
    # signature_delta never produces output.
    assert all("signature" not in delta_of(c) for c in chunks)


def test_reasoning_dropped_when_disabled():
    events = [
        message_start(),
        thinking_start(0),
        thinking_delta(0, "hmm"),
        signature_delta(0, "xx"),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events, include_reasoning=False)
    assert all("reasoning_content" not in delta_of(c) for c in chunks)
    # Only role chunk + final chunk.
    assert delta_of(chunks[0]) == {"role": "assistant"}
    assert finish_of(chunks[-1]) == "stop"


def test_signature_delta_always_ignored_even_with_reasoning():
    events = [
        message_start(),
        thinking_start(0),
        signature_delta(0, "only-signature"),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events, include_reasoning=True)
    assert all("reasoning_content" not in delta_of(c) for c in chunks)


# ── Tests: finish_reason mapping ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("pause_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        (None, "stop"),
        ("something_unknown", "stop"),
    ],
)
def test_finish_reason_mapping(stop_reason, expected):
    events = [
        message_start(),
        text_start(0),
        text_delta("x", 0),
        block_stop(0),
        message_delta(stop_reason),
        message_stop(),
    ]
    chunks = convert(events)
    assert finish_of(chunks[-1]) == expected


def test_finish_reason_when_message_delta_missing():
    # No message_delta at all -> default to "stop".
    events = [
        message_start(),
        text_start(0),
        text_delta("x", 0),
        block_stop(0),
        message_stop(),
    ]
    chunks = convert(events)
    finish_chunks = [c for c in chunks if finish_of(c) is not None]
    assert len(finish_chunks) == 1
    assert finish_of(finish_chunks[0]) == "stop"


# ── Tests: usage ─────────────────────────────────────────────────────────────
def test_usage_chunk_emitted_when_enabled():
    events = [
        message_start(input_tokens=12, output_tokens=1),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn", output_tokens=34),
        message_stop(),
    ]
    chunks = convert(events, include_usage=True)

    # Final finish chunk comes before the usage chunk.
    finish_idx = next(i for i, c in enumerate(chunks) if finish_of(c) is not None)
    usage_chunks = [c for c in chunks if "usage" in c]
    assert len(usage_chunks) == 1
    usage_chunk = usage_chunks[0]
    usage_idx = chunks.index(usage_chunk)
    assert usage_idx > finish_idx
    # Usage chunk is last and has empty choices.
    assert usage_chunk is chunks[-1]
    assert usage_chunk["choices"] == []
    assert_skeleton(usage_chunk)
    assert usage_chunk["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 34,
        "total_tokens": 46,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_usage_chunk_includes_cached_tokens_when_present():
    # message_start carries cache_read_input_tokens; the trailing usage chunk
    # must reflect it as prompt_tokens_details.cached_tokens and the totals
    # must include both cache_read and cache_creation as input.
    events = [
        message_start(input_tokens=12, output_tokens=1,
                      cache_read=13334, cache_creation=701),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn", output_tokens=34),
        message_stop(),
    ]
    chunks = convert(events, include_usage=True)
    usage_chunk = chunks[-1]
    assert usage_chunk["usage"] == {
        "prompt_tokens": 12 + 13334 + 701,
        "completion_tokens": 34,
        "total_tokens": 12 + 13334 + 701 + 34,
        "prompt_tokens_details": {"cached_tokens": 13334},
    }


def test_no_usage_chunk_when_disabled():
    events = [
        message_start(input_tokens=12, output_tokens=1),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn", output_tokens=34),
        message_stop(),
    ]
    chunks = convert(events, include_usage=False)
    assert all("usage" not in c for c in chunks)
    assert all(c["choices"] for c in chunks)  # no empty-choices chunk


# ── Tests: robustness / no DONE sentinel ─────────────────────────────────────
def test_no_done_sentinel():
    events = [
        message_start(),
        text_start(0),
        text_delta("hi", 0),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events)
    assert all(c != "[DONE]" for c in chunks)
    assert all(isinstance(c, dict) for c in chunks)


def test_malformed_events_do_not_raise():
    events = [
        message_start(),
        {"type": "content_block_delta"},  # no index, no delta
        {"type": "content_block_start", "index": 5},  # no content_block
        {"type": "content_block_delta", "index": 0, "delta": {"type": "unknown_delta"}},
        {"weird": "object"},
        text_start(0),
        text_delta("ok", 0),
        block_stop(0),
        message_delta("end_turn"),
        message_stop(),
    ]
    chunks = convert(events)
    # Still produced a role chunk, the text, and a finish chunk.
    assert delta_of(chunks[0]) == {"role": "assistant"}
    contents = [delta_of(c)["content"] for c in chunks if "content" in delta_of(c)]
    assert contents == ["ok"]
    assert finish_of(chunks[-1]) == "stop"


def test_empty_event_stream_still_emits_role_and_finish():
    chunks = convert([])
    assert delta_of(chunks[0]) == {"role": "assistant"}
    finish_chunks = [c for c in chunks if finish_of(c) is not None]
    assert len(finish_chunks) == 1
    assert finish_of(finish_chunks[0]) == "stop"
