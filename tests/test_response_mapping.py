"""Tests for oauth_proxy.response_mapping.anthropic_message_to_openai.

The converter is a pure function: Anthropic ``Message.model_dump()`` dict in,
OpenAI ``chat.completion`` dict out. No network/SDK involved.
"""
from __future__ import annotations

import json

import pytest

from oauth_proxy.response_mapping import anthropic_message_to_openai

# Common kwargs for the converter.
KW = dict(model="gpt-4o", completion_id="chatcmpl-abc", created=1700000000)


def _msg(**overrides):
    """Build a minimal Anthropic Message.model_dump()-style dict."""
    base = {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-7",
        "content": [],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 12,
            "output_tokens": 34,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 5,
        },
    }
    base.update(overrides)
    return base


# ── envelope ──────────────────────────────────────────────────────────────
def test_envelope_fields():
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "hi"}]), **KW
    )
    assert out["id"] == "chatcmpl-abc"
    assert out["object"] == "chat.completion"
    assert out["created"] == 1700000000
    assert out["model"] == "gpt-4o"
    assert len(out["choices"]) == 1
    choice = out["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"


# ── plain text ──────────────────────────────────────────────────────────────
def test_plain_text_response():
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "Hello there"}], stop_reason="end_turn"),
        **KW,
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Hello there"
    assert "tool_calls" not in msg
    assert "reasoning_content" not in msg
    assert out["choices"][0]["finish_reason"] == "stop"
    # _msg's default usage includes cache_creation_input_tokens=5, so
    # prompt_tokens = input(12) + cache_read(0) + cache_creation(5) = 17.
    assert out["usage"] == {
        "prompt_tokens": 17,
        "completion_tokens": 34,
        "total_tokens": 51,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_multiple_text_blocks_concatenated_in_order():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]
        ),
        **KW,
    )
    assert out["choices"][0]["message"]["content"] == "Hello world"


# ── tool_use ────────────────────────────────────────────────────────────────
def test_single_tool_use():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "SF"},
                }
            ],
            stop_reason="tool_use",
        ),
        **KW,
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_01"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "SF"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_text_and_tool_use_together():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "text", "text": "Let me check."},
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            stop_reason="tool_use",
        ),
        **KW,
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"


def test_multiple_tool_use_preserves_order():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "tool_use", "id": "t1", "name": "a", "input": {"n": 1}},
                {"type": "tool_use", "id": "t2", "name": "b", "input": {"n": 2}},
                {"type": "tool_use", "id": "t3", "name": "c", "input": {"n": 3}},
            ],
            stop_reason="tool_use",
        ),
        **KW,
    )
    tcs = out["choices"][0]["message"]["tool_calls"]
    assert [tc["id"] for tc in tcs] == ["t1", "t2", "t3"]
    assert [tc["function"]["name"] for tc in tcs] == ["a", "b", "c"]


def test_tool_use_missing_input_defaults_to_empty_object():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "noargs"}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    tc = out["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["arguments"] == "{}"
    assert json.loads(tc["function"]["arguments"]) == {}


def test_arguments_round_trip_via_json_loads():
    original = {"city": "São Paulo", "nested": {"a": [1, 2, 3]}, "flag": True, "x": None}
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "f", "input": original}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    args = out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(args) == original


def test_arguments_ensure_ascii_false():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"city": "München"}}
            ],
            stop_reason="tool_use",
        ),
        **KW,
    )
    args = out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    # ensure_ascii=False keeps the non-ASCII char literal, not \uXXXX escaped.
    assert "München" in args


# ── mcp_ prefix stripping ────────────────────────────────────────────────────
def test_mcp_prefix_stripped_by_default():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "tool_use", "id": "t1", "name": "mcp_get_weather", "input": {}}
            ],
            stop_reason="tool_use",
        ),
        **KW,
    )
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_mcp_prefix_preserved_when_disabled():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "tool_use", "id": "t1", "name": "mcp_get_weather", "input": {}}
            ],
            stop_reason="tool_use",
        ),
        strip_mcp_prefix=False,
        **KW,
    )
    assert (
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        == "mcp_get_weather"
    )


def test_name_without_mcp_prefix_untouched():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_only_one_mcp_prefix_stripped():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "mcp_mcp_x", "input": {}}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    assert out["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "mcp_x"


# ── reasoning_content ────────────────────────────────────────────────────────
def test_reasoning_included_when_requested():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "thinking", "thinking": "let me think...", "signature": "abc"},
                {"type": "text", "text": "Answer"},
            ]
        ),
        include_reasoning=True,
        **KW,
    )
    msg = out["choices"][0]["message"]
    assert msg["reasoning_content"] == "let me think..."
    assert msg["content"] == "Answer"


def test_reasoning_omitted_when_not_requested():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "thinking", "thinking": "secret thoughts", "signature": "abc"},
                {"type": "text", "text": "Answer"},
            ]
        ),
        include_reasoning=False,
        **KW,
    )
    assert "reasoning_content" not in out["choices"][0]["message"]


def test_multiple_thinking_blocks_concatenated():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "thinking", "thinking": "first ", "signature": "a"},
                {"type": "thinking", "thinking": "second", "signature": "b"},
                {"type": "text", "text": "ok"},
            ]
        ),
        include_reasoning=True,
        **KW,
    )
    assert out["choices"][0]["message"]["reasoning_content"] == "first second"


def test_redacted_thinking_contributes_no_text():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "redacted_thinking", "data": "ENCRYPTED"},
                {"type": "text", "text": "ok"},
            ]
        ),
        include_reasoning=True,
        **KW,
    )
    # Only redacted_thinking (no plaintext) -> no reasoning_content key.
    assert "reasoning_content" not in out["choices"][0]["message"]


def test_redacted_thinking_mixed_with_thinking():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "redacted_thinking", "data": "ENCRYPTED"},
                {"type": "thinking", "thinking": "visible", "signature": "s"},
                {"type": "text", "text": "ok"},
            ]
        ),
        include_reasoning=True,
        **KW,
    )
    assert out["choices"][0]["message"]["reasoning_content"] == "visible"


def test_reasoning_requested_but_no_thinking_block():
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "ok"}]),
        include_reasoning=True,
        **KW,
    )
    assert "reasoning_content" not in out["choices"][0]["message"]


# ── finish_reason mapping ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "stop_reason,expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("pause_turn", "stop"),
        ("max_tokens", "length"),
        # Per contract rule 5 the base mapping tool_use->tool_calls is
        # unconditional; it does not depend on tool blocks being present.
        ("tool_use", "tool_calls"),
        ("refusal", "content_filter"),
        (None, "stop"),
        ("something_unknown", "stop"),
    ],
)
def test_finish_reason_mapping(stop_reason, expected):
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "x"}], stop_reason=stop_reason),
        **KW,
    )
    assert out["choices"][0]["finish_reason"] == expected


def test_tool_use_stop_with_tool_block_yields_tool_calls():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_missing_stop_reason_key_defaults_to_stop():
    m = _msg(content=[{"type": "text", "text": "x"}])
    del m["stop_reason"]
    out = anthropic_message_to_openai(m, **KW)
    assert out["choices"][0]["finish_reason"] == "stop"


# ── usage mapping ────────────────────────────────────────────────────────────
def test_usage_mapping():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "text", "text": "x"}],
            usage={"input_tokens": 100, "output_tokens": 250},
        ),
        **KW,
    )
    assert out["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 250,
        "total_tokens": 350,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_usage_missing_defaults_to_zeros():
    m = _msg(content=[{"type": "text", "text": "x"}])
    del m["usage"]
    out = anthropic_message_to_openai(m, **KW)
    assert out["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_usage_none_defaults_to_zeros():
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "x"}], usage=None),
        **KW,
    )
    assert out["usage"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


def test_usage_partial_keys_default_to_zero():
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "text", "text": "x"}], usage={"input_tokens": 7}),
        **KW,
    )
    assert out["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 0,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0},
    }


# ── cache-token surfacing (Anthropic reports input/cache_read/cache_creation
#    separately; OpenAI's prompt_tokens = total input, with cached carved out
#    under prompt_tokens_details.cached_tokens) ────────────────────────────
def test_cache_read_surfaced_in_prompt_tokens_details():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "text", "text": "x"}],
            usage={"input_tokens": 50, "output_tokens": 10,
                   "cache_read_input_tokens": 200},
        ),
        **KW,
    )
    assert out["usage"]["prompt_tokens"] == 250  # 50 fresh + 200 cached
    assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 200}
    assert out["usage"]["total_tokens"] == 260


def test_cache_creation_included_in_prompt_tokens_but_not_cached_count():
    # cache_creation tokens were processed (and written to cache), so they
    # count toward prompt_tokens — but they're NOT cached reads.
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "text", "text": "x"}],
            usage={"input_tokens": 50, "output_tokens": 10,
                   "cache_creation_input_tokens": 500},
        ),
        **KW,
    )
    assert out["usage"]["prompt_tokens"] == 550  # 50 + 500 written
    assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}


def test_full_cache_breakdown_input_plus_read_plus_creation():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "text", "text": "x"}],
            usage={"input_tokens": 12, "output_tokens": 34,
                   "cache_read_input_tokens": 13334,
                   "cache_creation_input_tokens": 701},
        ),
        **KW,
    )
    assert out["usage"]["prompt_tokens"] == 12 + 13334 + 701
    assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 13334}
    assert out["usage"]["completion_tokens"] == 34
    assert out["usage"]["total_tokens"] == 12 + 13334 + 701 + 34


# ── content edge cases ───────────────────────────────────────────────────────
def test_empty_content_list_yields_empty_string():
    out = anthropic_message_to_openai(_msg(content=[]), **KW)
    msg = out["choices"][0]["message"]
    assert msg["content"] == ""
    assert "tool_calls" not in msg


def test_no_text_but_tool_use_yields_none_content():
    out = anthropic_message_to_openai(
        _msg(
            content=[{"type": "tool_use", "id": "t1", "name": "f", "input": {}}],
            stop_reason="tool_use",
        ),
        **KW,
    )
    assert out["choices"][0]["message"]["content"] is None


def test_thinking_only_no_text_no_tool_yields_empty_string():
    # Thinking present but no text and no tool_use -> content "".
    out = anthropic_message_to_openai(
        _msg(content=[{"type": "thinking", "thinking": "hmm", "signature": "s"}]),
        **KW,
    )
    assert out["choices"][0]["message"]["content"] == ""


# ── defensive: malformed input ───────────────────────────────────────────────
def test_missing_content_key():
    m = _msg()
    del m["content"]
    out = anthropic_message_to_openai(m, **KW)
    assert out["choices"][0]["message"]["content"] == ""


def test_content_not_a_list():
    out = anthropic_message_to_openai(_msg(content="oops not a list"), **KW)
    assert out["choices"][0]["message"]["content"] == ""


def test_content_none():
    out = anthropic_message_to_openai(_msg(content=None), **KW)
    assert out["choices"][0]["message"]["content"] == ""


def test_empty_message_dict():
    out = anthropic_message_to_openai({}, **KW)
    msg = out["choices"][0]["message"]
    assert msg["content"] == ""
    assert "tool_calls" not in msg
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["total_tokens"] == 0


def test_unknown_block_types_ignored():
    out = anthropic_message_to_openai(
        _msg(
            content=[
                {"type": "text", "text": "hi"},
                {"type": "server_tool_use", "id": "x", "name": "y"},
                {"type": "future_block_kind", "stuff": 1},
            ]
        ),
        **KW,
    )
    msg = out["choices"][0]["message"]
    assert msg["content"] == "hi"
    assert "tool_calls" not in msg
