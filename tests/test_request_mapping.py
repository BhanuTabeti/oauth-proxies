"""Tests for OpenAI-request -> Anthropic-kwargs mapping."""
from oauth_proxy import request_mapping as rm
from oauth_proxy.models import ChatCompletionRequest


def _req(**kw):
    base = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    base.update(kw)
    return ChatCompletionRequest.model_validate(base)


def test_claude_model_passes_through():
    kwargs = rm.build_kwargs(_req(model="claude-sonnet-4-6"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["model"] == "claude-sonnet-4-6"


def test_non_claude_model_substituted_with_default():
    kwargs = rm.build_kwargs(_req(model="gpt-4o"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["model"] == "claude-opus-4-7"


def test_openrouter_dotted_claude_name_normalized():
    kwargs = rm.build_kwargs(_req(model="anthropic/claude-opus-4.6"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["model"] == "claude-opus-4-6"


def test_reasoning_off_means_no_thinking():
    kwargs = rm.build_kwargs(_req(), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert "thinking" not in kwargs


def test_reasoning_effort_maps_to_adaptive_thinking():
    kwargs = rm.build_kwargs(_req(reasoning_effort="high"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["thinking"]["type"] == "adaptive"
    assert kwargs["output_config"]["effort"] == "high"


def test_default_reasoning_effort_applies_when_request_silent():
    kwargs = rm.build_kwargs(_req(), default_model="claude-opus-4-7", default_reasoning_effort="medium")
    assert kwargs["output_config"]["effort"] == "medium"


def test_oauth_system_prefix_and_mcp_tool_prefix_applied():
    tools = [{"type": "function", "function": {"name": "calc", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    req = _req(messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}], tools=tools)
    kwargs = rm.build_kwargs(req, default_model="claude-opus-4-7", default_reasoning_effort="off")
    # is_oauth=True -> Claude Code identity prepended to system.
    assert kwargs["system"][0]["text"].startswith("You are Claude Code")
    # is_oauth=True -> tool names prefixed with mcp_.
    assert kwargs["tools"][0]["name"] == "mcp_calc"


def test_tool_choice_specific_function():
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]
    req = _req(tools=tools, tool_choice={"type": "function", "function": {"name": "calc"}})
    kwargs = rm.build_kwargs(req, default_model="claude-opus-4-7", default_reasoning_effort="off")
    # Adapter prefixes the requested tool name too is NOT done; tool_choice carries the raw name.
    assert kwargs["tool_choice"] == {"type": "tool", "name": "calc"}


def test_tool_choice_required_maps_to_any():
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]
    kwargs = rm.build_kwargs(_req(tools=tools, tool_choice="required"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["tool_choice"] == {"type": "any"}


def test_tool_choice_none_drops_tools():
    tools = [{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]
    kwargs = rm.build_kwargs(_req(tools=tools, tool_choice="none"), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert "tools" not in kwargs


def test_max_tokens_alias_and_passthrough():
    kwargs = rm.build_kwargs(_req(max_completion_tokens=321), default_model="claude-opus-4-7", default_reasoning_effort="off")
    assert kwargs["max_tokens"] == 321
