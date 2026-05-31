"""Endpoint tests using FastAPI TestClient with a mocked Anthropic client.

No real OAuth token or network is involved: a fake TokenProvider returns a fake
client whose ``messages.create`` yields canned responses/events.
"""
import json

import pytest
from fastapi.testclient import TestClient

from oauth_proxy.app import build_app
from oauth_proxy.auth import TokenError
from oauth_proxy.config import Config


class _Dumpable:
    """Stands in for an Anthropic SDK object exposing ``.model_dump()``."""

    def __init__(self, data):
        self._data = data

    def model_dump(self):
        return self._data


class _FakeMessages:
    def __init__(self, message=None, events=None):
        self._message = message
        self._events = events or []
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return iter(self._events)
        return self._message


class _FakeClient:
    def __init__(self, message=None, events=None):
        self.messages = _FakeMessages(message=message, events=events)


class _FakeTokens:
    def __init__(self, client=None, error=None):
        self._client = client
        self._error = error

    def build_client(self):
        if self._error is not None:
            raise self._error
        return self._client


def _message_payload():
    return _Dumpable(
        {
            "id": "msg_1",
            "role": "assistant",
            "model": "claude-opus-4-7",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
    )


def _stream_events():
    return [
        _Dumpable({"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 5, "output_tokens": 1}}}),
        _Dumpable({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        _Dumpable({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}),
        _Dumpable({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}),
        _Dumpable({"type": "content_block_stop", "index": 0}),
        _Dumpable({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
        _Dumpable({"type": "message_stop"}),
    ]


def _client(cfg=None, fake_client=None, error=None):
    cfg = cfg or Config()
    return TestClient(build_app(cfg, token_provider=_FakeTokens(client=fake_client, error=error)))


def test_health():
    c = _client(fake_client=_FakeClient())
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_list_models():
    c = _client(fake_client=_FakeClient())
    r = c.get("/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "claude-opus-4-7" in ids


def test_chat_completion_non_stream():
    fake = _FakeClient(message=_message_payload())
    c = _client(fake_client=fake)
    r = c.post("/v1/chat/completions", json={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello!"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    # The proxy forwarded an OAuth-mode request upstream.
    assert fake.messages.last_kwargs["model"] == "claude-opus-4-7"


def test_chat_completion_stream():
    fake = _FakeClient(events=_stream_events())
    c = _client(fake_client=fake)
    r = c.post("/v1/chat/completions", json={"model": "claude-opus-4-7", "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    # Parse SSE data lines.
    chunks = []
    done_seen = False
    for line in r.text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            done_seen = True
            continue
        chunks.append(json.loads(payload))

    assert done_seen
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    text = "".join(
        ch["choices"][0]["delta"].get("content", "")
        for ch in chunks
        if ch["choices"] and "content" in ch["choices"][0]["delta"]
    )
    assert text == "Hello!"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    # stream=True was forwarded upstream.
    assert fake.messages.last_kwargs.get("stream") is True


def test_token_error_returns_401():
    c = _client(error=TokenError("Run `claude setup-token`."))
    r = c.post("/v1/chat/completions", json={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"
    assert "setup-token" in r.json()["error"]["message"]


def test_proxy_api_key_enforced():
    cfg = Config(proxy_api_key="secret123")
    fake = _FakeClient(message=_message_payload())
    c = _client(cfg=cfg, fake_client=fake)

    # Missing key -> 401.
    r = c.post("/v1/chat/completions", json={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401

    # Wrong key -> 401.
    r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer nope"}, json={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401

    # Correct key -> 200.
    r = c.post("/v1/chat/completions", headers={"Authorization": "Bearer secret123"}, json={"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_invalid_request_body_returns_400():
    c = _client(fake_client=_FakeClient(message=_message_payload()))
    # Missing required "messages".
    r = c.post("/v1/chat/completions", json={"model": "claude-opus-4-7"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
