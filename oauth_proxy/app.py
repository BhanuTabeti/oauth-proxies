"""FastAPI app: OpenAI-compatible surface over Claude via OAuth subscription.

Routes:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions   (stream + non-stream)

Endpoints are synchronous on purpose — FastAPI runs them in a worker thread, so
the blocking Anthropic SDK calls (and the streaming generator) don't block the
event loop. This is a single-user, localhost-first proxy; that trade-off keeps
the wiring simple and correct.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oauth_proxy import request_mapping, response_mapping, stream_mapping
from oauth_proxy.auth import TokenError, TokenProvider
from oauth_proxy.config import Config, load_config
from oauth_proxy.models import ChatCompletionRequest, model_catalog


def _error_response(status: int, message: str, etype: str, code: Optional[str] = None) -> JSONResponse:
    """OpenAI-style error envelope."""
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "param": None, "code": code}},
    )


def _new_completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def build_app(config: Optional[Config] = None, token_provider: Optional[TokenProvider] = None) -> FastAPI:
    cfg = config or load_config()
    tokens = token_provider or TokenProvider(timeout=cfg.request_timeout_seconds)
    app = FastAPI(title="oauth-proxy", version="0.1.0")
    app.state.config = cfg
    app.state.tokens = tokens

    def _check_client_auth(authorization: Optional[str]) -> Optional[JSONResponse]:
        """Enforce the optional shared secret. Returns an error response if the
        client is not authorized, else None."""
        if not cfg.proxy_api_key:
            return None
        expected = f"Bearer {cfg.proxy_api_key}"
        if authorization != expected:
            return _error_response(
                401, "Invalid or missing API key.", "authentication_error", "invalid_api_key"
            )
        return None

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models(authorization: Optional[str] = Header(default=None)):
        auth_err = _check_client_auth(authorization)
        if auth_err is not None:
            return auth_err
        return model_catalog()

    @app.post("/v1/chat/completions")
    def chat_completions(
        raw: Dict[str, Any],
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth_err = _check_client_auth(authorization)
        if auth_err is not None:
            return auth_err

        # Validate request body.
        try:
            req = ChatCompletionRequest.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError
            return _error_response(400, f"Invalid request: {exc}", "invalid_request_error")

        # Resolve credentials + build the upstream client.
        try:
            client = tokens.build_client()
        except TokenError as exc:
            return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")
        except Exception as exc:  # pragma: no cover - unexpected client build failure
            return _error_response(500, f"Failed to build Anthropic client: {exc}", "api_error")

        kwargs = request_mapping.build_kwargs(
            req,
            default_model=cfg.default_model,
            default_reasoning_effort=cfg.default_reasoning_effort,
        )
        resolved_model = kwargs.get("model", req.model)
        completion_id = _new_completion_id()
        created = int(time.time())

        if req.stream:
            return StreamingResponse(
                _stream_sse(
                    client,
                    kwargs,
                    model=resolved_model,
                    completion_id=completion_id,
                    created=created,
                    include_reasoning=cfg.include_reasoning,
                    include_usage=req.wants_usage(),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming.
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:
            status, etype, code = _classify_upstream_error(exc)
            return _error_response(status, str(exc), etype, code)

        dumped = message.model_dump() if hasattr(message, "model_dump") else dict(message)
        body = response_mapping.anthropic_message_to_openai(
            dumped,
            model=resolved_model,
            completion_id=completion_id,
            created=created,
            include_reasoning=cfg.include_reasoning,
        )
        return JSONResponse(content=body)

    return app


def _stream_sse(
    client: Any,
    kwargs: Dict[str, Any],
    *,
    model: str,
    completion_id: str,
    created: int,
    include_reasoning: bool,
    include_usage: bool,
) -> Iterator[str]:
    """Yield Server-Sent-Events for an OpenAI streaming chat completion."""
    try:
        upstream = client.messages.create(**kwargs, stream=True)
        anth_events = (
            event.model_dump() if hasattr(event, "model_dump") else event for event in upstream
        )
        for chunk in stream_mapping.anthropic_events_to_openai_chunks(
            anth_events,
            model=model,
            completion_id=completion_id,
            created=created,
            include_reasoning=include_reasoning,
            include_usage=include_usage,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as exc:
        # Mid-stream errors can't change the HTTP status; surface as an SSE
        # error event so the client sees what happened.
        _status, etype, code = _classify_upstream_error(exc)
        err = {"error": {"message": str(exc), "type": etype, "param": None, "code": code}}
        yield f"data: {json.dumps(err)}\n\n"
    yield "data: [DONE]\n\n"


def _classify_upstream_error(exc: Exception):
    """Map an Anthropic SDK exception to (http_status, openai_error_type, code)."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 401:
            return status, "authentication_error", "invalid_oauth_token"
        if status == 429:
            return status, "rate_limit_error", "rate_limit_exceeded"
        if 400 <= status < 500:
            return status, "invalid_request_error", None
        return status, "api_error", None
    return 502, "api_error", "upstream_error"


def main() -> None:
    """Console entry point: ``oauth-proxy``."""
    import uvicorn

    cfg = load_config()
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
