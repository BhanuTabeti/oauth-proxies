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
import logging
import sys
import time
import uuid
from typing import Any, Dict, Iterator, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from oauth_proxy import request_mapping, response_mapping, stream_mapping
from oauth_proxy.auth import TokenError, TokenProvider
from oauth_proxy.config import Config, load_config, load_dotenv
from oauth_proxy.models import ChatCompletionRequest, model_catalog

log = logging.getLogger("oauth_proxy")


def configure_logging(level: str = "INFO") -> None:
    """Attach a stdout handler to the proxy's logger. Idempotent."""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    log.setLevel(lvl)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s oauth-proxy %(message)s", "%H:%M:%S")
        )
        log.addHandler(handler)
        log.propagate = False


def _usage_summary(usage: Optional[Dict[str, Any]]) -> str:
    """Compact, secret-free one-liner of token usage incl. cache activity."""
    u = usage or {}
    return (
        f"prompt={u.get('input_tokens', 0)} completion={u.get('output_tokens', 0)} "
        f"cache_read={u.get('cache_read_input_tokens', 0)} "
        f"cache_write={u.get('cache_creation_input_tokens', 0)}"
    )


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
        t0 = time.monotonic()
        auth_err = _check_client_auth(authorization)
        if auth_err is not None:
            return auth_err

        # Validate request body.
        try:
            req = ChatCompletionRequest.model_validate(raw)
        except Exception as exc:  # pydantic ValidationError
            log.warning("400 invalid request: %s", exc)
            return _error_response(400, f"Invalid request: {exc}", "invalid_request_error")

        # Resolve credentials + build the upstream client.
        try:
            client = tokens.build_client()
        except TokenError as exc:
            log.warning("401 token unavailable: %s", exc)
            return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")
        except Exception as exc:  # pragma: no cover - unexpected client build failure
            log.warning("500 client build failed: %s", exc)
            return _error_response(500, f"Failed to build Anthropic client: {exc}", "api_error")

        kwargs = request_mapping.build_kwargs(
            req,
            default_model=cfg.default_model,
            default_reasoning_effort=cfg.default_reasoning_effort,
            prompt_cache=cfg.prompt_cache,
        )
        resolved_model = kwargs.get("model", req.model)
        completion_id = _new_completion_id()
        created = int(time.time())
        log.info("→ POST /v1/chat/completions model=%s stream=%s", resolved_model, req.stream)

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
                    started=t0,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Non-streaming.
        try:
            message = client.messages.create(**kwargs)
        except Exception as exc:
            status, etype, code = _classify_upstream_error(exc)
            log.warning("← %s %s (%dms)", status, etype, int((time.monotonic() - t0) * 1000))
            return _error_response(status, str(exc), etype, code)

        dumped = message.model_dump() if hasattr(message, "model_dump") else dict(message)
        log.info(
            "← 200 model=%s %dms %s",
            resolved_model,
            int((time.monotonic() - t0) * 1000),
            _usage_summary(dumped.get("usage")),
        )
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
    started: Optional[float] = None,
) -> Iterator[str]:
    """Yield Server-Sent-Events for an OpenAI streaming chat completion."""
    captured: Dict[str, Any] = {}

    def _anth_events() -> Iterator[Dict[str, Any]]:
        # Sniff usage off the raw events as they pass through (message_start
        # carries input + cache tokens; message_delta carries output tokens).
        for event in upstream:
            d = event.model_dump() if hasattr(event, "model_dump") else event
            if isinstance(d, dict):
                etype = d.get("type")
                if etype == "message_start":
                    captured.update((d.get("message") or {}).get("usage") or {})
                elif etype == "message_delta":
                    captured.update(d.get("usage") or {})
            yield d

    try:
        upstream = client.messages.create(**kwargs, stream=True)
        for chunk in stream_mapping.anthropic_events_to_openai_chunks(
            _anth_events(),
            model=model,
            completion_id=completion_id,
            created=created,
            include_reasoning=include_reasoning,
            include_usage=include_usage,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        ms = int((time.monotonic() - started) * 1000) if started else -1
        log.info("← 200 (stream) model=%s %dms %s", model, ms, _usage_summary(captured))
    except Exception as exc:
        # Mid-stream errors can't change the HTTP status; surface as an SSE
        # error event so the client sees what happened.
        _status, etype, code = _classify_upstream_error(exc)
        log.warning("stream error: %s: %s", etype, exc)
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


def _login_cli(args: list) -> None:
    """``oauth-proxy login {codex|grok}`` — run a provider's OAuth login."""
    provider = (args[0] if args else "").lower()
    load_dotenv()
    configure_logging("INFO")
    if provider == "codex":
        from oauth_proxy import codex_auth as prov
    elif provider == "grok":
        from oauth_proxy import grok_auth as prov
    else:
        print("usage: oauth-proxy login {codex|grok}")
        raise SystemExit(2)
    try:
        record = prov.login()
    except prov.TokenError as exc:
        print(f"Login failed: {exc}")
        raise SystemExit(1)
    acc = record.get("account_id") or "(none)"
    print(f"✓ {provider} login stored (account_id={acc}). You can now make requests.")


def main() -> None:
    """Console entry point: ``oauth-proxy`` (server) / ``oauth-proxy login ...``."""
    import sys

    argv = sys.argv[1:]
    if argv and argv[0] == "login":
        _login_cli(argv[1:])
        return

    import uvicorn

    load_dotenv()  # load .env from the working directory, if present
    cfg = load_config()
    configure_logging(cfg.log_level)
    log.info(
        "starting on %s:%d (prompt_cache=%s, default_model=%s)",
        cfg.host, cfg.port, cfg.prompt_cache, cfg.default_model,
    )
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
