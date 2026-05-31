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

from oauth_proxy import (
    codex_auth,
    codex_client,
    codex_request_mapping,
    codex_response_mapping,
    codex_stream_mapping,
    grok_auth,
    grok_client,
    request_mapping,
    response_mapping,
    stream_mapping,
)
from oauth_proxy.auth import TokenError, TokenProvider
from oauth_proxy.codex_auth import CodexTokenProvider
from oauth_proxy.config import Config, load_config, load_dotenv
from oauth_proxy.grok_auth import GrokTokenProvider
from oauth_proxy.models import ChatCompletionRequest, model_catalog
from oauth_proxy.routing import CODEX, GROK, route_provider

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


def _logged_in(provider: Any) -> bool:
    """Whether a token provider has usable credentials (for /v1/models gating).

    Providers without an ``is_logged_in`` method are assumed available."""
    check = getattr(provider, "is_logged_in", None)
    if check is None:
        return True
    try:
        return bool(check())
    except Exception:  # pragma: no cover - defensive
        return False


def build_app(
    config: Optional[Config] = None,
    token_provider: Optional[TokenProvider] = None,
    codex_token_provider: Optional[CodexTokenProvider] = None,
    grok_token_provider: Optional[GrokTokenProvider] = None,
) -> FastAPI:
    cfg = config or load_config()
    tokens = token_provider or TokenProvider(timeout=cfg.request_timeout_seconds)
    codex_tokens = codex_token_provider or CodexTokenProvider(timeout=cfg.request_timeout_seconds)
    grok_tokens = grok_token_provider or GrokTokenProvider(timeout=cfg.request_timeout_seconds)
    app = FastAPI(title="oauth-proxy", version="0.1.0")
    app.state.config = cfg
    app.state.tokens = tokens
    app.state.codex_tokens = codex_tokens
    app.state.grok_tokens = grok_tokens

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
        # Only advertise models for subscriptions that are actually logged in,
        # and fetch each provider's real allowlist live where supported.
        available = set()
        live: Dict[str, Any] = {}
        if _logged_in(tokens):
            available.add("anthropic")
        if _logged_in(codex_tokens):
            available.add("codex")
            try:
                live["codex"] = codex_client.list_models(codex_tokens.headers(), timeout=15)
            except Exception as exc:  # fall back to the curated list
                log.warning("codex live model list unavailable: %s", exc)
        if _logged_in(grok_tokens):
            available.add("grok")
            try:
                live["grok"] = grok_client.list_models(grok_tokens.headers(), timeout=15)
            except Exception as exc:  # fall back to the curated list
                log.warning("grok live model list unavailable: %s", exc)
        return model_catalog(available, live)

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

        completion_id = _new_completion_id()
        created = int(time.time())
        provider = route_provider(req.model, default=cfg.default_provider)

        if provider == CODEX:
            return _codex_chat(
                cfg, codex_tokens, req,
                completion_id=completion_id, created=created, started=t0,
            )
        if provider == GROK:
            return _grok_chat(cfg, grok_tokens, req, raw, started=t0)

        # ── Anthropic (default) ────────────────────────────────────────────
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

    @app.post("/v1/responses")
    def responses(
        raw: Dict[str, Any],
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        auth_err = _check_client_auth(authorization)
        if auth_err is not None:
            return auth_err
        model = raw.get("model", "") if isinstance(raw, dict) else ""
        provider = route_provider(model, default=cfg.default_provider)
        if provider == CODEX:
            return _codex_responses(cfg, codex_tokens, raw)
        if provider == GROK:
            return _grok_responses(cfg, grok_tokens, raw)
        return _error_response(
            400,
            f"/v1/responses serves Responses-native providers (Codex/Grok); "
            f"model '{model}' routed to '{provider}'.",
            "invalid_request_error", "unsupported_model",
        )

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


# ── Codex (ChatGPT-subscription) handlers ────────────────────────────────────

def _classify_codex_error(exc: Exception):
    """Map a Codex transport/auth exception to (http_status, openai_type, code)."""
    if isinstance(exc, codex_auth.TokenError):
        return 401, "authentication_error", "oauth_token_unavailable"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 401:
            return status, "authentication_error", "invalid_oauth_token"
        if status == 403:
            return status, "permission_error", "subscription_not_entitled"
        if status == 429:
            return status, "rate_limit_error", "rate_limit_exceeded"
        if 400 <= status < 500:
            return status, "invalid_request_error", None
        return status, "api_error", None
    return 502, "api_error", "upstream_error"


def _codex_stream_sse(
    body: Dict[str, Any],
    *,
    auth_headers: Dict[str, str],
    cfg: Config,
    model: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    started: Optional[float] = None,
) -> Iterator[str]:
    """Yield OpenAI SSE chunks for a streaming Codex chat completion."""
    try:
        events = codex_client.stream_events(
            body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
        )
        for chunk in codex_stream_mapping.responses_events_to_openai_chunks(
            events,
            model=model,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            include_reasoning=cfg.include_reasoning,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        ms = int((time.monotonic() - started) * 1000) if started else -1
        log.info("← 200 (stream) provider=codex model=%s %dms", model, ms)
    except Exception as exc:
        _status, etype, code = _classify_codex_error(exc)
        log.warning("codex stream error: %s: %s", etype, exc)
        err = {"error": {"message": str(exc), "type": etype, "param": None, "code": code}}
        yield f"data: {json.dumps(err)}\n\n"
    yield "data: [DONE]\n\n"


def _codex_chat(cfg: Config, tokens, req: ChatCompletionRequest, *, completion_id, created, started):
    """Handle a Codex-routed /v1/chat/completions request (stream + non-stream)."""
    try:
        auth_headers = tokens.headers()
    except codex_auth.TokenError as exc:
        log.warning("401 codex token unavailable: %s", exc)
        return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")

    body = codex_request_mapping.build_responses_body(
        req,
        default_model=cfg.codex_default_model,
        default_reasoning_effort=cfg.default_reasoning_effort,
        stream=req.stream,
    )
    resolved_model = body.get("model", req.model)
    log.info("→ POST /v1/chat/completions provider=codex model=%s stream=%s", resolved_model, req.stream)

    if req.stream:
        return StreamingResponse(
            _codex_stream_sse(
                body, auth_headers=auth_headers, cfg=cfg, model=resolved_model,
                completion_id=completion_id, created=created,
                include_usage=req.wants_usage(), started=started,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        final = codex_client.collect_final(
            codex_client.stream_events(
                body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
            )
        )
    except Exception as exc:
        status, etype, code = _classify_codex_error(exc)
        log.warning("← %s codex %s (%dms)", status, etype, int((time.monotonic() - started) * 1000))
        return _error_response(status, str(exc), etype, code)

    if final is None:
        return _error_response(
            502, "Codex backend returned no completed response.", "api_error", "upstream_error"
        )
    out = codex_response_mapping.responses_to_openai(
        final, model=resolved_model, completion_id=completion_id, created=created,
        include_reasoning=cfg.include_reasoning,
    )
    log.info("← 200 provider=codex model=%s %dms", resolved_model, int((time.monotonic() - started) * 1000))
    return JSONResponse(content=out)


def _codex_responses(cfg: Config, tokens, raw: Dict[str, Any]):
    """Native /v1/responses passthrough to the Codex subscription backend."""
    try:
        auth_headers = tokens.headers()
    except codex_auth.TokenError as exc:
        return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")
    body = dict(raw) if isinstance(raw, dict) else {}
    body["store"] = False  # the subscription backend rejects stored responses
    wants_stream = bool(body.get("stream"))
    log.info("→ POST /v1/responses provider=codex model=%s stream=%s", body.get("model"), wants_stream)

    if wants_stream:
        def _gen() -> Iterator[str]:
            try:
                for ev in codex_client.stream_events(
                    body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
                ):
                    yield f"data: {json.dumps(ev)}\n\n"
            except Exception as exc:
                _s, etype, code = _classify_codex_error(exc)
                yield f"data: {json.dumps({'error': {'message': str(exc), 'type': etype, 'code': code}})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        final = codex_client.collect_final(
            codex_client.stream_events(body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds)
        )
    except Exception as exc:
        status, etype, code = _classify_codex_error(exc)
        return _error_response(status, str(exc), etype, code)
    if final is None:
        return _error_response(
            502, "Codex backend returned no completed response.", "api_error", "upstream_error"
        )
    return JSONResponse(content=final)


# ── Grok (SuperGrok subscription) handlers ───────────────────────────────────

def _classify_grok_error(exc: Exception):
    """Map a Grok transport/auth exception to (http_status, openai_type, code)."""
    if isinstance(exc, grok_auth.TokenError):
        return 401, "authentication_error", "oauth_token_unavailable"
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status == 401:
            return status, "authentication_error", "invalid_oauth_token"
        if status == 403:
            return status, "permission_error", "subscription_not_entitled"
        if status == 429:
            return status, "rate_limit_error", "rate_limit_exceeded"
        if 400 <= status < 500:
            return status, "invalid_request_error", None
        return status, "api_error", None
    return 502, "api_error", "upstream_error"


def _grok_model(requested: str, default_model: str) -> str:
    return requested if (requested or "").lower().startswith("grok") else default_model


def _grok_chat(cfg: Config, tokens, req: ChatCompletionRequest, raw: Dict[str, Any], *, started=None):
    """Grok-routed /v1/chat/completions — OpenAI-compatible passthrough to api.x.ai."""
    try:
        auth_headers = tokens.headers()
    except grok_auth.TokenError as exc:
        log.warning("401 grok token unavailable: %s", exc)
        return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")

    body = dict(raw) if isinstance(raw, dict) else {}
    body["model"] = _grok_model(req.model, cfg.grok_default_model)
    log.info("→ POST /v1/chat/completions provider=grok model=%s stream=%s", body["model"], req.stream)

    if req.stream:
        def _gen() -> Iterator[bytes]:
            try:
                yield from grok_client.stream_raw(
                    "/chat/completions", body, auth_headers=auth_headers,
                    timeout=cfg.request_timeout_seconds,
                )
            except Exception as exc:
                _s, etype, code = _classify_grok_error(exc)
                log.warning("grok stream error: %s: %s", etype, exc)
                err = {"error": {"message": str(exc), "type": etype, "param": None, "code": code}}
                yield f"data: {json.dumps(err)}\n\n".encode()

        return StreamingResponse(
            _gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        out = grok_client.post_json(
            "/chat/completions", body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
        )
    except Exception as exc:
        status, etype, code = _classify_grok_error(exc)
        log.warning("← %s grok %s", status, etype)
        return _error_response(status, str(exc), etype, code)
    log.info("← 200 provider=grok model=%s", body["model"])
    return JSONResponse(content=out)


def _grok_responses(cfg: Config, tokens, raw: Dict[str, Any]):
    """Grok-routed /v1/responses — passthrough to api.x.ai/v1/responses."""
    try:
        auth_headers = tokens.headers()
    except grok_auth.TokenError as exc:
        return _error_response(401, str(exc), "authentication_error", "oauth_token_unavailable")
    body = dict(raw) if isinstance(raw, dict) else {}
    if not (str(body.get("model") or "").lower().startswith("grok")):
        body["model"] = cfg.grok_default_model
    wants_stream = bool(body.get("stream"))
    log.info("→ POST /v1/responses provider=grok model=%s stream=%s", body.get("model"), wants_stream)

    if wants_stream:
        def _gen() -> Iterator[bytes]:
            try:
                yield from grok_client.stream_raw(
                    "/responses", body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
                )
            except Exception as exc:
                _s, etype, code = _classify_grok_error(exc)
                err = {"error": {"message": str(exc), "type": etype, "param": None, "code": code}}
                yield f"data: {json.dumps(err)}\n\n".encode()

        return StreamingResponse(
            _gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        out = grok_client.post_json(
            "/responses", body, auth_headers=auth_headers, timeout=cfg.request_timeout_seconds
        )
    except Exception as exc:
        status, etype, code = _classify_grok_error(exc)
        return _error_response(status, str(exc), etype, code)
    return JSONResponse(content=out)


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
