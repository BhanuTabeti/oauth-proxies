# oauth-proxies — OpenAI-compatible Claude (OAuth) proxy

A small, **local, single-user** HTTP server that speaks the OpenAI
`/v1/chat/completions` API and forwards to `api.anthropic.com` using **your own
Claude Code OAuth / subscription token**. It vendors hermes-agent's
`anthropic_adapter.py` for the OpenAI→Anthropic request translation + OAuth
client identity, and adds the missing **Anthropic→OpenAI response translation**
(streaming + non-streaming).

## Scope & ToS boundary

- **In scope:** localhost, single user, *your own* subscription token (the
  narrowly-sanctioned "reuse your own `claude` login on your own machine" case).
- **Out of scope (deliberately):** multi-tenant/serverless deployment, fanning a
  subscription out to arbitrary apps/hosts (the grey area flagged in prior
  research), API-key auth, Bedrock/Azure/Kimi/MiniMax branches, embeddings,
  legacy `/v1/completions`, persistent request logging.
- The OAuth path works only on Anthropic plans that permit it and bills against
  the subscription/overage lane; tool-carrying requests have historically been
  reclassified into overage. This proxy does not change any of that.

## Architecture

```
oauth_proxy/
  app.py              # FastAPI app + routes (/v1/chat/completions, /v1/models, /health)   [integration]
  auth.py             # TokenProvider: OAuth token resolve/refresh + client build          [Agent C]
  request_mapping.py  # OpenAI request -> adapter.build_anthropic_kwargs()                 [integration]
  response_mapping.py # Anthropic Message -> OpenAI ChatCompletion (non-stream)            [Agent A]
  stream_mapping.py   # Anthropic events -> OpenAI chat.completion.chunk (stream)          [Agent B]
  models.py           # pydantic request schemas + model catalog
  config.py           # env-var config
  _vendor/
    anthropic_adapter.py  # VERBATIM copy from hermes-agent
    hermes_constants.py / utils.py / tools/schema_sanitizer.py / tools/lazy_deps.py  # shims
tests/                # pytest; converters are pure (dict->dict), tested without network
```

`_vendor/__init__.py` puts `_vendor/` on `sys.path` so the unedited adapter
resolves its hermes-internal imports against the shims. Access it via
`from oauth_proxy._vendor import adapter`.

## Request flow

1. `POST /v1/chat/completions` validated by `models.ChatCompletionRequest`.
2. `request_mapping` calls `adapter.build_anthropic_kwargs(model, messages,
   tools, max_tokens, reasoning_config, tool_choice, is_oauth=True,
   base_url=None)`. `is_oauth=True` triggers the Claude Code system prefix +
   `mcp_` tool-name prefixing the subscription token requires.
3. `auth.TokenProvider.build_client()` builds the OAuth `anthropic.Anthropic`.
4. `client.messages.create(**kwargs)` (non-stream) or `stream=True`.

## Response flow (the new code)

- **Non-stream** (`response_mapping`): map `text`→`content`, `tool_use`→
  `tool_calls` (`arguments = json.dumps(input)`), `thinking`→optional
  `reasoning_content`. **Strip leading `mcp_`** from tool names (the adapter
  added it for OAuth; clients expect their original name). `stop_reason` map:
  `end_turn|stop_sequence|pause_turn`→`stop`, `max_tokens`→`length`,
  `tool_use`→`tool_calls`, `refusal`→`content_filter`. usage:
  `input_tokens/output_tokens`→`prompt_tokens/completion_tokens/total_tokens`.
- **Stream** (`stream_mapping`): state machine mapping Anthropic content-block
  index → OpenAI tool-call index. `text_delta`→`delta.content`;
  `tool_use` start + `input_json_delta`→`delta.tool_calls[k]`; final
  `message_delta.stop_reason`→`finish_reason`. App appends `data: [DONE]`.

## Auth & refresh (`auth.TokenProvider`)

Prefer the refreshable Claude Code credential store
(`adapter.read_claude_code_credentials()` → keychain / `~/.claude/.credentials.json`);
validate with `adapter.is_claude_code_token_valid()`; refresh when expired;
fall back to `adapter.resolve_anthropic_token()` for env-provided tokens.
Assert `adapter._is_oauth_token(token)`; otherwise raise `TokenError` advising
`claude setup-token`. Cache in-process; re-resolve near expiry.

## Config (env)

`PROXY_HOST`/`PROXY_PORT` (127.0.0.1:8787), `PROXY_API_KEY` (optional client
shared secret), `DEFAULT_MODEL` (fallback for non-Claude model names),
`DEFAULT_REASONING_EFFORT` (`off`), `PROXY_INCLUDE_REASONING`,
`PROXY_REQUEST_TIMEOUT`.

## Testing

Converters are pure `dict -> dict` / `iter -> iter`, tested directly with
recorded Anthropic payloads (no token, no network). Endpoint tests mock the
anthropic client. TDD: tests written before implementation in each module.
