# Codex (ChatGPT subscription) OAuth provider — design

**Date:** 2026-05-31
**Branch:** `feat/codex-oauth-subscription`
**Status:** Design — pending user review, then implementation plan.

## Goal

Add a second upstream provider to the proxy: **OpenAI Codex / ChatGPT subscription via
OAuth**. A user with a ChatGPT Plus/Pro subscription should be able to point any
OpenAI-compatible client at this proxy and have requests served against their
**subscription** (no API key), exactly mirroring how the existing Anthropic path
serves a Claude Code OAuth subscription.

Two client-facing surfaces are exposed:

1. `POST /v1/responses` — native **OpenAI Responses API**, Codex-only. The
   high-fidelity ("best") path: a thin preflight + auth-inject passthrough to the
   Codex Responses backend. Responses-in → Responses-out, minimal translation loss.
2. `POST /v1/chat/completions` — existing endpoint, now **routes by model name**.
   `claude-*` → Anthropic (today's behavior, unchanged). `gpt-*`/`o*`/`codex-*` →
   Codex (translated Chat ↔ Responses). Unknown names → a configurable default
   provider.

This is **additive**. The existing Anthropic OAuth path is preserved; the Codex
path is layered alongside it behind a small provider seam.

## Non-goals

- API-key auth to OpenAI (`api.openai.com` with `OPENAI_API_KEY`). The point is the
  *subscription*, not pay-as-you-go keys.
- Serving Anthropic through `/v1/responses` (would require a Responses→Anthropic→Responses
  translation we don't need). `/v1/responses` is Codex-only.
- Multi-user / serverless / LAN exposure, embeddings, image/audio, the legacy
  `/v1/completions` route — same scope boundary as today (`DESIGN.md`).
- Depending on the `hermes` CLI or `hermes proxy` at runtime (see rationale below).

## Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Endpoint matrix | `/v1/responses` = Codex passthrough; `/v1/chat/completions` = both providers |
| Provider routing (chat) | By model name (`gpt*`/`o*`/`codex*` → Codex; `claude*` → Anthropic; unknown → default) |
| Codex token source | Our **own** OAuth from scratch — not reading `~/.codex/auth.json` or `~/.claude` |
| Codex login flavor | **PKCE loopback** (mirrors the canonical Codex CLI client). Device-code is the documented fallback. |
| Translation code | **Vendor** Hermes' `codex_responses_adapter.py` (MIT + attribution), as we did for the Anthropic adapter |

## Why build, not reuse `hermes proxy`

We evaluated using NousResearch's `hermes proxy` directly. Adversarially-verified
research (2026-05-31) shows it cannot meet either core goal:

- **No Codex/ChatGPT upstream.** Shipped proxy upstreams are only `nous` (Nous
  Portal) and `xai` (Grok), selected via `--provider <nous|xai>`. "ChatGPT Pro" is
  v0.14.0 marketing copy contradicted by the shipped provider list. *(verify: refuted, high confidence)*
- **No Responses API.** `hermes proxy` forwards only `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/models`; `/v1/responses` returns 404.
  The Responses API lives in a *separate* heavyweight component (the Hermes "API
  Server", port 8642) that exposes the full agent, not a subscription passthrough. *(verify: refuted, high confidence)*
- **Heavy, unstable dependency.** Monolithic ~11 MB wheel, 15 exact-pinned deps
  (`openai==2.24.0`, `pydantic==2.13.4`), no proxy-only install, no SemVer
  stability (v0.14.0 → v0.15.0 in 12 days; ~9–10 day cadence).

But Hermes is **plain MIT** (verified against `LICENSE`), so the right move is to
**vendor the translation source we need** with attribution — the same pattern this
repo already uses for `anthropic_adapter.py` — and own a thin OAuth + transport
layer. We gain native `/v1/responses`, the Codex subscription path, and per-request
model routing that `hermes proxy` lacks, with no heavy runtime dependency.

### Evidence to confirm against live code (not yet verified)

The Codex backend specifics below come largely from Hermes GitHub issues + marketing,
**not** core docs; one research agent flagged some sources as unverifiable. Treat as
the *working shape to confirm empirically* during implementation, not gospel:

- Backend URL `https://chatgpt.com/backend-api/codex/responses`
- Headers: `Authorization: Bearer <access_token>`, `chatgpt-account-id`,
  `originator: codex_cli_rs`, `OpenAI-Beta` / version headers
- `chatgpt_account_id` decoded from the `id_token` JWT
- `api_mode = codex_responses`; `"store": false` enforced
- OAuth client identity = the official public Codex CLI `client_id` against
  `auth.openai.com` (PKCE). A privately-registered client will **not** reach
  subscription billing.

These will be pinned down by reading the vendored `codex_responses_adapter.py` and
Hermes' Codex transport/credential code at implementation time, and validated with a
real login.

## Architecture (additive)

```
oauth_proxy/
  app.py                       # +POST /v1/responses; chat route gains model→provider routing; extend /v1/models
  providers/                   # NEW: small provider seam
    __init__.py                #   Provider protocol + registry + route(model) -> provider
    anthropic_provider.py      #   wraps existing auth/request/response/stream mapping (behavior-preserving)
    codex_provider.py          #   NEW Codex/ChatGPT subscription provider
  auth.py                      # existing Anthropic TokenProvider — UNCHANGED
  codex_auth.py                # NEW: own PKCE OAuth login + token store + refresh + account-id extraction
  codex_client.py              # NEW: thin HTTP/SSE client to the Codex Responses backend
  codex_request_mapping.py     # NEW: OpenAI Chat -> Responses input items (wraps vendored adapter)
  codex_response_mapping.py    # NEW: Responses object -> OpenAI ChatCompletion (non-stream)
  codex_stream_mapping.py      # NEW: Responses SSE events -> OpenAI chat.completion.chunk
  request_mapping.py           # existing (Anthropic) — UNCHANGED
  response_mapping.py          # existing (Anthropic) — UNCHANGED
  stream_mapping.py            # existing (Anthropic) — UNCHANGED
  models.py                    # +Responses request schema; +gpt/codex model_catalog entries; model→provider map
  config.py                    # +Codex config block
  _vendor/
    codex_responses_adapter.py # NEW vendored (MIT)
tests/
  test_codex_auth.py           # PKCE URL build, JWT account-id decode, refresh, token cache (mocked I/O)
  test_codex_request_mapping.py# Chat -> Responses input (pure)
  test_codex_response_mapping.py # Responses -> ChatCompletion (pure)
  test_codex_stream_mapping.py # Responses events -> chunks (pure)
  test_routing.py              # model -> provider routing
  test_endpoints.py            # extended: /v1/responses + chat routing (mocked clients)
```

## Component contracts

### `providers/` — the seam

A minimal `Provider` protocol so `app.py` routes without knowing upstream details.
The Anthropic refactor is **behavior-preserving** — the existing pure converters and
`auth.py` are untouched; `AnthropicProvider` simply calls them behind the interface.
Existing tests are the guardrail (`pytest` must stay green throughout).

```python
class Provider(Protocol):
    name: str
    def build_chat_kwargs(self, req, *, cfg) -> dict: ...          # OpenAI ChatCompletionRequest -> upstream kwargs
    def chat_create(self, kwargs, *, stream: bool): ...            # call upstream
    def chat_to_openai(self, upstream_result, *, ...) -> dict: ... # non-stream -> OpenAI ChatCompletion
    def chat_to_openai_stream(self, upstream_events, *, ...) -> Iterator[dict]: ...

def route(model: str, *, default: str) -> str:   # "claude*"->anthropic, "gpt*/o*/codex*"->codex, else default
```

`CodexProvider` additionally exposes the native Responses passthrough used by
`POST /v1/responses` (preflight + auth-inject; stream + non-stream).

### `codex_auth.py` — `CodexTokenProvider`

Mirrors `auth.py`'s shape (in-process cache, refresh near expiry, `TokenError`).

- **Login (PKCE loopback):** `oauth-proxy login codex` opens the browser to
  `auth.openai.com` with PKCE + a localhost redirect, exchanges the code, and stores
  the credential bundle in `~/.oauth-proxy/.codex_oauth.json`
  (`access_token`, `refresh_token`, `id_token`, `account_id`, `expires_at`).
- **`get_token()`** serves the cached access token; refreshes via the token endpoint
  when within the expiry skew; raises a typed re-auth `TokenError` on terminal
  refresh failure (4xx / `invalid_grant`).
- **`account_id()`** = `chatgpt_account_id` decoded from the `id_token` JWT (cached).
- Pure helpers (PKCE pair, auth-URL build, JWT claim decode, expiry math) are unit-tested
  without network/browser.

### `codex_client.py`

Thin client building the request to the Codex Responses backend: base URL +
`Authorization: Bearer` + `chatgpt-account-id` + originator/version headers; SSE
streaming. Kept deliberately small and auditable. (Exact URL/headers confirmed at
implementation per "evidence to confirm" above.)

### Translation modules (pure `dict -> dict` / `iter -> iter`)

- `codex_request_mapping.py`: OpenAI Chat → Responses `input` items (wraps the
  vendored adapter's converter; enforces `store:false`, `instructions`, model-normalize).
- `codex_response_mapping.py`: Responses object → OpenAI `ChatCompletion`
  (text, tool_calls with stable `fc_`/`call_` ids, optional reasoning; usage map;
  finish_reason map).
- `codex_stream_mapping.py`: Responses SSE events
  (`response.output_text.delta`, `response.output_item.done`, …) → OpenAI
  `chat.completion.chunk`. Tested directly with recorded Responses payloads, no network.

### `models.py` / `config.py`

- `models.py`: add a `ResponsesRequest` schema (lenient passthrough); extend
  `model_catalog()` with the gpt/codex models; expose the `model → provider` map.
- `config.py`: add `PROXY_DEFAULT_PROVIDER` (default for unknown models),
  `CODEX_*` defaults (default model, account-id override, request timeout). All
  existing env vars unchanged.

## Request flows

**`POST /v1/responses` (Codex):** validate → `CodexTokenProvider` resolves token +
account id → preflight kwargs → `codex_client` calls backend (stream/non-stream) →
return Responses payload near-verbatim (SSE passes through; app appends `[DONE]`).

**`POST /v1/chat/completions` (Codex branch):** validate → `route(model)` = codex →
`codex_request_mapping` (Chat→Responses) → backend → `codex_response_mapping` /
`codex_stream_mapping` (Responses→Chat) → OpenAI ChatCompletion/chunks.

**`POST /v1/chat/completions` (Anthropic branch):** unchanged from today.

## Error handling

Reuse the existing OpenAI-style error envelope and `_classify_upstream_error`
pattern. Add Codex-specific mapping: 401 → re-auth (`oauth_token_unavailable` /
`invalid_oauth_token`), 403 "Workspace not authorized in this region", 429 →
`rate_limit_error`. Mid-stream errors surface as an SSE error event (as today).

## Testing strategy (TDD)

Tests written before implementation per module. Converters are pure and tested with
recorded Responses payloads (no token, no network). `codex_auth` I/O (keychain/file,
browser, HTTP) is mocked. Endpoint tests mock the Codex client. **The existing test
suite must stay green at every step** (the Anthropic refactor is behavior-preserving).

## Licensing

`_vendor/codex_responses_adapter.py` is a modified copy from Hermes Agent (MIT,
© 2025 Nous Research). Add a Hermes-Codex section to `THIRD_PARTY_NOTICES.md`
alongside the existing Anthropic-adapter entry, with the MIT text and a note on
modifications (identifier neutralization, shim wiring). Update `README.md` and
`DESIGN.md` to document the new provider, endpoints, and config.

## Build sequence (rough; detailed in the implementation plan)

1. Provider seam + behavior-preserving `AnthropicProvider` refactor (tests stay green).
2. Vendor `codex_responses_adapter.py` + attribution + shims.
3. Pure translation modules (TDD): request → response → stream.
4. `codex_auth.py` (PKCE login, store, refresh, account-id) — pure helpers TDD'd, I/O mocked.
5. `codex_client.py` thin transport.
6. `CodexProvider` wiring; `models.py`/`config.py` additions; routing.
7. `app.py`: `/v1/responses` route + chat routing; extend `/v1/models`.
8. Live validation with a real ChatGPT subscription login; confirm the "evidence to
   confirm" items; docs.

## Open questions / risks

- **Backend specifics unverified** (URL, headers, account-id header name, model list).
  Mitigation: confirm against vendored adapter + a real login early (step 8 pulled
  forward if needed).
- **OAuth client acceptance:** the subscription backend only honors the official
  Codex public client identity. If reusing it via our own PKCE flow is rejected,
  fall back to device-code (the Hermes-documented flavor).
- **ToS:** using a ChatGPT subscription token for arbitrary OpenAI-compatible clients
  is grayer than the narrowly-sanctioned "your own login on your own machine" framing
  the Anthropic path leans on. We keep the same scope guard (localhost, single user)
  and document it in `DESIGN.md`'s ToS section.
