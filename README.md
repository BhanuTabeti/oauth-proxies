# oauth-proxies

A small, **local, single-user** server that exposes an **OpenAI-compatible
`/v1/chat/completions` API** and forwards requests to Claude using your **Claude
Code OAuth / subscription token**. Point any OpenAI client (aider, Continue,
LibreChat, the `openai` SDK, plain `curl`) at it and talk to Claude.

It includes a vendored Anthropic-Messages adapter (see
`THIRD_PARTY_NOTICES.md` for attribution) for the OpenAI→Anthropic
request translation and the OAuth client identity, and adds the
Anthropic→OpenAI **response** translation (streaming and non-streaming).

## Scope

A **local, single-user** proxy for indie developers who want to use their
**Claude subscription's Agent SDK credit** with OpenAI-compatible tools
(aider, Continue, LibreChat, the `openai` SDK, plain `curl`) — without
routing through Claude Code itself.

Anthropic provides a monthly **Agent SDK credit** on Pro, Max, Team, and
Enterprise plans starting June 15, 2026 that explicitly covers *"Claude
Agent SDK usage in your own projects"* and *"third-party apps built on the
Agent SDK"* — see [Use the Claude Agent SDK with your Claude plan][docs].
This proxy is one way to tap that credit from any OpenAI-compatible client,
locally. When the monthly credit is exhausted, additional usage flows to
your plan's usage credits at standard API rates (if you've enabled them);
otherwise requests pause until the next billing cycle.

[docs]: https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -e '.[dev]' to run tests
```

## Get a token

You need a Claude Code OAuth/subscription token. The proxy reads it from (in
order) the macOS Keychain (`Claude Code-credentials`),
`~/.claude/.credentials.json`, or the `CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_TOKEN`
environment variables. The simplest way to create one:

```bash
claude setup-token        # requires the `claude` CLI installed and logged in
```

When the stored token expires, the proxy refreshes it automatically (if a
refresh token is present).

## Run

```bash
oauth-proxy               # serves on http://127.0.0.1:8787
# or: python -m oauth_proxy.app
```

On startup the server loads a **`.env`** file from the working directory if one
is present (real environment variables take precedence). Put your token there:

```bash
# .env  (gitignored — never commit it)
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

See `.env.example`. `.env` loading happens only when you run the server, not
when the package is imported (so tests never pick up a developer's `.env`).

### Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `PROXY_HOST` | `127.0.0.1` | Bind host |
| `PROXY_PORT` | `8787` | Bind port |
| `PROXY_API_KEY` | _(unset)_ | If set, clients must send `Authorization: Bearer <key>` |
| `DEFAULT_MODEL` | `claude-opus-4-7` | Substituted when a client requests a non-Claude model (e.g. `gpt-4o`) |
| `DEFAULT_REASONING_EFFORT` | `off` | `off`/`low`/`medium`/`high`/`xhigh`/`max` — extended-thinking effort |
| `PROXY_INCLUDE_REASONING` | `false` | Surface Claude thinking as a non-standard `reasoning_content` field |
| `PROXY_REQUEST_TIMEOUT` | `900` | Upstream read timeout (seconds) |

## Use it

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-opus-4-7","messages":[{"role":"user","content":"Say hi"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
resp = client.chat.completions.create(
    model="claude-opus-4-7",
    messages=[{"role": "user", "content": "Say hi"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

Endpoints: `POST /v1/chat/completions` (stream + non-stream),
`GET /v1/models`, `GET /health`.

## Develop

```bash
pip install -e '.[dev]'
pytest -q
```

Architecture and design decisions: see [DESIGN.md](DESIGN.md). The converters
(`response_mapping.py`, `stream_mapping.py`) are pure `dict -> dict` functions,
tested without any network or token.

## Not included (by design)

API-key auth, multi-user/serverless deployment, Bedrock/Azure/Kimi/MiniMax
endpoints, embeddings, the legacy `/v1/completions` route.
