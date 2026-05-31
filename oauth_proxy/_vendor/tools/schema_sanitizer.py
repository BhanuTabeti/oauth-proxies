"""Shim providing ``tools.schema_sanitizer.strip_nullable_unions``.

Anthropic's tool-schema validator rejects nullable unions such as
``anyOf: [{"type": "string"}, {"type": "null"}]`` that Pydantic/MCP commonly
emit for optional fields. Optionality is carried by the parent ``required``
array instead, so we collapse such unions to their single non-null branch
(merging in sibling metadata like ``description``/``default``), recursing
through nested object/array schemas.

The adapter calls this with ``keep_nullable_hint=False`` (the Anthropic
validator does not understand the OpenAPI ``nullable`` extension).
"""
from copy import deepcopy
from typing import Any

_UNION_KEYS = ("anyOf", "oneOf")


def _is_null_schema(s: Any) -> bool:
    """True for a bare ``{"type": "null"}`` branch (no other constraints)."""
    return (
        isinstance(s, dict)
        and s.get("type") == "null"
        and all(k == "type" for k in s.keys())
    )


def strip_nullable_unions(schema: Any, *, keep_nullable_hint: bool = False) -> Any:
    """Collapse nullable unions and recurse through nested schemas."""
    if not isinstance(schema, dict):
        return schema
    schema = deepcopy(schema)

    for key in _UNION_KEYS:
        union = schema.get(key)
        if not isinstance(union, list):
            continue
        non_null = [m for m in union if not _is_null_schema(m)]
        had_null = len(non_null) != len(union)
        if not had_null:
            continue
        if len(non_null) == 1 and isinstance(non_null[0], dict):
            # Single non-null branch → merge it up into this schema, preserving
            # any sibling metadata already present (description/default/title…).
            merged = dict(non_null[0])
            for mk, mv in schema.items():
                if mk in _UNION_KEYS:
                    continue
                merged.setdefault(mk, mv)
            schema = merged
            if keep_nullable_hint:
                schema["nullable"] = True
            break
        else:
            # Multiple non-null branches → keep them, drop only the null branch.
            schema[key] = non_null
            if keep_nullable_hint:
                schema["nullable"] = True

    # Recurse into nested schema locations.
    props = schema.get("properties")
    if isinstance(props, dict):
        schema["properties"] = {
            k: strip_nullable_unions(v, keep_nullable_hint=keep_nullable_hint)
            for k, v in props.items()
        }

    items = schema.get("items")
    if isinstance(items, dict):
        schema["items"] = strip_nullable_unions(items, keep_nullable_hint=keep_nullable_hint)
    elif isinstance(items, list):
        schema["items"] = [
            strip_nullable_unions(it, keep_nullable_hint=keep_nullable_hint) for it in items
        ]

    addl = schema.get("additionalProperties")
    if isinstance(addl, dict):
        schema["additionalProperties"] = strip_nullable_unions(
            addl, keep_nullable_hint=keep_nullable_hint
        )

    for key in _UNION_KEYS:
        union = schema.get(key)
        if isinstance(union, list):
            schema[key] = [
                strip_nullable_unions(m, keep_nullable_hint=keep_nullable_hint) for m in union
            ]

    return schema
