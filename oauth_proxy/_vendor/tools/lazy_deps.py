"""Shim for the ``tools.lazy_deps`` module the vendored adapter expects.

Upstream, ``ensure()`` lazily pip-installs provider extras on first use. Here
every runtime dependency is declared in ``pyproject.toml`` and installed up
front, so this is a no-op and the adapter's ``_get_anthropic_sdk()`` falls
through to a plain ``import anthropic``.
"""


def ensure(*args, **kwargs) -> None:
    return None
