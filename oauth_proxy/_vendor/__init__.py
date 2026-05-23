"""Vendored copy of hermes-agent's ``anthropic_adapter.py`` plus minimal shims.

``anthropic_adapter.py`` is copied **verbatim** from
https://github.com/NousResearch/hermes-agent/blob/main/agent/anthropic_adapter.py

The verbatim file imports several hermes-internal modules
(``hermes_constants``, ``utils``, ``tools.schema_sanitizer``,
``tools.lazy_deps``). We ship minimal local shims for exactly the symbols the
OAuth-subscription code path touches, and place this directory on ``sys.path``
so the unedited file resolves them.

Usage::

    from oauth_proxy._vendor import adapter
    kwargs = adapter.build_anthropic_kwargs(...)

Keeping the file unedited means we can re-vendor a newer copy by replacing a
single file. The trade-off (a copy that can drift from upstream) was an
explicit design choice — see DESIGN.md.
"""
import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
# Front-insert so the verbatim file's bare ``import hermes_constants`` /
# ``from utils import ...`` / ``from tools.schema_sanitizer import ...`` resolve
# to the shims that live beside it in this directory.
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import anthropic_adapter as adapter  # noqa: E402

__all__ = ["adapter"]
