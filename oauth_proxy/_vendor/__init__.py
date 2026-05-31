"""Vendored Anthropic-Messages adapter plus minimal shims.

``anthropic_adapter.py`` is a **modified copy** of an upstream Anthropic-Messages
adapter, used under the MIT License. The file has been adapted to this proxy's
scope: upstream-specific identifiers renamed to neutral names, unused runtime
sanitization removed, and provenance comments neutralized. See
``THIRD_PARTY_NOTICES.md`` in the project root for the original source,
copyright holder, and full license text.

The adapter imports three sibling modules by name at module load —
``_paths``, ``utils``, ``tools.schema_sanitizer``, ``tools.lazy_deps``. Those
files live next to it in this directory and are provided as minimal shims;
this ``__init__.py`` puts the vendor directory on ``sys.path`` so the
adapter's bare imports resolve.

Usage::

    from oauth_proxy._vendor import adapter
    kwargs = adapter.build_anthropic_kwargs(...)
"""
import os
import sys

_VENDOR_DIR = os.path.dirname(os.path.abspath(__file__))
# Front-insert so the adapter's bare ``from _paths import _app_home`` /
# ``from utils import ...`` / ``from tools.schema_sanitizer import ...``
# resolve to the shims living beside it in this directory.
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import anthropic_adapter as adapter  # noqa: E402

__all__ = ["adapter"]
