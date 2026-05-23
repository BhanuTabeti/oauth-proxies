"""Shim for hermes' ``hermes_constants`` module.

The adapter imports ``get_hermes_home`` at module top to compute
``_HERMES_OAUTH_FILE`` (``~/.hermes/.anthropic_oauth.json``). This proxy reuses
Claude Code's own credential store and never exercises the hermes-native OAuth
file path, so the value is effectively unused — but the import must resolve.
"""
import os
from pathlib import Path


def get_hermes_home() -> Path:
    """Return the directory hermes would use for its managed files."""
    base = os.environ.get("HERMES_HOME")
    return Path(base) if base else Path.home() / ".hermes"
