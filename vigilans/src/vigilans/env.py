"""Read ``.env`` into the process environment. Carried from the prototype.

The prototype documented ``.env`` as the source of its publish destination and nothing
loaded it, so every run published to a built-in default whatever the file said (spec
§12). A real environment variable always wins over the file.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

DEFAULT_ENV_FILE = Path(".env")

_QUOTES = ("'", '"')


def parse_env(text: str) -> dict[str, str]:
    """Parse ``.env`` content. Blank lines, comments and lines with no ``=`` are skipped."""
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        values[key] = value
    return values


def load_env_file(
    path: Path | None = None, *, environ: MutableMapping[str, str] | None = None
) -> dict[str, str]:
    """Load ``path`` into ``environ`` without overriding what is set. Returns what it applied."""
    into = os.environ if environ is None else environ
    source = DEFAULT_ENV_FILE if path is None else path
    try:
        text = source.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError):
        return {}
    applied: dict[str, str] = {}
    for key, value in parse_env(text).items():
        if key not in into:
            into[key] = value
            applied[key] = value
    return applied
