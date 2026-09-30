"""Helpers shared by the engine tests (on sys.path via pytest's `pythonpath`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vigilans.config import RunConfig

REPO = Path(__file__).resolve().parents[2]
OBSERVATION_FIXTURES = REPO / "contract" / "fixtures" / "observation"


def fixture(kind: str, name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((OBSERVATION_FIXTURES / kind / f"{name}.json").read_text("utf-8"))
    return loaded


def offline_config(**changes: Any) -> RunConfig:
    """An offline run: as fast as possible, no server, no lingering."""
    config = RunConfig(rate=None, listen=None, linger=False)
    for key, value in changes.items():
        setattr(config, key, value)
    return config
