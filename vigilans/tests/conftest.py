from __future__ import annotations

from typing import Any

import pytest
from _support import fixture


@pytest.fixture
def bearing() -> dict[str, Any]:
    return fixture("valid", "bearing-measured")


@pytest.fixture
def position() -> dict[str, Any]:
    return fixture("valid", "position-ellipse")
