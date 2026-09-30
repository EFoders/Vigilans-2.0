"""Loading a classification library (spec §9.1). Data only, validated, and bounded.

The engine is open; the knowledge is not. The repository ships a synthetic library; a
private one is loaded at runtime from a path **outside** the repository tree, and the
loader refuses one found inside it, because a private library inside the tree is one
``git add .`` from a permanent disclosure (rule 6).

Nothing a library contains is executed. It is parsed as YAML or JSON with a safe loader,
validated against ``library.v1``, and used as data.

Classification itself is Phase 5. Until then a loaded library is reported in the run
header and the picture's hello, and nothing is classified against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vigilans_contract import validate_library

#: The repository root. A private library may not live anywhere under it.
REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_LIBRARY = REPO_ROOT / "vigilans" / "libraries" / "synthetic.yaml"


class LibraryError(ValueError):
    """A library that cannot be used, with the reason. The run degrades to unclassified."""


@dataclass(frozen=True, slots=True)
class Library:
    name: str
    version: str
    private: bool
    path: Path
    classes: tuple[dict[str, Any], ...]
    #: The span each feature can take for anything at all: "none of the above".
    background: dict[str, Any] = field(default_factory=dict)

    def picture_ref(self) -> dict[str, Any]:
        return {"available": True, "name": self.name, "version": self.version, "private": self.private}

    def describe(self) -> str:
        where = "private, outside the repository" if self.private else "repository, not private"
        return f"{self.name} {self.version} ({len(self.classes)} classes; {where})"


UNAVAILABLE: dict[str, Any] = {"available": False}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def load_library(path: Path, *, private: bool, repo_root: Path = REPO_ROOT) -> Library:
    if private and _inside(path, repo_root):
        raise LibraryError(
            f"{path} is marked private but lives inside the repository ({repo_root}). Private "
            f"libraries are loaded from outside the tree and never committed (rule 6). Move it."
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LibraryError(f"cannot read library {path}: {error}") from error
    try:
        document = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise LibraryError(
            f"library {path} is not valid {path.suffix.lstrip('.') or 'YAML'}: {error}"
        ) from error
    result = validate_library(document)
    if not result.ok:
        # A private library's content must never reach a log line (spec §9.1), so its
        # problems are reported by location only.
        detail = "; ".join(issue.path or "/" for issue in result.issues) if private else result.summary()
        raise LibraryError(f"library {path} does not meet library.v1: {detail}")
    return Library(
        name=document["name"],
        version=document["version"],
        private=private,
        path=path,
        classes=tuple(document["classes"]),
        background=dict(document["background"]),
    )
