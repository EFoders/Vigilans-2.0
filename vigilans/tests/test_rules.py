"""Hard rules that are cheap to check mechanically.

- The pipeline never imports the simulator (spec §6): only ``vigilans.sources.sim`` may.
- No targeting vocabulary in code, scenarios, libraries or schemas (rule 4).
- The library boundary (spec §9.1): a private library inside the tree is refused.
- Nothing decodes or demodulates content, and nothing transmits (rules 2, 3).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from vigilans.library import BUNDLED_LIBRARY, LibraryError, load_library
from vigilans_contract import forbidden_words

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "vigilans" / "src" / "vigilans"
SIM_IMPORTERS_ALLOWED = {"sources/sim.py"}


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.architecture
def test_the_engine_never_imports_the_hub() -> None:
    # ADR-0010: the hub simulates and knows the truth; the engine only hears it over the
    # network. An import the other way would put the truth one call away from the pipeline.
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if any(name == "vigilans_hub" or name.startswith("vigilans_hub.") for name in _imports(path))
    ]
    assert offenders == []


@pytest.mark.architecture
def test_only_the_sim_input_imports_the_simulator() -> None:
    offenders = []
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("sim/") or relative in SIM_IMPORTERS_ALLOWED:
            continue
        if any(name == "vigilans.sim" or name.startswith("vigilans.sim.") for name in _imports(path)):
            offenders.append(relative)
    assert offenders == [], f"pipeline code importing the simulator: {offenders}"


def _checked_files() -> list[Path]:
    roots = [
        SRC,
        REPO / "vigilans" / "scenarios",
        REPO / "vigilans" / "libraries",
        REPO / "contract" / "schemas",
        REPO / "contract" / "src",
        REPO / "contract" / "fixtures",
    ]
    return [p for root in roots for p in root.rglob("*") if p.suffix in {".py", ".yaml", ".json", ".toml"}]


def test_no_targeting_language() -> None:
    this_rule = {REPO / "contract" / "src" / "vigilans_contract" / "semantic.py"}  # defines the list
    found = {
        str(path.relative_to(REPO)): words
        for path in _checked_files()
        if path not in this_rule and (words := forbidden_words(path.read_text("utf-8")))
    }
    assert found == {}


def test_no_content_decoding_or_transmit_code() -> None:
    pattern = re.compile(r"\b(demodulat\w*|decode_audio|transmit\(|jam\w*\(|spoof\w*\()", re.IGNORECASE)
    hits = [str(p.relative_to(REPO)) for p in SRC.rglob("*.py") if pattern.search(p.read_text("utf-8"))]
    assert hits == []


def test_bundled_library_loads() -> None:
    library = load_library(BUNDLED_LIBRARY, private=False)
    assert library.name == "synthetic" and len(library.classes) >= 3


def test_a_private_library_inside_the_tree_is_refused() -> None:
    with pytest.raises(LibraryError, match="outside the tree"):
        load_library(BUNDLED_LIBRARY, private=True)


def test_a_private_library_outside_the_tree_loads(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(BUNDLED_LIBRARY.read_text("utf-8").replace("name: synthetic", "name: site"), "utf-8")
    library = load_library(path, private=True, repo_root=REPO)
    assert library.private and library.picture_ref()["private"] is True


def test_a_broken_private_library_does_not_leak_its_content(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    path.write_text(
        "schema: library.v1\nname: s\nversion: '1'\n"
        "background: { freq_hz: { min: 1, max: 1000 }, bandwidth_hz: { min: 1, max: 10 } }\nclasses:\n"
        "  - class_id: c\n    label: SECRET-LABEL\n    features: { freq_hz: { min: 777, max: 1 } }\n",
        "utf-8",
    )
    with pytest.raises(LibraryError) as caught:
        load_library(path, private=True, repo_root=REPO)
    assert "777" not in str(caught.value) and "SECRET" not in str(caught.value)
    assert "/classes/0/features/freq_hz" in str(caught.value)


def test_library_is_data_not_code(tmp_path: Path) -> None:
    path = tmp_path / "evil.yaml"
    path.write_text("!!python/object/apply:os.system ['echo nope']\n", "utf-8")
    with pytest.raises(LibraryError):
        load_library(path, private=False)
