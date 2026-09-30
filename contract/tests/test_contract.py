"""Phase 0 gate: the contract validates both observation flavours, and rejects what it must.

Every fixture under ``fixtures/`` runs here. Invalid fixtures must fail *for the stated
reason*: the validator's output has to contain the fixture's ``expect`` text, so a fixture
that breaks for some other reason is caught rather than counted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from vigilans_contract import (
    CONTRACTS,
    OBSERVATION_KINDS,
    ContractJSONError,
    date_time_format_is_checked,
    load_schema,
    parse_json,
    schema_dir,
    validate_library,
    validate_observation,
    validate_picture,
)

pytestmark = pytest.mark.contract

CONTRACT_DIR = Path(__file__).resolve().parents[1]
FIXTURES = CONTRACT_DIR / "fixtures" / "observation"
REPO = CONTRACT_DIR.parent


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


VALID = sorted((FIXTURES / "valid").glob("*.json"))
INVALID = sorted((FIXTURES / "invalid").glob("*.json"))


def test_there_are_fixtures() -> None:
    assert len(VALID) >= 10
    assert len(INVALID) >= 20


def test_every_kind_is_covered() -> None:
    kinds = {_load(path)["kind"] for path in VALID}
    assert kinds == set(OBSERVATION_KINDS)


@pytest.mark.parametrize("path", VALID, ids=lambda p: p.stem)
def test_valid_fixture_validates(path: Path) -> None:
    result = validate_observation(_load(path))
    assert result.ok, result.summary()


@pytest.mark.parametrize("path", INVALID, ids=lambda p: p.stem)
def test_invalid_fixture_fails_for_its_stated_reason(path: Path) -> None:
    fixture = _load(path)
    result = validate_observation(fixture["message"])
    assert not result.ok, f"{path.stem} validated, but: {fixture['reason']}"
    assert fixture["expect"] in result.summary(), (
        f"{path.stem} failed, but not for its stated reason ({fixture['reason']}). "
        f"Expected {fixture['expect']!r} in: {result.summary()}"
    )


def test_date_time_format_is_enforced() -> None:
    # jsonschema skips `format` silently when rfc3339-validator is missing.
    assert date_time_format_is_checked()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_parse_json_rejects_non_finite_numbers(token: str) -> None:
    with pytest.raises(ContractJSONError, match=token.lstrip("-")):
        parse_json(f'{{"bearing_deg": {token}}}')


def test_a_nan_that_got_past_the_parser_is_still_rejected() -> None:
    # json.loads would accept this; JSON Schema's `minimum` passes NaN because every
    # comparison with it is false. The validator must not.
    record = _load(FIXTURES / "valid" / "bearing-measured.json")
    record["bearing_deg"] = float("nan")
    result = validate_observation(record)
    assert not result.ok
    assert "/bearing_deg" in result.summary()
    assert "finite" in result.summary()


def test_not_an_object() -> None:
    assert not validate_observation([1, 2, 3]).ok


def test_errors_name_the_field_not_the_branch() -> None:
    record = _load(FIXTURES / "valid" / "bearing-measured.json")
    del record["bearing_uncertainty"]
    summary = validate_observation(record).summary()
    assert "not valid under any of the given schemas" not in summary
    assert "bearing_uncertainty" in summary


@pytest.mark.parametrize("contract", CONTRACTS)
def test_schemas_are_draft7_and_load(contract: str) -> None:
    schema = load_schema(contract)
    assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
    assert schema["$id"] == f"urn:vigilans:contract:{contract}"


def test_schema_dir_is_the_maintained_one_in_a_checkout() -> None:
    assert (schema_dir() / "observation.v1.schema.json").is_file()


# --- picture.v0 --------------------------------------------------------------------------


def _videns_dir() -> Path | None:
    candidates = [os.environ.get("VIDENS_DIR"), REPO / "videns", REPO.parent / "Videns"]
    for candidate in candidates:
        if candidate and (Path(candidate) / "contract" / "picture.v0.schema.json").is_file():
            return Path(candidate)
    return None


VIDENS = _videns_dir()
VIDENS_INVALID = sorted((VIDENS / "fixtures" / "invalid").glob("*.json")) if VIDENS else []


@pytest.mark.skipif(VIDENS is None, reason="no Videns checkout beside this one; set VIDENS_DIR")
def test_picture_schema_matches_videns() -> None:
    assert VIDENS is not None
    ours = load_schema("picture.v0")
    theirs = _load(VIDENS / "contract" / "picture.v0.schema.json")
    assert ours == theirs, (
        "contract/schemas/picture.v0.schema.json has drifted from Videns' copy. "
        "picture.v0 is unstable and Videns is ahead of it: copy theirs, then fix Vigilans."
    )


@pytest.mark.skipif(not VIDENS_INVALID, reason="no Videns invalid picture fixtures available")
@pytest.mark.parametrize("path", VIDENS_INVALID, ids=lambda p: p.stem)
def test_videns_invalid_picture_fixtures_fail_here_too(path: Path) -> None:
    # VIDENS_SPEC.md section 10: the same fixtures run against both languages' validators,
    # so they cannot disagree about what a valid picture is.
    fixture = _load(path)
    result = validate_picture(fixture["message"])
    assert not result.ok, f"{path.stem} validated in Python, but: {fixture['reason']}"
    assert fixture["expect"] in result.summary(), (
        f"{path.stem}: expected {fixture['expect']!r} in: {result.summary()}"
    )


def _heartbeat(t: str) -> dict[str, Any]:
    return {"schema": "picture.v0", "type": "heartbeat", "run_id": "r", "seq": 1, "t": t, "wall_t": t}


def test_picture_heartbeat() -> None:
    assert validate_picture(_heartbeat("2026-01-01T00:00:00Z")).ok
    assert not validate_picture(_heartbeat("2026-01-01T00:00:00+00:00")).ok


def test_picture_unknown_type_is_named() -> None:
    message = _heartbeat("2026-01-01T00:00:00Z") | {"type": "chatter"}
    assert "/type" in validate_picture(message).summary()


# --- library.v1 --------------------------------------------------------------------------


def _library(**changes: Any) -> dict[str, Any]:
    cls = {
        "class_id": "syn.a",
        "label": "synthetic net radio",
        "features": {"freq_hz": {"min": 400e6, "max": 410e6}},
    }
    library: dict[str, Any] = {
        "schema": "library.v1",
        "name": "t",
        "version": "1",
        "background": {"freq_hz": {"min": 20e6, "max": 6e9}, "bandwidth_hz": {"min": 1e3, "max": 40e6}},
        "classes": [cls],
    }
    for key, value in changes.items():
        cls[key] = value
    return library


def test_library_minimal() -> None:
    assert validate_library(_library()).ok, validate_library(_library()).summary()


@pytest.mark.parametrize(
    ("changes", "expect"),
    [
        ({"label": "likely synthetic net radio"}, "adds the hedge"),
        ({"features": {"freq_hz": {"min": 5.0, "max": 1.0}}}, "greater than max"),
        ({"features": {"duration_s": {"min": 1.0, "max": 2.0}}}, "background has no duration_s"),
        ({"features": {"freq_hz": {"min": 0.0, "max": 1e6}}}, "log scale"),
        ({"parent": "syn.missing"}, "no class"),
        ({"parent": "syn.a"}, "cycle"),
        ({"description": "for targeting"}, "targeting language"),
        ({"affiliation": {"identity": "hostile", "reasons": []}}, "reasons"),
        ({"symbol": {"sidc_template": "SHGP-----------"}}, "sidc_template"),
        ({"rule": "lambda e: True"}, "rule"),
    ],
    ids=[
        "hedged-label",
        "inverted-range",
        "feature-without-background",
        "zero-on-a-log-scale",
        "unknown-parent",
        "self-parent",
        "forbidden-word",
        "affiliation-no-reasons",
        "fixed-identity",
        "executable-rule",
    ],
)
def test_library_rejects(changes: dict[str, Any], expect: str) -> None:
    result = validate_library(_library(**changes))
    assert not result.ok
    assert expect in result.summary()


def test_library_duplicate_class_ids() -> None:
    library = _library()
    library["classes"].append(dict(library["classes"][0]))
    assert "duplicate id" in validate_library(library).summary()
