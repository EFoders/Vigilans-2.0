"""Schema validation for every contract in the package.

The JSON Schemas in ``contract/schemas/`` are the single source of truth. This module
only loads them and reports what they say, with two deliberate additions:

- **Dispatch on the discriminator.** A top-level ``oneOf`` failure reads "is not valid
  under any of the given schemas", which tells an adapter author nothing. So the
  ``schema`` field is checked first, then ``kind`` (observations) or ``type`` (picture
  messages) picks the one definition the record claims to be, and the errors reported are
  that definition's.
- **Non-finite numbers are rejected.** Python's ``json`` accepts ``NaN`` and
  ``Infinity``, and JSON Schema's ``minimum`` passes a NaN because every comparison with it
  is false. A NaN bearing would sail through the schema and poison a fix.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError, best_match

OBSERVATION_V1 = "observation.v1"
PICTURE_V0 = "picture.v0"
LIBRARY_V1 = "library.v1"
SCENARIO_V2 = "scenario.v2"
#: Scenario schemas scenario.v2 accepts: v1 files are v2 files without the v2 additions.
SCENARIO_SCHEMAS: tuple[str, ...] = ("scenario.v1", "scenario.v2")

#: Every contract this package can validate, by the name records carry in ``schema``.
CONTRACTS: tuple[str, ...] = (OBSERVATION_V1, PICTURE_V0, LIBRARY_V1, SCENARIO_V2)

OBSERVATION_KINDS: dict[str, str] = {
    "bearing": "bearing_observation",
    "position": "position_observation",
    "coverage": "coverage_record",
    "occupancy": "occupancy_record",
}

#: The kinds that report a signal, as opposed to describing a sensor.
SIGNAL_KINDS: frozenset[str] = frozenset({"bearing", "position"})

PICTURE_TYPES: tuple[str, ...] = ("hello", "snapshot", "delta", "cot", "heartbeat", "notice")


@dataclass(frozen=True, slots=True)
class Issue:
    """One reason a record is invalid. ``path`` is a JSON Pointer into the record."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message}"


@dataclass(frozen=True, slots=True)
class Result:
    """The outcome of validating one record."""

    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        """All issues on one line, for a log or a notice."""
        return "; ".join(str(issue) for issue in self.issues)


class ContractJSONError(ValueError):
    """Text that is not strict JSON: malformed, or carrying NaN or Infinity."""


def parse_json(text: str | bytes) -> Any:
    """Parse strict JSON. ``NaN``, ``Infinity`` and ``-Infinity`` are errors, not numbers."""

    def _reject(token: str) -> Any:
        raise ContractJSONError(f"{token} is not a JSON number")

    try:
        return json.loads(text, parse_constant=_reject)
    except json.JSONDecodeError as error:
        raise ContractJSONError(f"not JSON: {error}") from error


def schema_dir() -> Path:
    """Where the schemas are: inside the installed wheel, or the repository's ``schemas/``.

    An editable install does not copy the schemas into the package, so fall back to the
    directory they are maintained in. Both paths hold the same files.
    """
    packaged = resources.files("vigilans_contract").joinpath("schemas")
    if packaged.is_dir():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[2] / "schemas"


@cache
def load_schema(contract: str) -> dict[str, Any]:
    """The JSON Schema for a contract, parsed."""
    if contract not in CONTRACTS:
        raise KeyError(f"unknown contract {contract!r}; expected one of {', '.join(CONTRACTS)}")
    path = schema_dir() / f"{contract}.schema.json"
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@cache
def _validator(contract: str, definition: str | None) -> Draft7Validator:
    schema = load_schema(contract)
    if definition is not None:
        # A $ref beside definitions resolves against this document, so a single
        # definition can be validated with every other definition still in reach.
        schema = {"$ref": f"#/definitions/{definition}", "definitions": schema["definitions"]}
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema, format_checker=Draft7Validator.FORMAT_CHECKER)


def date_time_format_is_checked() -> bool:
    """Whether ``format: date-time`` is enforced. It needs ``rfc3339-validator`` installed."""
    return not _validator(PICTURE_V0, "utc_time").is_valid("2026-02-30T00:00:00Z")


def _pointer(parts: Iterable[Any]) -> str:
    return "".join(f"/{str(part).replace('~', '~0').replace('/', '~1')}" for part in parts)


def _describe(error: ValidationError) -> Issue:
    # For a oneOf/anyOf that failed every branch, the branch closest to matching says
    # more than "not valid under any of the given schemas".
    if error.validator in ("oneOf", "anyOf") and error.context:
        inner = best_match(error.context)
        return Issue(_pointer(inner.absolute_path), inner.message)
    return Issue(_pointer(error.absolute_path), error.message)


def _schema_issues(validator: Draft7Validator, instance: Any) -> list[Issue]:
    errors = sorted(validator.iter_errors(instance), key=lambda e: list(map(str, e.absolute_path)))
    seen: set[Issue] = set()
    issues: list[Issue] = []
    for error in errors:
        issue = _describe(error)
        if issue not in seen:
            seen.add(issue)
            issues.append(issue)
    return issues


def non_finite_issues(value: Any, path: str = "") -> list[Issue]:
    """Every NaN or infinity anywhere in a parsed record."""
    issues: list[Issue] = []
    if isinstance(value, float) and not math.isfinite(value):
        issues.append(Issue(path, f"{value!r} is not a finite number"))
    elif isinstance(value, Mapping):
        for key, item in value.items():
            issues.extend(non_finite_issues(item, f"{path}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            issues.extend(non_finite_issues(item, f"{path}/{index}"))
    return issues


def _validate(
    record: Any,
    contract: str,
    discriminator: str | None,
    definitions: Mapping[str, str],
    semantic: Callable[[Mapping[str, Any]], list[Issue]],
) -> Result:
    if not isinstance(record, Mapping):
        return Result((Issue("", f"a {contract} record is a JSON object, not {type(record).__name__}"),))
    # Wording matches Videns' validator where the two overlap, so the shared fixtures'
    # `expect` strings hold in both languages.
    if record.get("schema") != contract:
        found = json.dumps(record.get("schema"))
        return Result((Issue("/schema", f'unsupported schema {found}; expected "{contract}"'),))

    definition: str | None = None
    if discriminator is not None:
        tag = record.get(discriminator)
        if tag not in definitions:
            noun = "message type" if discriminator == "type" else "observation kind"
            expected = ", ".join(json.dumps(k) for k in definitions)
            return Result(
                (Issue(f"/{discriminator}", f"unknown {noun} {json.dumps(tag)}; expected one of {expected}"),)
            )
        definition = definitions[str(tag)]

    issues = non_finite_issues(record)
    issues.extend(_schema_issues(_validator(contract, definition), record))
    if not issues:
        # Semantic checks assume a schema-valid record; on an invalid one they would
        # only repeat what the schema already said, or crash on a missing field.
        issues.extend(semantic(record))
    return Result(tuple(issues))


def validate_observation(record: Any) -> Result:
    """Validate one ``observation.v1`` record."""
    from vigilans_contract.semantic import observation_issues

    return _validate(record, OBSERVATION_V1, "kind", OBSERVATION_KINDS, observation_issues)


def validate_picture(message: Any) -> Result:
    """Validate one ``picture.v0`` message."""
    from vigilans_contract.semantic import picture_issues

    return _validate(message, PICTURE_V0, "type", {t: t for t in PICTURE_TYPES}, picture_issues)


def validate_scenario(scenario: Any) -> Result:
    """Validate a parsed ``scenario.v2`` document (``scenario.v1`` files are accepted as v2)."""
    from vigilans_contract.semantic import scenario_issues

    if not isinstance(scenario, Mapping):
        return Result((Issue("", "a scenario is a mapping"),))
    if scenario.get("schema") not in SCENARIO_SCHEMAS:
        found = json.dumps(scenario.get("schema"))
        return Result(
            (Issue("/schema", f'unsupported schema {found}; expected "scenario.v2" (or "scenario.v1")'),)
        )
    issues = non_finite_issues(scenario)
    issues.extend(_schema_issues(_validator(SCENARIO_V2, None), scenario))
    if not issues:
        issues.extend(scenario_issues(scenario))
    return Result(tuple(issues))


def validate_library(library: Any) -> Result:
    """Validate a parsed ``library.v1`` document."""
    from vigilans_contract.semantic import library_issues

    return _validate(library, LIBRARY_V1, None, {}, library_issues)
