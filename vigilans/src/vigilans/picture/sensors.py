"""Sensors in the picture (VIDENS_SPEC.md §6.6), from the observations that name them.

A sensor is where a bearing was measured from, so bearing observations place it. A
position observation names a sensor only if the source distinguishes one, and says
nothing about where it is.

Two things are deliberately **not** inferred:

- **Health.** ``unknown`` until Audiens settles how sources report it (Audiens §11). A
  sensor that has gone quiet and one that has stopped look identical from here, and
  "silent" would be a guess.
- **Affiliation.** ``unknown`` with basis ``default`` unless the operator declared one for
  the source, in configuration or in the scenario that stands in for an operator.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from vigilans.clock import utc_text
from vigilans.observation import AnyObservation, BearingObservation
from vigilans.picture.publisher import Message
from vigilans_contract import IDENTITIES, StandardIdentity, sidc_with_identity

#: MIL-STD-2525C ground equipment, sensor. The identity character is filled per sensor.
SENSOR_SIDC_TEMPLATE = "S*GPES---------"


class AffiliationError(ValueError):
    """A declared affiliation that is not a standard identity."""


@dataclass(frozen=True, slots=True)
class Declaration:
    identity: StandardIdentity
    declared_in: str


def declaration(identity: str, declared_in: str) -> Declaration:
    if identity not in IDENTITIES:
        raise AffiliationError(
            f"{declared_in}: affiliation {identity!r} is not a standard identity; "
            f"expected one of {', '.join(IDENTITIES)}"
        )
    return Declaration(identity, declared_in)


def picture_sensor_id(source_id: str, sensor_id: str | None) -> str:
    """Sensor ids are only unique within a source, so the picture qualifies them."""
    return f"{source_id}:{sensor_id}" if sensor_id else source_id


class SensorBoard:
    def __init__(self, declarations: Mapping[str, Declaration] | None = None) -> None:
        self._declarations = dict(declarations or {})
        self.sensors: dict[str, Message] = {}

    def _affiliation(self, source_id: str) -> tuple[dict[str, object], str]:
        declared = self._declarations.get(source_id)
        if declared is None:
            return {"identity": "unknown", "basis": "default", "reasons": []}, sidc_with_identity(
                SENSOR_SIDC_TEMPLATE, "unknown"
            )
        return (
            {
                "identity": declared.identity,
                "basis": "operator",
                "reasons": [
                    f"Declared {declared.identity.replace('_', ' ')} for {source_id} "
                    f"in {declared.declared_in}."
                ],
            },
            sidc_with_identity(SENSOR_SIDC_TEMPLATE, declared.identity),
        )

    def update(self, observations: Iterable[AnyObservation]) -> list[Message]:
        """Apply a batch; return every sensor that changed, whole."""
        changed: dict[str, Message] = {}
        for observation in observations:
            key = picture_sensor_id(observation.source_id, observation.sensor_id)
            current = self.sensors.get(key)
            if current is None:
                affiliation, sidc = self._affiliation(observation.source_id)
                current = {
                    "sensor_id": key,
                    "source_id": observation.source_id,
                    "label": (observation.sensor_id or observation.source_id)[:32],
                    "health": "unknown",
                    "affiliation": affiliation,
                    "symbol": {"sidc": sidc},
                }
            updated = dict(current)
            last_heard = updated.get("last_heard")
            heard = utc_text(observation.t)
            if last_heard is None or heard > str(last_heard):
                updated["last_heard"] = heard
            if isinstance(observation, BearingObservation):
                position: dict[str, float] = {"lat": observation.sensor_lat, "lon": observation.sensor_lon}
                if observation.sensor_alt_m is not None:
                    position["alt_m"] = observation.sensor_alt_m
                updated["position"] = position
            if updated != current or key not in self.sensors:
                self.sensors[key] = updated
                changed[key] = updated
        return list(changed.values())
