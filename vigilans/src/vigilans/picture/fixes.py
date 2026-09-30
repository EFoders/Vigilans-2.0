"""Fixes in the picture, until tracking exists (ADR-0008).

Phase 2 locates transmissions but does not yet follow emitters, so a fix is not an entity
in the spec's sense — it claims nothing about identity over time. It is still worth seeing:
the bearing lines, their weights and the region are exactly what developing geolocation
needs to watch. So each fix is published as a **tentative** entity that says, in its
classification reason and meta, that it is a single fix; it carries its evidence; and it is
withdrawn after a short time rather than left looking current.

Symbol: 2525C unknown identity, unknown battle dimension (``SUZP``). Nothing here knows
whether it is on the ground or in the air, and the symbol does not pretend to.
"""

from __future__ import annotations

from typing import Any

from vigilans.clock import utc_text
from vigilans.locate.fix import Fix, Region
from vigilans.observation import ScalarUncertainty

FIX_SIDC = "SUZP-----------"

SINGLE_FIX = (
    "Not classified: a single fix with unreported uncertainty, which cannot join a track, "
    "and classification needs a track."
)


def _assumption(region: Region) -> dict[str, str] | None:
    if not region.assumptions:
        return None
    return {
        "declared_by": "; ".join(sorted({a.declared_by for a in region.assumptions}))[:2000],
        "note": " / ".join(sorted({a.note for a in region.assumptions}))[:2000],
    }


def region_json(region: Region) -> dict[str, Any]:
    out: dict[str, Any] = {"basis": region.basis}
    if region.cov_en_m2 is not None:
        ee, en, nn = region.cov_en_m2
        out["cov_en_m2"] = {"ee": ee, "en": en, "nn": nn}
    elif region.ellipse is not None:
        major, minor, orientation, confidence = region.ellipse
        out["ellipse"] = {
            "semi_major_m": major,
            "semi_minor_m": minor,
            "orientation_deg": orientation % 360.0,
            "confidence": confidence,
        }
    assumption = _assumption(region)
    if assumption is not None and region.basis in ("assumed", "mixed"):
        out["assumption"] = assumption
    if region.mixture and region.basis in ("mixed", "unreported"):
        out["mixture"] = dict(region.mixture)
    return out


def _bearing_uncertainty(u: ScalarUncertainty) -> dict[str, Any]:
    return {"basis": u.basis} if u.sigma is None else {"basis": u.basis, "sigma_deg": u.sigma}


def fix_entity(fix: Fix, library: dict[str, Any], label: str, ttl_s: float) -> dict[str, Any]:
    t = utc_text(fix.t)
    entity: dict[str, Any] = {
        "entity_id": fix.fix_id,
        "label": label[:32],
        "state": "tentative",
        "affiliation": {"identity": "unknown", "basis": "default", "reasons": []},
        "symbol": {"sidc": FIX_SIDC},
        "position": {"lat": fix.lat, "lon": fix.lon},
        "position_uncertainty": region_json(fix.region),
        "freq_hz": fix.freq_hz,
        "bandwidth_hz": fix.bandwidth_hz,
        "first_seen": t,
        "last_seen": t,
        "sources": [{"source_id": s, "observations": n} for s, n in sorted(fix.sources.items())],
        "classification": {"status": "unclassified", "reason": SINGLE_FIX},
        "library": library,
    }
    evidence = fix_evidence(fix)
    if evidence:
        entity["fix_evidence"] = evidence
    entity["meta"] = _fix_meta(fix, ttl_s)
    return entity


def fix_evidence(fix: Fix) -> dict[str, Any]:
    """The observations behind a fix, as picture.v0 ``fix_evidence`` (VIDENS_SPEC.md §6.7)."""
    if fix.bearings:
        return {
            "bearings": [
                {
                    "observation_id": wb.observation.observation_id,
                    "source_id": wb.observation.source_id,
                    **({"sensor_id": wb.observation.sensor_id} if wb.observation.sensor_id else {}),
                    "sensor_position": {
                        "lat": wb.observation.sensor_lat,
                        "lon": wb.observation.sensor_lon,
                        **(
                            {"alt_m": wb.observation.sensor_alt_m}
                            if wb.observation.sensor_alt_m is not None
                            else {}
                        ),
                    },
                    "bearing_deg": wb.observation.bearing_deg,
                    "bearing_uncertainty": _bearing_uncertainty(wb.observation.bearing_uncertainty),
                    "t": utc_text(wb.observation.t),
                    "weight": round(wb.weight, 6),
                }
                for wb in fix.bearings
            ]
        }
    if fix.position is not None:
        p = fix.position
        return {
            "positions": [
                {
                    "observation_id": p.observation_id,
                    "source_id": p.source_id,
                    "position": {"lat": p.lat, "lon": p.lon},
                    "uncertainty": region_json(fix.region),
                    "t": utc_text(p.t),
                }
            ]
        }
    return {}


def fix_details(fix: Fix) -> dict[str, Any]:
    """Human-readable facts about how a fix was made, for ``meta``."""
    meta: dict[str, Any] = {
        "method": "crossed lines of bearing"
        if fix.method == "bearings"
        else "position reported by the source",
    }
    if fix.residual_rms_deg is not None:
        meta["bearing_residual_rms_deg"] = round(fix.residual_rms_deg, 3)
    if fix.best_crossing_deg is not None:
        meta["best_crossing_deg"] = round(fix.best_crossing_deg, 1)
    if fix.height is not None:
        # picture.v0 has no altitude uncertainty yet, so height travels as text with its
        # sigma and basis rather than as a bare alt_m a viewer would draw as exact.
        meta["height"] = fix.height.describe()
    excluded = [wb for wb in fix.bearings if wb.excluded]
    if excluded:
        meta["not_used_in_solve"] = [
            f"{wb.observation.observation_id} ({wb.observation.source_id}): {wb.excluded}" for wb in excluded
        ]
    if fix.notes:
        meta["notes"] = list(fix.notes)
    return meta


def _fix_meta(fix: Fix, ttl_s: float) -> dict[str, Any]:
    return {
        "what_this_is": (
            "A single fix, not an emitter identity. Its uncertainty is unreported, so it cannot "
            f"update a track; it is shown alone and withdrawn after {ttl_s:g} s."
        ),
        **fix_details(fix),
    }
