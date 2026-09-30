"""Tracks in the picture: one entity per track (ADR-0009).

A track is an emitter, or several on one channel too close to tell apart. The entity carries
everything the reader needs to judge it: the filtered position with a region whose basis
comes from the fixes that built it, velocity with its covariance, the motion-model
probabilities (the classifier's "stationary" and "moving" features), lineage for splits and
merges, the latest fix's evidence, a trail, and the departures recorded against it.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from vigilans import geo
from vigilans.classify.classifier import Assessment, classification_json
from vigilans.classify.features import Feature
from vigilans.clock import utc_text
from vigilans.corrections import overlay_affiliation
from vigilans.locate.fix import Region, region_basis
from vigilans.picture.fixes import FIX_SIDC, fix_details, fix_evidence, region_json
from vigilans.track.reid import Reidentification
from vigilans.track.tracker import Track, Tracker
from vigilans_contract import StandardIdentity, sidc_with_identity

NO_LIBRARY = "Unclassified: no classification library is available, so nothing can be classified."


def _speed(ve: float, vn: float, ee: float, en: float, nn: float) -> tuple[float, float | None]:
    speed = math.hypot(ve, vn)
    if speed == 0.0:
        return 0.0, None
    var = (ve * ve * ee + 2 * ve * vn * en + vn * vn * nn) / (speed * speed)
    return speed, math.sqrt(max(var, 0.0))


def _identity_and_symbol(assessment: Assessment | None) -> tuple[dict[str, Any], str]:
    """Affiliation and symbol: the library's, only for a 'likely' class that declares one.
    An operator's declaration is laid over this (ADR-0015, ``overlay_affiliation``)."""
    default = {"identity": "unknown", "basis": "default", "reasons": []}
    best = assessment.best if assessment is not None else None
    if best is None or not best.wording.startswith("likely"):
        return default, FIX_SIDC
    identity: StandardIdentity = best.affiliation["identity"] if best.affiliation else "unknown"
    affiliation: dict[str, Any] = default
    if best.affiliation:
        affiliation = {
            "identity": identity,
            "basis": "library",
            "reasons": [
                *best.affiliation["reasons"],
                f"Applies because this is {best.wording} ({best.probability:.2f}).",
            ],
        }
    sidc = sidc_with_identity(best.sidc_template, identity) if best.sidc_template else FIX_SIDC
    if not best.sidc_template and identity != "unknown":
        sidc = sidc_with_identity("S*ZP-----------", identity)
    return affiliation, sidc


def features_text(features: dict[str, Feature]) -> list[str]:
    """What the classifier had to go on, for the inspector."""
    out = []
    for f in features.values():
        if f.ok and f.value is not None:
            spread = f" ± {f.sigma:.3g}" if f.sigma else ""
            basis = "" if f.basis == "measured" else f" [{f.basis}]"
            out.append(f"{f.name} {f.value:.4g}{spread} (n={f.n}){basis}")
        elif f.ok:
            out.append(f"{f.name}: {', '.join(sorted(f.categories))}")
        else:
            out.append(f"{f.name}: {f.status} -- {f.note}")
    return out


def track_entity(
    track: Track,
    tracker: Tracker,
    library: dict[str, Any],
    now: datetime,
    assessment: Assessment | None = None,
    features: dict[str, Feature] | None = None,
    *,
    operator_affiliation: dict[str, Any] | None = None,
    reidentification: Reidentification | None = None,
) -> dict[str, Any]:
    imm = track.imm
    if track.state == "coasting":
        imm = imm.copy()
        imm.predict((now - track.t).total_seconds())  # not heard: show the prediction, and its growth
    x, p = imm.combined()
    if track.anchor and track.anchor_m is not None:
        # An anchor publishes where it stays, from its own fixed-point estimate (ADR-0009).
        m, c = track.anchor_estimate(now, tracker.s.motion.q_stationary_m2ps)
        x, p = x.copy(), p.copy()
        x[:2], p[:2, :2] = m, c
    lat, lon, (pee, pen, pnn), (ve, vn, vee, ven, vnn) = tracker.from_frame(x, p)
    # Only fixes with a region update a track, so its bases are never unreported.
    basis, mixture = region_basis(list(track.bases))
    region = Region(basis, cov_en_m2=(pee, pen, pnn), assumptions=tuple(track.assumptions), mixture=mixture)

    affiliation, sidc = overlay_affiliation(*_identity_and_symbol(assessment), operator_affiliation)
    entity: dict[str, Any] = {
        "entity_id": track.track_id,
        "label": track.label,
        "state": track.state,
        "affiliation": affiliation,
        "symbol": {"sidc": sidc},
        "position": {"lat": lat, "lon": lon},
        "position_uncertainty": region_json(region),
        "velocity": {"east_mps": ve, "north_mps": vn, "cov_mps2": {"ee": vee, "en": ven, "nn": vnn}},
        "freq_hz": track.freq_hz,
        "bandwidth_hz": track.bandwidth_hz,
        "first_seen": utc_text(track.born),
        "last_seen": utc_text(track.last_heard),
        "sources": [{"source_id": s, "observations": n} for s, n in sorted(track.sources.items())],
        "classification": classification_json(assessment)
        if assessment is not None
        else {"status": "unclassified", "reason": NO_LIBRARY},
        "library": library,
    }
    lineage: dict[str, Any] = {}
    if track.split_from:
        lineage["split_from"] = track.split_from
    if track.merged_from:
        lineage["merged_from"] = list(track.merged_from)
        if track.merge_confidence is not None:
            lineage["merge_confidence"] = round(track.merge_confidence, 3)
    if lineage:
        entity["lineage"] = lineage
    if reidentification is not None:
        entity["reidentification"] = reidentification.to_json()
    if track.last_fix is not None:
        evidence = fix_evidence(track.last_fix)
        if evidence:
            entity["fix_evidence"] = evidence
    entity["trail"] = [{"t": utc_text(t), "lat": la, "lon": lo} for t, la, lo in list(track.trail)[-500:]]

    probabilities = imm.probabilities
    speed, speed_sigma = _speed(ve, vn, vee, ven, vnn)
    motion = " · ".join(f"{name} {value:.2f}" for name, value in probabilities.items())
    meta: dict[str, Any] = {
        "what_this_is": (
            "A tracked entity: one emitter, or several on one channel too close together to "
            "separate. Members that move apart become their own entities, recorded as splits."
        ),
        "motion_model_probabilities": motion,
        "fixes": f"{track.hits} fixes updated it"
        + (
            f"; {track.supporting_unweighted} more without uncertainty confirmed it is alive"
            if track.supporting_unweighted
            else ""
        ),
    }
    if speed_sigma is not None:
        meta["speed"] = (
            f"{speed:.1f} ± {speed_sigma:.1f} m/s, heading {geo.bearing_from_en(ve, vn):.0f}° true"
        )
        turn = imm.turn_rate()
        if turn is not None and speed > 1.0:
            rate, rate_sigma = turn
            side = "left" if rate > 0 else "right"
            meta["turn_rate"] = (
                f"{math.degrees(abs(rate)):.1f} ± {math.degrees(rate_sigma):.1f} °/s to the {side} "
                "(from the turning models; a sharper turn shows only as a wider spread)"
            )
    if track.anchor:
        meta["anchor"] = (
            "Has stayed put: it only accepts fixes consistent with staying put, so anything "
            "leaving it becomes its own entity."
        )
    if track.departures:
        meta["departures"] = [
            f"{d.departed} departed at {utc_text(d.t)} while this stayed "
            f"(stationary {d.stationary_probability:.2f})"
            for d in track.departures
        ]
        meta["departure_note"] = (
            f"{len(track.departures)} entities have departed from this one while it stayed. "
            "Grouping (Phase 7) weighs that as evidence of a command post or fixed site; "
            "no role is inferred yet."
        )
    if reidentification is not None:
        meta["reidentification_weighed"] = [
            f"{a['label']}: {a['probability']:.2f}" for a in reidentification.alternatives_json()
        ]
        meta["reidentification_note"] = (
            "Consistent with an earlier entity, not the same one for certain: externals are a weak "
            "fingerprint. The histories are kept apart; the earlier classification is carried "
            "forward only as a prior, weighted by this confidence."
        )
    if track.height is not None:
        meta["height"] = track.height.describe()
    if track.last_fix is not None:
        meta["latest_fix"] = fix_details(track.last_fix)
    if features is not None:
        meta["classification_features"] = features_text(features)
    entity["meta"] = meta
    return entity
