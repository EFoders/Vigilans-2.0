"""Phase 6 gate: re-identification confidence is carried downstream (spec §8, ADR-0016).

On ``reappear`` (hub/scenarios), run through ingest, geolocation and tracking as the engine does:
retired entities are remembered, and each new entity is weighed against them.

- A static emitter back at its spot after 15 minutes: consistent with its old entity, high.
- A mover back further along its way, reachably: consistent with its old entity.
- Negative controls: a different emitter on the same channel somewhere unreachable is not
  re-identified; an ambiguous reappearance between identical twins is not published, and
  names both.
- The confidence goes downstream: the classifier prior (classifier-design §6.2) and the
  bound on anything built on several re-identifications (§10.3).

Thresholds are stated as measured, with margin, and are not to be loosened to pass (rule 8).
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _support import offline_config

from vigilans import geo
from vigilans.app import Run
from vigilans.classify.classifier import BACKGROUND, Assessment, Classifier
from vigilans.classify.features import Feature, extract
from vigilans.clock import SIM_EPOCH, parse_utc
from vigilans.ingest import Ingestor, Raw
from vigilans.library import BUNDLED_LIBRARY, load_library
from vigilans.locate.bearings import LocateSettings
from vigilans.locate.cochannel import locate_group
from vigilans.locate.fix import Fix
from vigilans.locate.grouping import Grouper, GroupingSettings
from vigilans.observation import BearingObservation
from vigilans.picture.sinks import MemorySink
from vigilans.sources.file import FileInput
from vigilans.track.reid import (
    Consideration,
    Fingerprint,
    Reidentification,
    Reidentifier,
    ReidSettings,
    carried_prior,
    carried_reason,
    identity_bound,
)
from vigilans.track.tracker import Track, Tracker, TrackSettings
from vigilans_contract import forbidden_words
from vigilans_hub.world import build_world, load_world, simulate

REPO = Path(__file__).resolve().parents[2]
SCENARIO = REPO / "hub" / "scenarios" / "reappear.scenario.yaml"
#: New entities are weighed every this often, from confirmation until this age.
CONSIDER_EVERY_S = 10
DECIDE_WITHIN_S = 900.0


# --- the pipeline ------------------------------------------------------------------------------


@dataclass
class ReidRun:
    tracker: Tracker
    reid: Reidentifier
    emitter_of: dict[str, str] = field(default_factory=dict)
    tracks: dict[str, Track] = field(default_factory=dict)
    remembered: dict[str, Fingerprint] = field(default_factory=dict)
    assessments: dict[str, Assessment] = field(default_factory=dict)
    #: new entity id -> every consideration made for it, in order.
    considered: dict[str, list[Consideration]] = field(default_factory=lambda: defaultdict(list))
    #: new entity id -> its first published re-identification.
    published: dict[str, Reidentification] = field(default_factory=dict)

    def emitter(self, track_id: str) -> str:
        held = Counter(self.emitter_of[f] for f in self.tracks[track_id].fix_ids if f in self.emitter_of)
        return held.most_common(1)[0][0]

    def entities_of(
        self, emitter_id: str, born_after_s: float = 0.0, born_before_s: float = 1e12
    ) -> list[str]:
        """Published entities (not folded or merged away) mostly made of this emitter's fixes."""
        out = []
        for track_id, track in sorted(self.tracks.items(), key=lambda kv: kv[1].number):
            born = (track.born - SIM_EPOCH).total_seconds()
            if (
                track.hits < 3
                or track_id in self.tracker.absorbed
                or not born_after_s <= born < born_before_s
            ):
                continue
            if self.emitter(track_id) == emitter_id:
                out.append(track_id)
        return out

    def remembered_of(self, emitter_id: str, born_before_s: float) -> set[str]:
        """The remembered entities of an emitter (more than one if tracking fragmented it)."""
        return {t for t in self.entities_of(emitter_id, born_before_s=born_before_s) if t in self.remembered}

    def old_and_new(self, emitter_id: str, silent_from_s: float, back_at_s: float) -> tuple[set[str], str]:
        old = self.remembered_of(emitter_id, silent_from_s)
        new = self.entities_of(emitter_id, born_after_s=back_at_s - 5)
        assert old, f"{emitter_id}: nothing remembered"
        assert new, f"{emitter_id}: no new entity after {back_at_s} s"
        return old, new[0]


def run_reappear(seed: int | None = None) -> ReidRun:
    world = load_world(SCENARIO)
    if seed is not None:
        world = build_world({**world.raw, "seed": seed})
    records, truth = simulate(world)
    tracker = Tracker(TrackSettings(), geo.LocalFrame(*world.origin))
    run = ReidRun(tracker, Reidentifier(tracker))
    classifier = Classifier(load_library(BUNDLED_LIBRARY, private=False))
    ingestor = Ingestor()
    grouper = Grouper(GroupingSettings())
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_step[int((parse_utc(record["t"]) - SIM_EPOCH).total_seconds()) + 1].append(record)
    last = int(world.duration_s + 60)
    count = 0
    for step in range(1, last + 1):
        now = SIM_EPOCH + timedelta(seconds=step)
        batch = ingestor.ingest(Raw("sim", record=r) for r in by_step.get(step, []))
        grouper.add([o for o in batch.accepted if isinstance(o, BearingObservation)])
        fixes: list[Fix] = []
        for group in grouper.close(None if step == last else now):

            def next_id() -> str:
                nonlocal count
                count += 1
                return f"f{count}"

            located, _ = locate_group(group, next_id, LocateSettings())
            for fix in located:
                used = [wb.observation.observation_id for wb in fix.bearings]
                run.emitter_of[fix.fix_id] = Counter(truth[o] for o in used).most_common(1)[0][0]
                fixes.append(fix)
        tracker.add(fixes)
        before = dict(tracker.tracks)
        out = tracker.step(now, flush=step == last)
        for track in tracker.tracks.values():
            run.tracks[track.track_id] = track
        # The hook the engine needs: a track retired for silence is remembered, with its
        # features as they were when it was last heard and its classification then.
        for event in out.events:
            if event["kind"] != "entity_retired":
                continue
            retired = before[event["objects"][0]["id"]]
            run.tracks[retired.track_id] = retired
            features = extract(retired, tracker, retired.last_heard)
            assessment = classifier.assess(features)
            if run.reid.remember(retired, features, assessment, now) is not None:
                run.remembered[retired.track_id] = run.reid.memory[retired.track_id]
                run.assessments[retired.track_id] = assessment
        # ...and a new entity is weighed against memory from confirmation, as its features
        # accumulate, until it is published or old enough that it is simply new.
        if step % CONSIDER_EVERY_S:
            continue
        for track in list(tracker.tracks.values()):
            age = (now - track.born).total_seconds()
            if (
                track.state != "confirmed"
                or track.probationary
                or track.track_id in run.published
                or age > DECIDE_WITHIN_S
            ):
                continue
            consideration = run.reid.consider(track, extract(track, tracker, now), now)
            if consideration.alternatives:
                run.considered[track.track_id].append(consideration)
            if consideration.published is not None:
                run.published[track.track_id] = consideration.published
                run.reid.accept(consideration.published)
    return run


@pytest.fixture(scope="module")
def reappear() -> ReidRun:
    return run_reappear()


# --- the scenario --------------------------------------------------------------------------------


def test_same_place_reappearance_is_reidentified_with_high_confidence(reappear: ReidRun) -> None:
    old, new = reappear.old_and_new("STAY", 900, 1800)
    reid = reappear.published.get(new)
    assert reid is not None and reid.consistent_with in old
    assert reid.confidence >= 0.8  # measured 0.89 (seeds 1-12: 0.89-0.90)
    assert reid.gap_s >= 600  # it really was retired first


def test_a_mover_is_reidentified_when_reachable(reappear: ReidRun) -> None:
    old, new = reappear.old_and_new("MOVER", 700, 1500)
    reid = reappear.published.get(new)
    assert reid is not None and reid.consistent_with in old
    assert reid.confidence >= 0.5  # measured 0.66 (seeds 1-12: 0.56-0.74, or not published)
    assert reid.distance_m >= 5000  # it had moved on, and that was reachable
    stay = reappear.published[reappear.old_and_new("STAY", 900, 1800)[1]]
    assert reid.confidence < stay.confidence


def test_an_unreachable_emitter_on_the_same_channel_is_not_reidentified(reappear: ReidRun) -> None:
    far_a = reappear.remembered_of("FAR-A", 600)
    assert far_a
    new = reappear.entities_of("FAR-B", born_after_s=1400)
    assert new
    for track_id in new:
        assert track_id not in reappear.published
        considered = reappear.considered.get(track_id, [])
        assert considered  # it was weighed -- same channel -- and turned down every time
        for consideration in considered:
            weighed = {a.entity_id: a for a in consideration.alternatives}
            for old in far_a:
                assert weighed[old].probability < 0.01  # measured 7e-6
                assert not weighed[old].reachable
                assert any("Not reachable" in r for r in weighed[old].reasons)


def test_an_ambiguous_reappearance_between_twins_is_not_published(reappear: ReidRun) -> None:
    twin_a, twin_b = reappear.remembered_of("TWIN-A", 800), reappear.remembered_of("TWIN-B", 800)
    assert twin_a and twin_b
    new = reappear.entities_of("TWIN-A", born_after_s=1695)
    assert new
    assert not any(t in reappear.published for t in new)
    considered = reappear.considered[new[0]]
    assert considered
    final = considered[-1]
    weighed = {a.entity_id: a.probability for a in final.alternatives}
    a = max(weighed[t] for t in twin_a)
    b = max(weighed[t] for t in twin_b)
    assert a < 0.5 and b < 0.5  # measured 0.37 and 0.36
    assert abs(a - b) < 0.1  # both named, neither preferred
    assert "Ambiguous" in final.note


def test_the_old_classification_is_carried_downstream(reappear: ReidRun) -> None:
    # The Phase 6 gate: the confidence goes into the new entity's classifier prior.
    _, new = reappear.old_and_new("STAY", 900, 1800)
    reid = reappear.published[new]
    before = reappear.assessments[reid.consistent_with]
    assert before.posteriors and before.best is not None
    assert reid.posteriors == {**before.posteriors, BACKGROUND: before.background_probability}
    classifier = Classifier(load_library(BUNDLED_LIBRARY, private=False))
    default = classifier.default_prior()
    prior = carried_prior(reid, default)
    r = reid.confidence
    for k, v in default.items():
        assert prior[k] == pytest.approx(r * reid.posteriors.get(k, 0.0) + (1 - r) * v)
    assert sum(prior.values()) == pytest.approx(1.0)
    # With little evidence of its own yet, the new entity leans to what the old one was --
    # by as much as the re-identification is believed, no more.
    early = {k: v for k, v in _stay_features(reappear, new).items() if k in ("freq_hz", "bandwidth_hz")}
    plain = classifier.assess(early)
    carried = classifier.assess(early, prior=prior)
    leaf = before.best.class_id
    assert carried.posteriors[leaf] >= plain.posteriors[leaf]
    assert "consistent with" in carried_reason(reid)


def _stay_features(reappear: ReidRun, track_id: str) -> dict[str, Feature]:
    track = reappear.tracks[track_id]
    return extract(track, reappear.tracker, track.last_heard)


def test_every_reason_is_hedged_and_clean(reappear: ReidRun) -> None:
    assert reappear.published
    texts: list[str] = []
    for reid in reappear.published.values():
        assert reid.reasons
        assert "consistent with" in reid.reasons[0].lower()
        texts.extend(reid.reasons)
        texts.append(carried_reason(reid))
    for considerations in reappear.considered.values():
        for c in considerations:
            texts.append(c.note)
            for a in c.alternatives:
                assert a.reasons
                texts.extend(a.reasons)
    for text in texts:
        assert not re.search(r"\bis\b", text), text
        assert not forbidden_words(text), text


# --- the model, on hand-made fingerprints ---------------------------------------------------------


def f(name: str, value: float, sigma: float | None = None, basis: str = "measured") -> Feature:
    return Feature(name, "ok", value, sigma, 10, basis)  # type: ignore[arg-type]


def features(**changes: float) -> dict[str, Feature]:
    base = {
        "freq_hz": f("freq_hz", 412_000_000.0, 50.0),
        "bandwidth_hz": f("bandwidth_hz", 12_500.0, 100.0),
        "duration_s": f("duration_s", 2.0, 0.1),
        "period_s": f("period_s", 30.0, 0.3),
        "regularity_cv": f("regularity_cv", 0.03, 0.01),
    }
    for name, value in changes.items():
        base[name] = f(name, value, base[name].sigma if name in base else None)
    return base


def fp(
    entity_id: str,
    east_m: float,
    north_m: float,
    heard: tuple[float, float],
    feats: dict[str, Feature] | None = None,
    *,
    stationary: float = 0.95,
    speed_mps: float = 0.0,
    sigma_m: float = 150.0,
    posteriors: dict[str, float] | None = None,
) -> Fingerprint:
    lat, lon = geo.offset(50.0, -105.0, east_m, north_m)
    feats = feats if feats is not None else features()
    return Fingerprint(
        entity_id=entity_id,
        label=entity_id,
        features=feats,
        freq_hz=float(feats["freq_hz"].value or 0.0),
        bandwidth_hz=12_500.0,
        lat=lat,
        lon=lon,
        cov_en_m2=(sigma_m**2, 0.0, sigma_m**2),
        position_t=SIM_EPOCH + timedelta(seconds=heard[1]),
        first_heard=SIM_EPOCH + timedelta(seconds=heard[0]),
        last_heard=SIM_EPOCH + timedelta(seconds=heard[1]),
        stationary=stationary,
        anchored=stationary >= 0.5,
        speed_mps=speed_mps,
        speed_sigma_mps=0.5,
        posteriors=posteriors or {},
    )


def reidentifier(settings: ReidSettings | None = None) -> Reidentifier:
    return Reidentifier(Tracker(TrackSettings(), geo.LocalFrame(50.0, -105.0)), settings)


def at(seconds: float) -> datetime:
    return SIM_EPOCH + timedelta(seconds=seconds)


def test_same_place_same_behaviour_publishes_with_reasons() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900)), at(1500))
    c = r.consider_fingerprint(fp("E9", 50, -40, (1800, 1900)), at(1900))
    assert c.published is not None and c.published.consistent_with == "E1"
    assert c.published.reasons[0].startswith("Consistent with E1")
    assert "same channel 412.000 MHz" in c.published.reasons[0]


def test_confidence_follows_the_stated_formula() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900)), at(1500))
    r.remember_fingerprint(fp("E2", 4000, 0, (0, 800), stationary=0.2, speed_mps=8.0), at(1500))
    c = r.consider_fingerprint(fp("E9", 50, -40, (1800, 1900)), at(1900))
    odds = {a.entity_id: a.prior_odds * a.feature_lr * a.geometry_lr for a in c.alternatives}
    total = 1.0 + sum(odds.values())
    for a in c.alternatives:
        assert a.probability == pytest.approx(odds[a.entity_id] / total)
    assert c.p_different == pytest.approx(1.0 / total)


def test_confidence_never_exceeds_what_the_features_support() -> None:
    s = ReidSettings()
    r = reidentifier(s)
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900)), at(1500))
    full = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900)), at(1900))
    assert full.best is not None
    # Perfect agreement cannot say more than "same type on this channel": bounded by twins.
    assert full.best.feature_lr <= 1.0 / s.twin_fraction + 1e-9
    # Fewer shared features: less confidence, and not enough to publish.
    thin = {k: v for k, v in features().items() if k in ("freq_hz", "bandwidth_hz")}
    few = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900), thin), at(1900))
    assert few.best is not None and few.best.probability < full.best.probability
    assert few.published is None and "Waiting" in few.note
    # A different interval between transmissions is evidence against: not published.
    other = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900), features(period_s=45.0)), at(1900))
    assert other.best is not None and other.best.probability < full.best.probability
    assert other.published is None
    assert any("(against)" in reason for reason in other.best.reasons)
    # Timing with no coverage records (basis unreported) counts at half: weaker evidence.
    unreported = {
        k: (Feature(k, "ok", v.value, v.sigma, v.n, "unreported") if k in ("duration_s", "period_s") else v)
        for k, v in features().items()
    }
    half = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900), unreported), at(1900))
    assert half.best is not None and half.best.feature_lr < full.best.feature_lr


def test_identical_twins_are_ambiguous_and_both_are_named() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", -3000, 0, (0, 800)), at(1400))
    r.remember_fingerprint(fp("E2", 3000, 0, (0, 800)), at(1400))
    c = r.consider_fingerprint(fp("E9", 0, 0, (1700, 1800)), at(1800))
    assert c.published is None
    names = {a.entity_id: a.probability for a in c.alternatives}
    assert set(names) == {"E1", "E2"}
    assert names["E1"] == pytest.approx(names["E2"], abs=0.02)
    assert max(names.values()) < 0.5
    assert "Ambiguous" in c.note and "E1" in c.note and "E2" in c.note
    assert r.last is c  # kept for debugging, though nothing was published


def test_unreachable_is_not_consistent() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", -12000, 8000, (0, 600)), at(1200))
    c = r.consider_fingerprint(fp("E9", 12000, -6000, (1500, 1600)), at(1600))
    assert c.published is None and c.best is not None
    assert not c.best.reachable and c.best.probability < 0.01
    assert c.best.reasons[0].startswith("Not consistent with E1")


def test_a_mover_further_along_is_reachable_but_weaker_than_staying_put() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 700), stationary=0.02, speed_mps=10.0), at(1300))
    c = r.consider_fingerprint(fp("E9", 6000, 5000, (1500, 1600), stationary=0.02, speed_mps=10.0), at(1600))
    assert c.best is not None and c.best.reachable
    r2 = reidentifier()
    r2.remember_fingerprint(fp("E1", 0, 0, (0, 700)), at(1300))
    still = r2.consider_fingerprint(fp("E9", 0, 0, (1500, 1600)), at(1600))
    assert still.best is not None and c.best.probability < still.best.probability


def test_different_channel_and_overlapping_time_are_not_candidates() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900), features(freq_hz=412_025_000.0)), at(1500))
    r.remember_fingerprint(fp("E2", 0, 0, (0, 1850)), at(1900))  # still heard after E9 appeared
    c = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900)), at(1900))
    assert c.alternatives == () and c.published is None


def test_memory_expires_and_is_bounded() -> None:
    r = reidentifier(ReidSettings(max_entries=2))
    for i in range(3):
        r.remember_fingerprint(fp(f"E{i}", 0, 0, (0, 100 * (i + 1))), at(1000))
    assert set(r.memory) == {"E1", "E2"}  # the longest-silent went first
    c = r.consider_fingerprint(fp("E9", 0, 0, (7 * 3600, 7 * 3600 + 60)), at(7 * 3600 + 60))
    assert c.alternatives == () and r.memory == {}


def test_accepting_uses_up_the_memory() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900)), at(1500))
    reid = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900)), at(1900)).published
    assert reid is not None
    r.accept(reid)
    assert "E1" not in r.memory


def test_carried_prior_arithmetic() -> None:
    default = {"a": 0.25, "b": 0.25, BACKGROUND: 0.5}
    old = {"a": 0.8, "b": 0.1, "cat": 0.9, BACKGROUND: 0.1}
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900), posteriors=old), at(1500))
    reid = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900)), at(1900)).published
    assert reid is not None
    c = reid.confidence
    prior = carried_prior(reid, default)
    assert set(prior) == set(default)  # only the classifier's hypotheses; categories are sums
    assert prior["a"] == pytest.approx(c * 0.8 + (1 - c) * 0.25)
    assert prior["b"] == pytest.approx(c * 0.1 + (1 - c) * 0.25)
    assert prior[BACKGROUND] == pytest.approx(c * 0.1 + (1 - c) * 0.5)
    assert sum(prior.values()) == pytest.approx(1.0)
    assert carried_prior(None, default) == default
    r2 = reidentifier()
    r2.remember_fingerprint(fp("E2", 0, 0, (0, 900)), at(1500))  # never classified
    unclassified = r2.consider_fingerprint(fp("E8", 0, 0, (1800, 1900)), at(1900)).published
    assert unclassified is not None and carried_prior(unclassified, default) == default


def test_identity_bound() -> None:
    assert identity_bound() == 1.0
    assert identity_bound(0.6, 0.6) == pytest.approx(0.36)  # §10.3: not 0.9, and not 0.6
    assert identity_bound(0.9, 0.5, 0.8) <= min(0.9, 0.5, 0.8)
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        identity_bound(1.2)


def test_to_json_shape() -> None:
    r = reidentifier()
    r.remember_fingerprint(fp("E1", 0, 0, (0, 900)), at(1500))
    reid = r.consider_fingerprint(fp("E9", 0, 0, (1800, 1900)), at(1900)).published
    assert reid is not None
    body = reid.to_json()
    assert body["consistent_with"] == "E1" and 0.0 < body["confidence"] <= 1.0 and body["reasons"]
    assert set(body) == {"consistent_with", "confidence", "reasons"}  # picture.v0 allows no more
    alternatives = reid.alternatives_json()
    assert math.isclose(sum(a["probability"] for a in alternatives), 1.0, abs_tol=0.01)


# --- in the engine: the picture carries it (Phase 6 gate) -------------------------------------------


def test_the_engine_publishes_reidentification_and_carries_it_into_classification(tmp_path: Path) -> None:
    records, _ = simulate(load_world(SCENARIO))
    path = tmp_path / "reappear.observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), "utf-8")
    sink = MemorySink()
    library = load_library(BUNDLED_LIBRARY, private=False)
    run = Run(offline_config(), [FileInput(path)], library, run_id="t", extra_sinks=[sink])
    asyncio.run(run.execute())  # the publisher is strict: every message met picture.v0

    events = [e for m in sink.messages for e in m.get("events", []) if e["kind"] == "reidentified"]
    assert len(events) >= 2  # STAY and MOVER; measured 2 on this seed
    published = [
        e
        for m in sink.messages
        for e in m.get("entities", [])
        if isinstance(e, dict) and "reidentification" in e
    ]
    assert published
    for entity in published:
        reid = entity["reidentification"]
        assert reid["confidence"] >= ReidSettings().publish_threshold
        assert all(r.startswith("Consistent with") for r in reid["reasons"][:1])
        assert entity["entity_id"] != reid["consistent_with"]  # never merged, never the old id
    # Downstream: the carried prior is in the classification, and says from where and how much.
    for track_id, reid in run.reidentifications.items():
        assessment = run.assessments[track_id]
        if assessment.status == "classified" and reid.posteriors:
            assert carried_reason(reid) in assessment.candidates[0].reasons
