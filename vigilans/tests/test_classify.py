"""Phase 5 gate: the classifier, against the library boundary (ADR-0004, classifier-design.md).

- A private library loads from outside the tree; nothing real is in the repository (test_rules).
- Classification is right on the synthetic scenarios, hedged, and always with reasons.
- "Unclassified" says which of three things it is.
- Uncertainty and unreported timing weaken evidence rather than being ignored.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from _support import offline_config

from vigilans.app import Run
from vigilans.classify.classifier import Classifier, classification_json
from vigilans.classify.features import Feature
from vigilans.library import BUNDLED_LIBRARY, Library, load_library
from vigilans.picture.sinks import MemorySink
from vigilans.sources.file import FileInput
from vigilans_contract import forbidden_words
from vigilans_hub.world import load_world, simulate

HUB = Path(__file__).resolve().parents[2] / "hub" / "scenarios"
SYNTHETIC = load_library(BUNDLED_LIBRARY, private=False)


def f(name: str, value: float, sigma: float | None = None, basis: str = "measured", n: int = 10) -> Feature:
    return Feature(name, "ok", value, sigma, n, basis)  # type: ignore[arg-type]


def narrowband(**changes: Feature) -> dict[str, Feature]:
    features = {
        "freq_hz": f("freq_hz", 401e6, 1e3),
        "bandwidth_hz": f("bandwidth_hz", 12_500),
        "period_s": f("period_s", 20, 0.5),
        "stationary": f("stationary", 0.9),
    }
    features.update(changes)
    return features


# --- on the scenarios, against truth ---------------------------------------------------------------


def _run(scenario: str, tmp_path: Path, library: Library | None = SYNTHETIC) -> tuple[Run, dict[str, Any]]:
    world = load_world(HUB / f"{scenario}.scenario.yaml")
    records, _ = simulate(world)
    path = tmp_path / f"{scenario}.observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    run = Run(offline_config(), [FileInput(path)], library, run_id="t")
    asyncio.run(run.execute())
    return run, {e["id"]: e for e in world.emitters}


def test_mixed_every_entity_is_classified_correctly_and_likely(tmp_path: Path) -> None:
    run, emitters = _run("mixed", tmp_path)
    checked = 0
    for track_id, assessment in run.assessments.items():
        track = run.tracker.tracks.get(track_id)
        if track is None or track.hits < 5:
            continue
        nearest = min(emitters.values(), key=lambda e: abs(e["freq_hz"] - track.freq_hz))
        assert assessment.status == "classified", assessment.reason
        best = assessment.best
        assert best is not None
        assert best.class_id == nearest["truth"]["class_id"], (track.label, best.wording)
        assert best.wording.startswith("likely ")
        checked += 1
    assert checked == 4


def test_a_likely_class_brings_its_declared_affiliation_and_symbol(tmp_path: Path) -> None:
    run, _ = _run("mixed", tmp_path)
    entity = next(
        e
        for e in run.publisher.entities.values()
        if e["classification"]["status"] == "classified"
        and e["classification"]["candidates"][0]["class"] == "syn.wideband_link"
    )
    assert entity["affiliation"]["identity"] == "suspect"
    assert entity["affiliation"]["basis"] == "library"
    assert entity["symbol"]["sidc"][1] == "S"  # the frame follows the identity


def test_no_library_means_unclassified_and_says_so(tmp_path: Path) -> None:
    run, _ = _run("mixed", tmp_path, library=None)
    for entity in run.publisher.entities.values():
        assert entity["classification"] == {
            "status": "unclassified",
            "reason": "Unclassified: no classification library is available, so nothing can be classified.",
        }
        assert entity["affiliation"]["basis"] == "default"


def test_every_word_is_hedged_and_clean(tmp_path: Path) -> None:
    run, _ = _run("cp_departure", tmp_path)
    for entity in run.publisher.entities.values():
        text = json.dumps(entity["classification"])
        assert forbidden_words(text) == []
        for candidate in entity["classification"].get("candidates", []):
            assert candidate["wording"].split()[0] in ("likely", "possible")
            assert candidate["reasons"]


# --- the three kinds of unclassified --------------------------------------------------------------


def test_insufficient() -> None:
    assessment = Classifier(SYNTHETIC).assess({"freq_hz": f("freq_hz", 401e6)})
    assert (assessment.status, assessment.kind) == ("unclassified", "insufficient")
    assert "Insufficient" in assessment.reason


def test_unlike_the_library() -> None:
    features = {
        "freq_hz": f("freq_hz", 5.5e9, 1e3),
        "bandwidth_hz": f("bandwidth_hz", 3_000),
        "stationary": f("stationary", 0.9),
    }
    assessment = Classifier(SYNTHETIC).assess(features)
    assert (assessment.status, assessment.kind) == ("unclassified", "unlike_library")
    assert assessment.background_probability >= 0.6


def _twin_library(tmp_path: Path, *, shared_parent: bool) -> Library:
    document = yaml.safe_load(BUNDLED_LIBRARY.read_text("utf-8"))
    twin = dict(next(c for c in document["classes"] if c["class_id"] == "syn.narrowband_net"))
    twin = {**twin, "class_id": "syn.twin", "label": "synthetic twin net radio"}
    if not shared_parent:
        twin.pop("parent")
        for c in document["classes"]:
            if c["class_id"] == "syn.narrowband_net":
                c.pop("parent")
    document["classes"].append(twin)
    path = tmp_path / "twin.yaml"
    path.write_text(yaml.safe_dump(document), "utf-8")
    return load_library(path, private=False)


def test_ambiguous_between_indistinguishable_classes(tmp_path: Path) -> None:
    assessment = Classifier(_twin_library(tmp_path, shared_parent=False)).assess(narrowband())
    assert (assessment.status, assessment.kind) == ("unclassified", "ambiguous")
    assert "synthetic narrowband net radio" in assessment.reason and "twin" in assessment.reason


def test_ambiguity_resolves_to_the_shared_parent(tmp_path: Path) -> None:
    assessment = Classifier(_twin_library(tmp_path, shared_parent=True)).assess(narrowband())
    assert assessment.status == "classified"
    best = assessment.best
    assert best is not None and best.class_id == "syn.net_radio"  # the deepest level it can support


# --- evidence is weighed honestly ------------------------------------------------------------------


def _llr(features: dict[str, Feature], feature: str, class_id: str = "syn.narrowband_net") -> float:
    classifier = Classifier(SYNTHETIC)
    contributions = classifier._contributions(classifier.classes[class_id], features)
    return sum(c.llr for c in contributions if c.feature == feature)


def test_unreported_timing_counts_at_half() -> None:
    measured = _llr(narrowband(), "period_s")
    unreported = _llr(narrowband(period_s=f("period_s", 20, 0.5, basis="unreported")), "period_s")
    assert measured > 0 and math.isclose(unreported, measured / 2)


def test_a_fuzzy_measurement_is_weaker_evidence_both_ways() -> None:
    # Inside the class's range: for, but less so when the estimate is fuzzy.
    sharp_in = _llr(narrowband(freq_hz=f("freq_hz", 402e6, 1e3)), "freq_hz")
    fuzzy_in = _llr(narrowband(freq_hz=f("freq_hz", 402e6, 30e6)), "freq_hz")
    assert sharp_in > fuzzy_in
    # Outside (and not so far that the evidence saturates at the clip): against, less so when fuzzy.
    sharp_out = _llr(narrowband(freq_hz=f("freq_hz", 411.6e6, 1e3)), "freq_hz")
    fuzzy_out = _llr(narrowband(freq_hz=f("freq_hz", 411.6e6, 0.5e6)), "freq_hz")
    assert sharp_out < fuzzy_out < 0 < sharp_in


def test_soft_edges_not_vetoes() -> None:
    just_outside = _llr(narrowband(freq_hz=f("freq_hz", 405.3e6, 1e3)), "freq_hz")
    far_outside = _llr(narrowband(freq_hz=f("freq_hz", 450e6, 1e3)), "freq_hz")
    assert far_outside < just_outside


def test_a_source_claim_is_weighed_not_trusted() -> None:
    uas = {
        "freq_hz": f("freq_hz", 2.44e9, 1e3),
        "bandwidth_hz": f("bandwidth_hz", 10e6),
        "duty_cycle": f("duty_cycle", 0.95),
    }
    without = Classifier(SYNTHETIC).assess(uas)
    claimed = Classifier(SYNTHETIC).assess(
        {**uas, "source_claims": Feature("source_claims", "ok", n=3, categories=frozenset({"UAS"}))}
    )
    assert without.best is not None and claimed.best is not None
    assert claimed.best.probability >= without.best.probability
    assert any("the source claims UAS" in r for r in claimed.best.reasons)


def test_classification_json_meets_the_contract_shape() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    payload = classification_json(assessment)
    assert payload["status"] == "classified"
    candidate = payload["candidates"][0]
    assert set(candidate) == {"class", "confidence", "wording", "reasons"} and candidate["reasons"]


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_stationary_expectation(value: float) -> None:
    moving_llr = _llr(narrowband(stationary=f("stationary", value)), "stationary", "syn.mobile_net")
    assert (moving_llr > 0) == (value == 0.0)


# --- operator corrections in a run (ADR-0015) ------------------------------------------------------


def test_operator_corrections_reach_the_picture(tmp_path: Path) -> None:
    corrections = tmp_path / "corrections.toml"
    corrections.write_text(
        '[runs."t".entities."E-00001"]\n'
        'reject = "syn.narrowband_net"\n'
        'note = "heard it change channel on the hour"\n'
        'by = "watch 2"\n\n'
        '[runs."t".entities."E-00004"]\n'
        'affiliation = "friend"\n'
        'note = "our own link, per the exercise plan"\n\n'
        '[runs."another-run".entities."E-00002"]\n'
        'affiliation = "hostile"\n'
        'note = "for a different run"\n',
        "utf-8",
    )
    world = load_world(HUB / "mixed.scenario.yaml")
    records, _ = simulate(world)
    path = tmp_path / "mixed.observations.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    config = offline_config()
    config.corrections = corrections
    sink = MemorySink()
    run = Run(config, [FileInput(path)], SYNTHETIC, run_id="t", extra_sinks=[sink])
    asyncio.run(run.execute())
    entities = run.publisher.entities

    rejected = entities["E-00001"]["classification"]
    assert all(c["class"] != "syn.narrowband_net" for c in rejected.get("candidates", []))
    assert "watch 2" in json.dumps(rejected)

    link = entities["E-00004"]
    assert link["affiliation"]["identity"] == "friend" and link["affiliation"]["basis"] == "operator"
    assert link["symbol"]["sidc"][1] == "F"
    assert any("Library, not applied" in r for r in link["affiliation"]["reasons"])

    assert entities["E-00002"]["affiliation"]["basis"] != "operator"  # another run's entry
    codes = {m.get("code") for m in sink.of_type("notice")}
    assert "corrections_other_run" in codes


# --- a carried prior (re-identification, classifier-design §6.2) -----------------------------------


def test_a_carried_prior_moves_the_posterior_but_is_not_evidence() -> None:
    classifier = Classifier(SYNTHETIC)
    features = narrowband(stationary=f("stationary", 0.5))
    plain = classifier.assess(features)
    prior = classifier.default_prior()
    prior["syn.mobile_net"] *= 20
    carried = classifier.assess(features, prior=prior, prior_note="Prior carried from E-00007.")
    assert carried.posteriors["syn.mobile_net"] > plain.posteriors["syn.mobile_net"]
    for candidate in carried.candidates:
        assert "Prior carried from E-00007." in candidate.reasons
    # "likely" still needs the evidence itself: a prior alone never gets a class there.
    thin = {
        "freq_hz": f("freq_hz", 401e6, 1e3),
        "bandwidth_hz": f("bandwidth_hz", 12_500, basis="unreported"),
    }
    strong_prior = {k: 0.0 for k in classifier.default_prior()} | {"syn.mobile_net": 1.0}
    pushed = classifier.assess(thin, prior=strong_prior)
    assert all(
        not (c.class_id == "syn.mobile_net" and c.wording.startswith("likely")) for c in pushed.candidates
    )
