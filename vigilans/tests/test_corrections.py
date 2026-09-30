"""Operator corrections (ADR-0015, ADR-0006 decision 4).

- The file is strict: unknown keys, unknown classes, bad identities and missing notes are errors
  naming the file and the table.
- Entries for other runs are ignored and reported, never silently dropped.
- The file is re-read when it changes; a broken file keeps the last good corrections, loudly.
- A rejection renormalises; a confirmation moves the words, not the number; an operator
  affiliation outranks the library's. Wording stays hedged and every candidate has reasons.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from vigilans.classify.classifier import POSSIBLE, Assessment, Classifier, classification_json
from vigilans.classify.features import Feature
from vigilans.corrections import (
    Correction,
    CorrectionBook,
    CorrectionsError,
    apply,
    load_corrections,
    overlay_affiliation,
    parse_corrections,
)
from vigilans.library import BUNDLED_LIBRARY, load_library
from vigilans_contract import forbidden_words, sidc_identity

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "vigilans" / "config" / "corrections.example.toml"
SYNTHETIC = load_library(BUNDLED_LIBRARY, private=False)
RUN = "r1"
LEAVES = [c["class_id"] for c in SYNTHETIC.classes if c.get("features")]


def f(name: str, value: float, sigma: float | None = None) -> Feature:
    return Feature(name, "ok", value, sigma, 10, "measured")


def narrowband() -> dict[str, Feature]:
    return {
        "freq_hz": f("freq_hz", 401e6, 1e3),
        "bandwidth_hz": f("bandwidth_hz", 12_500),
        "period_s": f("period_s", 20, 0.5),
        "stationary": f("stationary", 0.9),
    }


def entry(body: str, run: str = RUN, entity: str = "E-00001") -> str:
    return f'[runs."{run}".entities."{entity}"]\n{body}\n'


def parse(text: str) -> dict[str, Correction]:
    return parse_corrections(text, "corrections.toml", SYNTHETIC).for_run(RUN)


def correction(**values: object) -> Correction:
    fields: dict[str, object] = {
        "run_id": RUN,
        "entity_id": "E-00001",
        "note": "Synthetic reason.",
        "declared_in": 'corrections.toml [runs."r1".entities."E-00001"]',
        "by": "watch 2",
    }
    fields.update(values)
    return Correction(**fields)  # type: ignore[arg-type]


def handmade(leaves: dict[str, float], background: float) -> Assessment:
    """An assessment with chosen leaf posteriors (the rest zero), summed up the tree like the classifier."""
    posteriors = {c["class_id"]: 0.0 for c in SYNTHETIC.classes}
    posteriors.update(leaves)
    for leaf, p in leaves.items():
        parent = next(c for c in SYNTHETIC.classes if c["class_id"] == leaf).get("parent")
        if parent:
            posteriors[parent] += p
    assert math.isclose(sum(leaves.values()) + background, 1.0)
    return Assessment(
        "unclassified",
        kind="ambiguous",
        reason="x",
        features_used=4,
        background_probability=background,
        posteriors=posteriors,
    )


def check_hedged(assessment: Assessment) -> None:
    body = classification_json(assessment)
    assert forbidden_words(json.dumps(body)) == []
    for candidate in body.get("candidates", []):
        assert candidate["wording"].split()[0] in ("likely", "possible")
        assert candidate["reasons"]
        assert all(0 < len(r) <= 2000 for r in candidate["reasons"])
        assert 0.0 <= candidate["confidence"] <= 1.0
    if body["status"] == "unclassified":
        assert body["reason"]


def write(path: Path, text: str) -> None:
    """Write, and move the modification time on, so a coarse filesystem clock cannot hide the change."""
    before = path.stat().st_mtime_ns if path.exists() else 0
    path.write_text(text, encoding="utf-8")
    stamp = max(path.stat().st_mtime_ns, before + 1_000_000_000)
    os.utime(path, ns=(stamp, stamp))


# --- the file format ----------------------------------------------------------------------------


def test_the_documented_example_is_valid() -> None:
    corrections = load_corrections(EXAMPLE, SYNTHETIC).for_run("mixed-demo")
    assert set(corrections) == {"E-00003", "E-00005", "E-00007"}
    assert corrections["E-00003"].confirm == "syn.wideband_link"
    assert corrections["E-00005"].reject == ("syn.uas_control",)
    assert corrections["E-00007"].affiliation == "neutral"
    assert corrections["E-00007"].by is None


def test_a_good_entry_records_where_it_was_declared() -> None:
    (only,) = parse(
        entry(
            'confirm = "syn.vhf_net"\nreject = "syn.uas_control"\n'
            'affiliation = "friend"\nnote = "Why."\nby = "w2"'
        )
    ).values()
    assert (only.confirm, only.reject, only.affiliation, only.note, only.by) == (
        "syn.vhf_net",
        ("syn.uas_control",),
        "friend",
        "Why.",
        "w2",
    )
    assert only.declared_in == 'corrections.toml [runs."r1".entities."E-00001"]'


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            'colour = "blue"\n' + entry('confirm = "syn.vhf_net"\nnote = "n"'),
            "corrections.toml: unknown key(s) colour",
        ),
        ('[runs.r1]\nlabel = "x"\n', 'corrections.toml [runs."r1"]: unknown key(s) label'),
        (
            entry('confirm = "syn.vhf_net"\nnote = "n"\nconfidence = 0.9'),
            '[runs."r1".entities."E-00001"]: unknown key(s) confidence',
        ),
        (entry('affiliation = "enemy"\nnote = "n"'), "affiliation 'enemy' is not a standard identity"),
        (entry('confirm = "syn.vhf_net"'), '[runs."r1".entities."E-00001"]: a correction needs a note'),
        (entry('confirm = "syn.vhf_net"\nnote = "  "'), "note must be a non-empty string"),
        (
            entry('confirm = "syn.nothing"\nnote = "n"'),
            "syn.nothing is not a class in library synthetic",
        ),
        (entry('reject = ["syn.vhf_net", "syn.nothing"]\nnote = "n"'), "syn.nothing is not a class"),
        (
            entry('confirm = "syn.vhf_net"\nreject = "syn.vhf_net"\nnote = "n"'),
            "confirms and rejects syn.vhf_net",
        ),
        (
            entry('confirm = "syn.vhf_net"\nreject = "syn.net_radio"\nnote = "n"'),
            "rejects syn.net_radio, which contains it",
        ),
        (
            entry('confirm = "syn.data_link"\nreject = ["syn.wideband_link", "syn.uas_control"]\nnote = "n"'),
            "rejects every class under it",
        ),
        (entry('confirm = ["syn.vhf_net"]\nnote = "n"'), "confirm names one class id"),
        (entry('note = "n"\nby = "w2"'), "says nothing to do"),
        (entry('reject = []\nnote = "n"'), "reject must be a class id or a non-empty list"),
        ("[runs.r1\n", "corrections.toml: not valid TOML"),
    ],
)
def test_a_bad_file_is_an_error_naming_file_and_table(text: str, expected: str) -> None:
    with pytest.raises(CorrectionsError) as caught:
        parse(text)
    assert expected in str(caught.value)
    assert "corrections.toml" in str(caught.value)


def test_a_note_may_not_carry_forbidden_vocabulary() -> None:
    word = "ki" + "ll"  # assembled, so this file does not carry the word itself
    with pytest.raises(CorrectionsError, match="forbidden vocabulary"):
        parse(entry(f'affiliation = "hostile"\nnote = "{word} box"'))


def test_class_ids_are_checked_only_when_a_library_is_given() -> None:
    parsed = parse_corrections(entry('confirm = "syn.nothing"\nnote = "n"'), "c.toml", None)
    assert parsed.for_run(RUN)["E-00001"].confirm == "syn.nothing"
    # And applying it against a library says it is not applied, rather than failing.
    result = apply(Classifier(SYNTHETIC).assess(narrowband()), parsed.for_run(RUN)["E-00001"], SYNTHETIC)
    assert result.best is not None and result.best.class_id == "syn.narrowband_net"


def test_other_runs_are_ignored_and_reported(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    write(
        path,
        entry('affiliation = "neutral"\nnote = "n"')
        + entry('affiliation = "hostile"\nnote = "n"', run="yesterday")
        + entry('affiliation = "friend"\nnote = "n"', run="yesterday", entity="E-00002"),
    )
    book = CorrectionBook(path, RUN, SYNTHETIC)
    result = book.load()
    assert set(book.entries) == {"E-00001"}
    assert result.other_runs == {"yesterday": ("E-00001", "E-00002")}
    warning = next(n for n in result.notices if n.code == "corrections_other_run")
    assert warning.severity == "warning" and "yesterday: E-00001, E-00002" in warning.text
    assert book.for_entity("yesterday", "E-00001") is None
    affiliation = book.affiliation_for(RUN, "E-00001")
    assert affiliation is not None and affiliation["identity"] == "neutral"
    assert "other runs ignored" in book.describe()


# --- reading and re-reading ----------------------------------------------------------------------


def test_reload_on_change(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    write(
        path,
        entry('confirm = "syn.vhf_net"\nnote = "first"')
        + entry('affiliation = "neutral"\nnote = "n"', entity="E-00002"),
    )
    book = CorrectionBook(path, RUN, SYNTHETIC)
    first = book.load()
    assert first.ok and first.added == ("E-00001", "E-00002")
    assert {n.code for n in first.notices} == {"correction_applied"}
    assert book.reload_if_changed() is None  # unchanged: nothing to do

    write(
        path,
        entry('confirm = "syn.vhf_net"\nnote = "second"')
        + entry('reject = "syn.fm_broadcast"\nnote = "n"', entity="E-00003"),
    )
    second = book.reload_if_changed()
    assert second is not None and second.ok
    assert (second.added, second.changed, second.withdrawn) == (("E-00003",), ("E-00001",), ("E-00002",))
    assert second.affected == ("E-00001", "E-00002", "E-00003")
    assert book.for_entity(RUN, "E-00001") is not None
    assert book.for_entity(RUN, "E-00001").note == "second"  # type: ignore[union-attr]
    assert book.affiliation_for(RUN, "E-00002") is None
    assert book.unmatched(["E-00001"]) == ["E-00003"]


def test_a_broken_file_keeps_the_last_good_and_says_so_loudly(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    write(path, entry('affiliation = "neutral"\nnote = "n"'))
    book = CorrectionBook(path, RUN, SYNTHETIC)
    book.load()
    good = dict(book.entries)

    write(path, entry('affiliation = "neutral"\nnote = "n"\ntypo = 1'))
    broken = book.reload_if_changed()
    assert broken is not None and not broken.ok
    assert broken.error is not None and "typo" in broken.error
    (notice,) = broken.notices
    assert notice.severity == "error" and "Keeping the 1 correction(s)" in notice.text
    assert book.entries == good
    assert book.affiliation_for(RUN, "E-00001") is not None
    assert "ERROR" in book.describe()
    assert book.reload_if_changed() is None  # reported once per version of the file, not every step

    write(path, entry('affiliation = "friend"\nnote = "fixed"'))
    fixed = book.reload_if_changed()
    assert fixed is not None and fixed.ok and fixed.changed == ("E-00001",)
    assert book.error is None


def test_a_bad_file_at_start_stops_the_run(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    write(path, entry('affiliation = "enemy"\nnote = "n"'))
    with pytest.raises(CorrectionsError, match="not a standard identity"):
        CorrectionBook(path, RUN, SYNTHETIC).load()


def test_a_file_that_appears_later_is_read_and_one_removed_keeps_its_corrections(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    book = CorrectionBook(path, RUN, SYNTHETIC)
    start = book.load()
    assert start.ok and start.notices[0].code == "corrections_absent"
    assert "not there yet" in book.describe()
    assert book.reload_if_changed() is None

    write(path, entry('affiliation = "neutral"\nnote = "n"'))
    appeared = book.reload_if_changed()
    assert appeared is not None and appeared.added == ("E-00001",)

    path.unlink()
    removed = book.reload_if_changed()
    assert removed is not None and not removed.ok
    assert removed.notices[0].severity == "warning" and "Keeping the 1" in removed.notices[0].text
    assert set(book.entries) == {"E-00001"}


def test_no_file_configured_says_so() -> None:
    book = CorrectionBook(None, RUN, SYNTHETIC)
    result = book.load()
    assert result.notices[0].code == "corrections_unavailable"
    assert book.reload_if_changed() is None
    assert book.describe().startswith("unavailable")


def test_a_byte_order_mark_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    path.write_bytes(b"\xef\xbb\xbf" + entry('affiliation = "neutral"\nnote = "n"').encode())
    assert set(load_corrections(path).for_run(RUN)) == {"E-00001"}


# --- rejection -----------------------------------------------------------------------------------


def test_rejection_renormalises_over_the_rest_and_the_background() -> None:
    before = handmade(
        {"syn.narrowband_net": 0.5, "syn.mobile_net": 0.3, "syn.fm_broadcast": 0.1}, background=0.1
    )
    after = apply(before, correction(reject=("syn.narrowband_net",)), SYNTHETIC)
    assert after.posteriors["syn.narrowband_net"] == 0.0
    assert math.isclose(after.posteriors["syn.mobile_net"], 0.6)
    assert math.isclose(after.posteriors["syn.fm_broadcast"], 0.2)
    assert math.isclose(after.background_probability, 0.2)
    assert math.isclose(sum(after.posteriors[c] for c in LEAVES) + after.background_probability, 1.0)
    assert math.isclose(after.posteriors["syn.net_radio"], 0.6)  # categories re-summed
    best = after.best
    assert best is not None and best.class_id == "syn.mobile_net"
    assert best.wording == "possible synthetic vehicle-borne net radio"
    assert any("Rejected by operator watch 2: synthetic narrowband net radio" in r for r in best.reasons)
    assert any("0.60 after the operator's correction, 0.30 before" in r for r in best.reasons)
    assert best.reasons[-1] == "Library synthetic 0.3.0."
    check_hedged(after)


def test_rejection_can_promote_to_likely_when_the_evidence_clears_the_sequential_test() -> None:
    before = handmade(
        {"syn.narrowband_net": 0.5, "syn.mobile_net": 0.45, "syn.vhf_net": 0.02, "syn.fm_broadcast": 0.02},
        background=0.01,
    )
    after = apply(before, correction(reject=("syn.narrowband_net",)), SYNTHETIC)
    best = after.best
    assert best is not None and best.class_id == "syn.mobile_net" and best.probability == pytest.approx(0.9)
    assert best.wording.startswith("likely ")


def test_rejection_on_a_real_assessment_moves_to_the_next_best_with_hedged_wording() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    assert assessment.best is not None and assessment.best.class_id == "syn.narrowband_net"
    after = apply(assessment, correction(reject=("syn.narrowband_net",)), SYNTHETIC)
    best = after.best
    assert best is not None and best.class_id == "syn.mobile_net"
    # 0.7 or more once renormalised, but its own evidence against the background has not
    # cleared the sequential test, so it stays "possible".
    assert best.probability >= 0.7 and best.wording.startswith("possible ")
    check_hedged(after)


def test_rejection_can_leave_it_unclassified_and_says_why() -> None:
    before = handmade({"syn.narrowband_net": 0.9, "syn.mobile_net": 0.02}, background=0.08)
    after = apply(before, correction(reject=("syn.narrowband_net",)), SYNTHETIC)
    assert (after.status, after.kind) == ("unclassified", "unlike_library")
    assert after.background_probability == pytest.approx(0.8)
    assert "once the operator's rejection is taken into account" in after.reason
    assert "Rejected by operator watch 2: synthetic narrowband net radio (syn.narrowband_net)" in after.reason
    assert 'corrections.toml [runs."r1".entities."E-00001"]' in after.reason
    check_hedged(after)


def test_rejecting_a_category_rejects_everything_under_it() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    after = apply(assessment, correction(reject=("syn.net_radio",)), SYNTHETIC)
    for leaf in ("syn.narrowband_net", "syn.mobile_net", "syn.vhf_net", "syn.net_radio"):
        assert after.posteriors[leaf] == 0.0
    assert after.status == "unclassified"


def test_rejecting_every_class_is_unlike_the_library() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    after = apply(
        assessment, correction(reject=("syn.net_radio", "syn.data_link", "syn.broadcast")), SYNTHETIC
    )
    assert (after.status, after.kind, after.background_probability) == ("unclassified", "unlike_library", 1.0)
    assert "every class in library synthetic 0.3.0 that could explain it has been rejected" in after.reason


def test_an_unrejected_winner_keeps_its_computed_reasons() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    assert assessment.best is not None
    after = apply(assessment, correction(reject=("syn.fm_broadcast",)), SYNTHETIC)
    assert after.best is not None and after.best.class_id == "syn.narrowband_net"
    assert set(assessment.best.reasons) <= set(after.best.reasons)
    assert after.best.wording == assessment.best.wording
    assert after.best.probability >= assessment.best.probability


# --- confirmation -------------------------------------------------------------------------------


def test_confirmation_keeps_the_number_and_says_who_why_and_where() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    computed = assessment.posteriors["syn.mobile_net"]
    assert computed < POSSIBLE  # the classifier would not show it at all
    after = apply(
        assessment, correction(confirm="syn.mobile_net", note="Seen leaving on the range camera."), SYNTHETIC
    )
    best = after.best
    assert after.status == "classified" and best is not None
    assert best.class_id == "syn.mobile_net"
    assert best.probability == pytest.approx(computed)  # not a faked number
    assert best.wording == "likely synthetic vehicle-borne net radio"
    assert best.reasons[0] == (
        "Confirmed by operator watch 2: Seen leaving on the range camera. "
        '(corrections.toml [runs."r1".entities."E-00001"]).'
    )
    assert f"computed probability, {computed:.2f}, below what it would show" in best.reasons[1]
    # The classifier's own view is still shown, as a capped alternative.
    alternative = after.candidates[1]
    assert alternative.class_id == "syn.narrowband_net" and alternative.wording.startswith("possible ")
    assert "confirmed synthetic vehicle-borne net radio instead" in alternative.reasons[0]
    check_hedged(after)


def test_confirming_the_classifiers_own_answer_adds_the_reason_to_its_own() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    assert assessment.best is not None
    after = apply(assessment, correction(confirm="syn.narrowband_net", by=None), SYNTHETIC)
    assert after.best is not None and after.best.probability == assessment.best.probability
    assert after.best.reasons[0].startswith("Confirmed by the operator: Synthetic reason.")
    assert set(assessment.best.reasons) <= set(after.best.reasons)
    assert len(after.candidates) == 1  # nothing unrelated is left at "possible"


def test_confirming_a_category_uses_the_summed_probability() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    after = apply(assessment, correction(confirm="syn.net_radio"), SYNTHETIC)
    assert after.best is not None and after.best.class_id == "syn.net_radio"
    assert after.best.probability == pytest.approx(assessment.posteriors["syn.net_radio"])
    assert all(c.class_id == "syn.net_radio" for c in after.candidates)  # its sub-classes are not rivals


def test_a_confirmed_class_brings_the_librarys_affiliation_with_the_confirmation() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    after = apply(assessment, correction(confirm="syn.wideband_link"), SYNTHETIC)
    best = after.best
    assert best is not None and best.affiliation is not None
    assert best.affiliation["identity"] == "suspect"
    assert "confirmed by operator watch 2" in best.affiliation["reasons"][-1]


def test_confirmation_with_nothing_computed_yet_shows_no_number() -> None:
    insufficient = Classifier(SYNTHETIC).assess({"freq_hz": f("freq_hz", 401e6)})
    after = apply(
        insufficient, correction(confirm="syn.narrowband_net", reject=("syn.fm_broadcast",)), SYNTHETIC
    )
    assert (after.status, after.kind, after.candidates) == ("unclassified", "insufficient", ())
    assert "Confirmed by operator watch 2 as synthetic narrowband net radio" in after.reason
    assert "Rejected by operator watch 2: synthetic FM broadcast transmitter" in after.reason


def test_affiliation_only_and_no_correction_leave_the_classification_alone() -> None:
    assessment = Classifier(SYNTHETIC).assess(narrowband())
    assert apply(assessment, None, SYNTHETIC) is assessment
    assert apply(assessment, correction(affiliation="neutral"), SYNTHETIC) is assessment
    assert apply(assessment, correction(confirm="syn.vhf_net"), None) is assessment


# --- affiliation --------------------------------------------------------------------------------


def test_an_operator_affiliation_outranks_the_librarys(tmp_path: Path) -> None:
    path = tmp_path / "corrections.toml"
    write(path, entry('affiliation = "neutral"\nnote = "Civil stand-in on the synthetic plan."\nby = "w2"'))
    book = CorrectionBook(path, RUN, SYNTHETIC)
    book.load()
    operator = book.affiliation_for(RUN, "E-00001")
    assert operator == {
        "identity": "neutral",
        "basis": "operator",
        "reasons": [
            "Declared neutral by operator w2: Civil stand-in on the synthetic plan. "
            '(corrections.toml [runs."r1".entities."E-00001"]).'
        ],
    }
    library = {"identity": "suspect", "basis": "library", "reasons": ["The library says so."]}
    affiliation, sidc = overlay_affiliation(library, "SSGPE----------", operator)
    assert affiliation["identity"] == "neutral" and affiliation["basis"] == "operator"
    assert sidc_identity(sidc) == "neutral" and sidc == "SNGPE----------"
    assert any("Replaces the library's suspect" in r for r in affiliation["reasons"])
    assert "Library, not applied: The library says so." in affiliation["reasons"]
    # Over the default, it simply wins; with no declaration, nothing changes.
    default = {"identity": "unknown", "basis": "default", "reasons": []}
    assert overlay_affiliation(default, "SUZP-----------", operator) == (operator, "SNZP-----------")
    assert overlay_affiliation(library, "SSGPE----------", None) == (library, "SSGPE----------")
    assert book.affiliation_for("another-run", "E-00001") is None


# --- invariants ---------------------------------------------------------------------------------


@pytest.mark.parametrize("confirm", [None, *LEAVES, "syn.net_radio", "syn.data_link"])
@pytest.mark.parametrize(
    "reject", [(), ("syn.narrowband_net",), ("syn.data_link",), ("syn.net_radio", "syn.broadcast")]
)
def test_every_corrected_assessment_is_hedged_with_reasons(
    confirm: str | None, reject: tuple[str, ...]
) -> None:
    if confirm is None and not reject:
        return
    lines = ['note = "n"']
    if confirm:
        lines.append(f"confirm = {json.dumps(confirm)}")
    if reject:
        lines.append(f"reject = {json.dumps(list(reject))}")
    try:
        parse(entry("\n".join(lines)))
    except CorrectionsError:
        return  # a contradictory correction never reaches apply
    for features in (
        narrowband(),
        {"freq_hz": f("freq_hz", 2.44e9, 1e3), "bandwidth_hz": f("bandwidth_hz", 10e6)},
    ):
        after = apply(
            Classifier(SYNTHETIC).assess(features), correction(confirm=confirm, reject=reject), SYNTHETIC
        )
        check_hedged(after)
        if confirm is not None:
            assert after.best is not None and after.best.class_id == confirm
            assert after.best.wording.startswith("likely ")
            assert sum(c.wording.startswith("likely ") for c in after.candidates) == 1
