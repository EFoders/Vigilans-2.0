"""Operator corrections: confirm or reject a classification, or declare an affiliation (ADR-0015).

An operator writes corrections in a TOML file that Vigilans reads at start and re-reads
whenever it changes (ADR-0006, decision 4). Each entry is about one entity **in one run** —
entity ids are only unique within a run — and always says why (``note``) and may say who
(``by``). See ``vigilans/config/corrections.example.toml``::

    [runs."mixed-demo".entities."E-00003"]
    confirm = "syn.wideband_link"      # a class id from the loaded library
    reject = ["syn.uas_control"]       # one class id, or a list of them
    affiliation = "suspect"            # a MIL-STD-2525 standard identity
    note = "Seen on the synthetic range schedule for this slot."
    by = "watch 2"

What a correction does:

- **reject** removes the class (a category: everything under it) and renormalises the
  remaining posterior over the other classes and the background — conditioning on "not
  that", which may leave the entity unclassified, and the reason then says so;
- **confirm** words the class "likely" whatever its computed posterior, and shows the
  computed posterior as the number: the operator's knowledge is outside the model, so it
  moves the words, never the arithmetic;
- **affiliation** replaces the library's affiliation (operator, then library, then default).

The file is strict: an unknown key, an unknown class, an identity that is not a standard
identity, a missing note or forbidden vocabulary is an error naming the file and the table.
A file that goes bad during a run is reported loudly and the last good corrections are kept;
nothing silently falls back. Entries for other runs are ignored and reported.

Corrections are provenance: every reason they add names where they were declared. They do
not feed any learning across runs (classifier-design.md §6.3).
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from vigilans.classify.classifier import (
    AMBIGUITY_GAP,
    LIKELY,
    POSSIBLE,
    SPRT_UPPER,
    Assessment,
    Candidate,
    ClassifierSettings,
)
from vigilans.library import Library
from vigilans_contract import IDENTITIES, StandardIdentity, forbidden_words, sidc_with_identity

Severity = Literal["info", "warning", "error"]

MAX_NOTE = 500
MAX_BY = 64
MAX_ID = 128

_TOP_KEYS = {"runs"}
_RUN_KEYS = {"entities"}
_ENTRY_KEYS = {"confirm", "reject", "affiliation", "note", "by"}


class CorrectionsError(ValueError):
    """A corrections file that cannot be used, with the file, the table and why."""


# --- the file -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Correction:
    """One entity's corrections, as declared. ``declared_in`` is the file and table."""

    run_id: str
    entity_id: str
    note: str
    declared_in: str
    by: str | None = None
    confirm: str | None = None
    reject: tuple[str, ...] = ()
    affiliation: StandardIdentity | None = None

    @property
    def operator(self) -> str:
        return f"operator {self.by}" if self.by else "the operator"

    @property
    def touches_classification(self) -> bool:
        return self.confirm is not None or bool(self.reject)

    def summary(self) -> str:
        parts = []
        if self.confirm:
            parts.append(f"confirm {self.confirm}")
        if self.reject:
            parts.append(f"reject {', '.join(self.reject)}")
        if self.affiliation:
            parts.append(f"affiliation {self.affiliation}")
        return (
            f"{self.entity_id}: {'; '.join(parts)}. Why: {self.note} "
            f"({self.by or 'operator not named'}; {self.declared_in})"
        )


@dataclass(frozen=True, slots=True)
class CorrectionsFile:
    """Every entry in one file, for every run it names."""

    name: str
    entries: Mapping[tuple[str, str], Correction] = field(default_factory=dict)

    def for_run(self, run_id: str) -> dict[str, Correction]:
        return {entity: c for (run, entity), c in self.entries.items() if run == run_id}

    def other_runs(self, run_id: str) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for run, entity in self.entries:
            if run != run_id:
                out.setdefault(run, []).append(entity)
        return {run: tuple(sorted(entities)) for run, entities in sorted(out.items())}


class _Tree:
    """The library's class tree, as the classifier sees it: leaves are classes with features."""

    def __init__(self, library: Library) -> None:
        self.library = library
        self.classes = {c["class_id"]: c for c in library.classes}
        self.leaves = [cid for cid, c in self.classes.items() if c.get("features")]

    def ancestors(self, class_id: str) -> list[str]:
        chain: list[str] = []
        node = self.classes[class_id].get("parent")
        while node is not None and node in self.classes and node not in chain:
            chain.append(node)
            node = self.classes[node].get("parent")
        return chain

    def leaves_under(self, class_id: str) -> set[str]:
        return {leaf for leaf in self.leaves if leaf == class_id or class_id in self.ancestors(leaf)}

    def related(self, a: str, b: str) -> bool:
        return a == b or a in self.ancestors(b) or b in self.ancestors(a)

    def label(self, class_id: str) -> str:
        return str(self.classes[class_id]["label"])

    @property
    def named(self) -> str:
        return f"library {self.library.name} {self.library.version}"

    @property
    def note(self) -> str:
        return f"Library {self.library.name} {self.library.version}."


def _check_keys(table: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise CorrectionsError(
            f"{where}: unknown key(s) {', '.join(unknown)}; expected any of {', '.join(sorted(allowed))}"
        )


def _text(value: Any, key: str, where: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorrectionsError(f"{where}: {key} must be a non-empty string, got {value!r}")
    text = " ".join(value.split())
    if len(text) > limit:
        raise CorrectionsError(f"{where}: {key} is {len(text)} characters; at most {limit}")
    words = forbidden_words(text)
    if words:
        raise CorrectionsError(
            f"{where}: {key} uses forbidden vocabulary ({', '.join(words)}); Vigilans never publishes it "
            f"(rule 4) -- reword it"
        )
    return text


def _identifier(value: str, what: str, where: str) -> str:
    if not value.strip() or len(value) > MAX_ID:
        raise CorrectionsError(f"{where}: {what} must be 1 to {MAX_ID} characters, got {value!r}")
    return value


def _class_ids(value: Any, key: str, where: str) -> tuple[str, ...]:
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not items or not all(isinstance(i, str) and i.strip() for i in items):
        raise CorrectionsError(f"{where}: {key} must be a class id or a non-empty list of class ids")
    return tuple(dict.fromkeys(i.strip() for i in items))


def _entry(
    run_id: str, entity_id: str, table: Mapping[str, Any], where: str, tree: _Tree | None
) -> Correction:
    _check_keys(table, _ENTRY_KEYS, where)
    if "note" not in table:
        raise CorrectionsError(
            f"{where}: a correction needs a note saying why -- it is an operator taking responsibility "
            f"for an assessment, and the note is what travels with it"
        )
    note = _text(table["note"], "note", where, MAX_NOTE)
    by = _text(table["by"], "by", where, MAX_BY) if "by" in table else None
    confirm: str | None = None
    if "confirm" in table:
        if not isinstance(table["confirm"], str):
            raise CorrectionsError(f"{where}: confirm names one class id")
        confirm = _class_ids(table["confirm"], "confirm", where)[0]
    reject = _class_ids(table["reject"], "reject", where) if "reject" in table else ()
    affiliation: StandardIdentity | None = None
    if "affiliation" in table:
        value = table["affiliation"]
        if value not in IDENTITIES:
            raise CorrectionsError(
                f"{where}: affiliation {value!r} is not a standard identity; expected one of "
                f"{', '.join(IDENTITIES)}"
            )
        affiliation = value
    if confirm is None and not reject and affiliation is None:
        raise CorrectionsError(f"{where}: says nothing to do; give confirm, reject and/or affiliation")
    if confirm is not None and confirm in reject:
        raise CorrectionsError(f"{where}: confirms and rejects {confirm}")
    if tree is not None:
        library = f"library {tree.library.name} {tree.library.version}"
        unknown = [c for c in (confirm, *reject) if c is not None and c not in tree.classes]
        if unknown:
            raise CorrectionsError(f"{where}: {', '.join(unknown)} is not a class in {library}")
        if confirm is not None:
            above = [r for r in reject if r in tree.ancestors(confirm)]
            if above:
                raise CorrectionsError(
                    f"{where}: confirms {confirm} but rejects {above[0]}, which contains it"
                )
            left = tree.leaves_under(confirm) - set[str]().union(*(tree.leaves_under(r) for r in reject))
            if not left:
                raise CorrectionsError(f"{where}: confirms {confirm} but rejects every class under it")
    return Correction(
        run_id=run_id,
        entity_id=entity_id,
        note=note,
        declared_in=where,
        by=by,
        confirm=confirm,
        reject=reject,
        affiliation=affiliation,
    )


def parse_corrections(text: str, name: str, library: Library | None = None) -> CorrectionsFile:
    """Validate a corrections document. Class ids are checked when a library is given."""
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise CorrectionsError(f"{name}: not valid TOML: {error}") from error
    _check_keys(document, _TOP_KEYS, name)
    runs = document.get("runs", {})
    if not isinstance(runs, dict):
        raise CorrectionsError(f"{name}: runs must be a table of run ids")
    tree = _Tree(library) if library is not None else None
    entries: dict[tuple[str, str], Correction] = {}
    for run_id, run_table in runs.items():
        where_run = f'{name} [runs."{run_id}"]'
        _identifier(run_id, "a run id", where_run)
        if not isinstance(run_table, dict):
            raise CorrectionsError(f"{where_run}: expected a table")
        _check_keys(run_table, _RUN_KEYS, where_run)
        entities = run_table.get("entities", {})
        if not isinstance(entities, dict):
            raise CorrectionsError(f"{where_run}: entities must be a table of entity ids")
        for entity_id, table in entities.items():
            where = f'{name} [runs."{run_id}".entities."{entity_id}"]'
            _identifier(entity_id, "an entity id", where)
            if not isinstance(table, dict):
                raise CorrectionsError(f"{where}: expected a table")
            entries[(run_id, entity_id)] = _entry(run_id, entity_id, table, where, tree)
    return CorrectionsFile(name, entries)


def load_corrections(path: Path, library: Library | None = None) -> CorrectionsFile:
    """Read and validate a corrections file. A byte-order mark is tolerated (Windows editors)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise CorrectionsError(f"cannot read corrections {path}: {error}") from error
    return parse_corrections(text, path.name, library)


# --- the book: the file, re-read when it changes ------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorrectionNotice:
    """Something the run should say, as a picture notice (``publisher.notice``)."""

    severity: Severity
    code: str
    text: str


@dataclass(frozen=True, slots=True)
class Reload:
    """What a read of the file changed. ``ok`` is False when the file on disk is not in use."""

    ok: bool
    added: tuple[str, ...] = ()
    changed: tuple[str, ...] = ()
    withdrawn: tuple[str, ...] = ()
    other_runs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    error: str | None = None
    notices: tuple[CorrectionNotice, ...] = ()

    @property
    def affected(self) -> tuple[str, ...]:
        """Entity ids whose published classification or affiliation must be recomputed."""
        return tuple(sorted({*self.added, *self.changed, *self.withdrawn}))


Stamp = tuple[int, int]


class CorrectionBook:
    """The corrections for one run, from one file, re-read when its modification time or size changes."""

    def __init__(self, path: Path | None, run_id: str, library: Library | None = None) -> None:
        self.path = path
        self.run_id = run_id
        self.library = library
        self.entries: dict[str, Correction] = {}
        self.other_runs: dict[str, tuple[str, ...]] = {}
        #: The problem with the file on disk, while the last good corrections stay in use.
        self.error: str | None = None
        self._stamp: Stamp | None = None
        self._read_once = False

    # --- reading ------------------------------------------------------------------------------

    def _stat(self) -> Stamp | None:
        assert self.path is not None
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def load(self) -> Reload:
        """The first read, at start. A file that exists but cannot be used raises ``CorrectionsError``.

        Like the configuration, a bad file at start stops the run: there is nothing good to
        keep yet, and starting without the operator's corrections would be a silent fallback.
        A file not there yet is fine; it is read when it appears.
        """
        if self.path is None:
            return Reload(
                ok=True,
                notices=(
                    CorrectionNotice(
                        "info",
                        "corrections_unavailable",
                        "No operator corrections file is configured, so no classification can be confirmed "
                        "or rejected and no entity affiliation declared during this run.",
                    ),
                ),
            )
        result = self._read()
        assert result is not None
        if result.error is not None:
            raise CorrectionsError(result.error)
        return result

    def reload_if_changed(self) -> Reload | None:
        """Re-read the file if its modification time or size changed; None when neither has."""
        if self.path is None:
            return None
        return self._read()

    def _read(self) -> Reload | None:
        assert self.path is not None
        first = not self._read_once
        self._read_once = True
        stamp = self._stat()
        if not first and stamp == self._stamp:
            return None
        previous, self._stamp = self._stamp, stamp
        if stamp is None:
            return self._missing(first, had_file=previous is not None)
        try:
            parsed = load_corrections(self.path, self.library)
        except CorrectionsError as error:
            self.error = str(error)
            kept = len(self.entries)
            return Reload(
                ok=False,
                error=self.error,
                notices=(
                    CorrectionNotice(
                        "error",
                        "corrections_invalid",
                        f"The operator corrections file cannot be used: {error}. Keeping the {kept} "
                        f"correction(s) for this run last read from it successfully; nothing in the new "
                        f"version applies until it is fixed.",
                    ),
                ),
            )
        return self._accept(parsed, first)

    def _missing(self, first: bool, *, had_file: bool) -> Reload:
        assert self.path is not None
        if first or not had_file:
            text = (
                f"The operator corrections file {self.path} is not there yet; it is read when it appears "
                f"and whenever it changes."
            )
            return Reload(ok=True, notices=(CorrectionNotice("info", "corrections_absent", text),))
        if not self.entries:
            text = (
                f"The operator corrections file {self.path} was removed; it held no corrections for this run."
            )
            return Reload(ok=True, notices=(CorrectionNotice("info", "corrections_absent", text),))
        self.error = f"{self.path} was removed"
        text = (
            f"The operator corrections file {self.path} was removed. Keeping the {len(self.entries)} "
            f"correction(s) for this run last read from it; to withdraw corrections, remove their "
            f"entries from the file rather than deleting the file."
        )
        return Reload(
            ok=False, error=self.error, notices=(CorrectionNotice("warning", "corrections_removed", text),)
        )

    def _accept(self, parsed: CorrectionsFile, first: bool) -> Reload:
        new = parsed.for_run(self.run_id)
        others = parsed.other_runs(self.run_id)
        old = self.entries
        added = tuple(sorted(e for e in new if e not in old))
        changed = tuple(sorted(e for e in new if e in old and new[e] != old[e]))
        withdrawn = tuple(sorted(e for e in old if e not in new))
        notices: list[CorrectionNotice] = []
        for entity in added:
            notices.append(
                CorrectionNotice(
                    "info", "correction_applied", f"Operator correction: {new[entity].summary()}"
                )
            )
        for entity in changed:
            notices.append(
                CorrectionNotice(
                    "info", "correction_changed", f"Operator correction changed: {new[entity].summary()}"
                )
            )
        for entity in withdrawn:
            notices.append(
                CorrectionNotice(
                    "info",
                    "correction_withdrawn",
                    f"Operator correction withdrawn: {entity} (was {old[entity].summary()})",
                )
            )
        if others and (first or others != self.other_runs):
            listed = "; ".join(f"{run}: {', '.join(entities)}" for run, entities in others.items())
            notices.append(
                CorrectionNotice(
                    "warning",
                    "corrections_other_run",
                    f"{parsed.name} has corrections for other runs, ignored in this one ({self.run_id}), "
                    f"because entity ids are only unique within a run: {listed}.",
                )
            )
        if self.library is None and any(c.touches_classification for c in new.values()):
            notices.append(
                CorrectionNotice(
                    "warning",
                    "corrections_no_library",
                    "No classification library is loaded, so confirmations and rejections in the "
                    "corrections file cannot be checked or applied; affiliations still apply.",
                )
            )
        if not notices:
            notices.append(
                CorrectionNotice(
                    "info",
                    "corrections_read",
                    f"Read {parsed.name}: {len(new)} correction(s) for this run, none changed.",
                )
            )
        self.entries, self.other_runs, self.error = new, others, None
        return Reload(True, added, changed, withdrawn, others, None, tuple(notices))

    # --- using -----------------------------------------------------------------------------------

    def for_entity(self, run_id: str, entity_id: str) -> Correction | None:
        """This run's correction for an entity. Other runs' entries never apply (they are reported)."""
        if run_id != self.run_id:
            return None
        return self.entries.get(entity_id)

    def affiliation_for(self, run_id: str, entity_id: str) -> dict[str, Any] | None:
        """A picture.v0 affiliation with basis ``operator``, or None when none is declared."""
        correction = self.for_entity(run_id, entity_id)
        if correction is None or correction.affiliation is None:
            return None
        return {
            "identity": correction.affiliation,
            "basis": "operator",
            "reasons": [
                f"Declared {correction.affiliation.replace('_', ' ')} by {correction.operator}: "
                f"{correction.note} ({correction.declared_in})."
            ],
        }

    def unmatched(self, known: Iterable[str]) -> list[str]:
        """Entity ids this run's corrections name that are not (yet) in ``known``."""
        seen = set(known)
        return sorted(e for e in self.entries if e not in seen)

    def describe(self) -> str:
        """One line for the run header."""
        if self.path is None:
            return "unavailable (no corrections file configured)"
        state = f"{len(self.entries)} for this run"
        if self.other_runs:
            state += f", {sum(len(v) for v in self.other_runs.values())} for other runs ignored"
        if self.error:
            state += f"; ERROR, keeping the last good: {self.error}"
        elif self._stamp is None:
            state = "not there yet; read when it appears"
        return f"{self.path} ({state})"


# --- applying -------------------------------------------------------------------------------------


def overlay_affiliation(
    affiliation: dict[str, Any], sidc: str, operator: dict[str, Any] | None
) -> tuple[dict[str, Any], str]:
    """Precedence: operator, then library, then default. The symbol's identity follows the winner."""
    if operator is None:
        return affiliation, sidc
    reasons = list(operator["reasons"])
    if affiliation.get("basis") == "library":
        reasons.append(
            f"Replaces the library's {affiliation['identity'].replace('_', ' ')} "
            f"(an operator declaration takes precedence over the library)."
        )
        reasons.extend(f"Library, not applied: {r}"[:2000] for r in affiliation.get("reasons", []))
    return {**operator, "reasons": reasons[:50]}, sidc_with_identity(sidc, operator["identity"])


def _join(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def _rejection_text(correction: Correction, tree: _Tree, class_id: str) -> str:
    return (
        f"Rejected by {correction.operator}: {tree.label(class_id)} ({class_id}). Why: {correction.note} "
        f"({correction.declared_in}). The remaining probability is renormalised over the other classes "
        f"and none-of-the-above."
    )


def apply(
    assessment: Assessment,
    correction: Correction | None,
    library: Library | None,
    *,
    settings: ClassifierSettings | None = None,
) -> Assessment:
    """The assessment with an entity's operator corrections applied (affiliation is separate).

    Rejection conditions the posterior on "not that class": the rejected classes' probability
    is removed and the rest renormalised over the remaining classes and the background, then
    the classifier's own decision rules are applied again. Evidence against the background is
    unchanged by a rejection, so the sequential test for "likely" is recovered from the ratio
    of each class's posterior to the background's (``settings`` must be the classifier's).

    Confirmation words the class "likely" and keeps its computed probability as the number.
    """
    if correction is None or library is None or not correction.touches_classification:
        return assessment
    tree = _Tree(library)
    s = settings or ClassifierSettings()
    notes = [
        f"The operator's correction names {c}, which {tree.named} does not have; it is not applied "
        f"({correction.declared_in})."
        for c in (correction.confirm, *correction.reject)
        if c is not None and c not in tree.classes
    ]
    rejected = [r for r in correction.reject if r in tree.classes]
    removed: set[str] = set[str]().union(*(tree.leaves_under(r) for r in rejected))
    confirm = correction.confirm if correction.confirm in tree.classes else None
    if confirm is not None and not tree.leaves_under(confirm) - removed:
        notes.append(f"The confirmation of {confirm} is not applied: every class under it is rejected.")
        confirm = None
    rejections = [_rejection_text(correction, tree, r) for r in rejected]

    if not assessment.posteriors:
        # Nothing computed yet (insufficient evidence): no number to show, so nothing to confirm.
        pending = (
            f"Confirmed by {correction.operator} as {tree.label(confirm)}: {correction.note} "
            f"({correction.declared_in}); it is shown once there is enough evidence to compute a "
            f"probability for it."
            if confirm is not None
            else ""
        )
        return replace(assessment, reason=_join(assessment.reason, pending, *rejections, *notes))

    before = assessment.posteriors
    background_before = assessment.background_probability
    remaining = [leaf for leaf in tree.leaves if leaf not in removed]
    # Nothing removed: nothing to renormalise (and no rounding to introduce).
    mass = background_before + sum(before.get(leaf, 0.0) for leaf in remaining) if removed else 1.0
    if not remaining or mass <= 0.0:
        return Assessment(
            "unclassified",
            kind="unlike_library",
            reason=_join(
                f"Unclassified: every class in {tree.named} that could explain it has been rejected "
                f"by the operator.",
                *rejections,
                *notes,
            ),
            features_used=assessment.features_used,
            background_probability=1.0,
            posteriors={cid: 0.0 for cid in tree.classes},
        )
    posterior = {cid: 0.0 for cid in tree.classes}
    for leaf in remaining:
        posterior[leaf] = before.get(leaf, 0.0) / mass
    for leaf in remaining:
        for ancestor in tree.ancestors(leaf):
            posterior[ancestor] += posterior[leaf]
    background = background_before / mass

    class_prior = (1.0 - s.background_prior) / len(tree.leaves)

    def evidence(cid: str) -> float:
        """The class's tempered log score against the background, as the classifier computed it."""
        if cid not in tree.leaves:
            return SPRT_UPPER  # a category passes the sequential test, as in the classifier
        p = before.get(cid, 0.0)
        if background_before <= 0.0:
            return math.inf
        if p <= 0.0:
            return -math.inf
        return math.log(p / background_before) + math.log(s.background_prior / class_prior)

    def level(cid: str) -> str:
        p = posterior[cid]
        if p >= LIKELY and evidence(cid) >= SPRT_UPPER:
            return "likely"
        return "possible" if p >= POSSIBLE else "not shown"

    original = {c.class_id: c for c in assessment.candidates}

    def computed_reasons(cid: str) -> list[str]:
        if cid in original:
            return [r for r in original[cid].reasons if r != tree.note]
        p, was = posterior[cid], before.get(cid, 0.0)
        if cid not in tree.leaves:
            return [
                f"Its sub-classes together account for {p:.2f} ({was:.2f} before the operator's correction)."
            ]
        score = evidence(cid)
        against = (
            f"; evidence for it over none-of-the-above: log ratio {score:+.1f}"
            if math.isfinite(score)
            else ""
        )
        return [f"Probability {p:.2f} after the operator's correction, {was:.2f} before it{against}."]

    def candidate(cid: str, *, lead: Iterable[str] = (), cap: bool = False) -> Candidate:
        cls = tree.classes[cid]
        wording = "possible" if cap or level(cid) != "likely" else "likely"
        return Candidate(
            class_id=cid,
            label=tree.label(cid),
            probability=posterior[cid],
            wording=f"{wording} {tree.label(cid)}",
            reasons=(*lead, *computed_reasons(cid), *rejections, *notes, tree.note),
            affiliation=cls.get("affiliation") if wording == "likely" else None,
            sidc_template=(cls.get("symbol") or {}).get("sidc_template"),
        )

    ranked = sorted(remaining, key=lambda c: -posterior[c])

    if confirm is not None:
        p = posterior[confirm]
        computed = level(confirm)
        shown = f"which it would word '{computed}'" if computed != "not shown" else "below what it would show"
        confirmed = candidate(
            confirm,
            lead=(
                f"Confirmed by {correction.operator}: {correction.note} ({correction.declared_in}).",
                f"Worded 'likely' because an operator confirmed it. The number is the classifier's own "
                f"computed probability, {p:.2f}, {shown}.",
            ),
        )
        confirmed = replace(
            confirmed,
            wording=f"likely {tree.label(confirm)}",
            affiliation=_confirmed_affiliation(tree.classes[confirm], correction),
        )
        alternatives = [
            candidate(
                cid,
                lead=(
                    f"The classifier's computed alternative; {correction.operator} confirmed "
                    f"{tree.label(confirm)} instead.",
                ),
                cap=True,
            )
            for cid in ranked
            if posterior[cid] >= POSSIBLE and not tree.related(cid, confirm)
        ][:2]
        return Assessment(
            "classified",
            (confirmed, *alternatives),
            features_used=assessment.features_used,
            background_probability=background,
            posteriors=posterior,
        )

    def unclassified(kind: Literal["ambiguous", "unlike_library"], text: str) -> Assessment:
        return Assessment(
            "unclassified",
            kind=kind,
            reason=_join(text, *rejections, *notes, tree.note),
            features_used=assessment.features_used,
            background_probability=background,
            posteriors=posterior,
        )

    def classified(*chosen: Candidate) -> Assessment:
        return Assessment(
            "classified",
            chosen,
            features_used=assessment.features_used,
            background_probability=background,
            posteriors=posterior,
        )

    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    p_best = posterior[best]
    if p_best >= POSSIBLE:
        if runner is not None and posterior[runner] >= p_best - AMBIGUITY_GAP:
            shared = [a for a in tree.ancestors(best) if a in tree.ancestors(runner)]
            if shared and posterior[shared[0]] >= POSSIBLE:
                return classified(candidate(shared[0]))
            return unclassified(
                "ambiguous",
                f"Ambiguous once the operator's rejection is taken into account: {tree.label(best)} "
                f"({p_best:.2f}) and {tree.label(runner)} ({posterior[runner]:.2f}) fit about equally well.",
            )
        chosen = [candidate(best)]
        if runner is not None and posterior[runner] >= POSSIBLE:
            chosen.append(candidate(runner))
        return classified(*chosen)
    if background >= 0.6:
        return unclassified(
            "unlike_library",
            f"Unclassified once the operator's rejection is taken into account: none of the remaining "
            f"classes explains this better than chance (nearest: {tree.label(best)}, {p_best:.2f}).",
        )
    return unclassified(
        "ambiguous",
        f"Unclassified once the operator's rejection is taken into account: no remaining class is "
        f"supported strongly enough (best: {tree.label(best)}, {p_best:.2f}).",
    )


def _confirmed_affiliation(cls: Mapping[str, Any], correction: Correction) -> dict[str, Any] | None:
    declared = cls.get("affiliation")
    if not declared:
        return None
    return {
        **declared,
        "reasons": [
            *declared.get("reasons", []),
            f"The class was confirmed by {correction.operator} ({correction.declared_in}).",
        ],
    }
