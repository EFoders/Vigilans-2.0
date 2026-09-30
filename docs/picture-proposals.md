# picture.v0 — proposals from the Vigilans side

picture.v0 belongs to Videns and is changing there; Vigilans does not edit it (ADR-0002).
These are the gaps Vigilans has hit, for the Videns side to take or refuse. Until then
Vigilans works within the current schema, as noted against each.

| # | Proposal | Why | Meanwhile |
|---|---|---|---|
| 1 | **Altitude uncertainty.** `position.alt_m` gains a sibling `alt_uncertainty: {basis, sigma_m}`, required with `alt_m` | A height without its uncertainty is drawn as exact (rule 5). Heights now exist (elevation-capable sensors, reported altitudes) | Height sent as text in `meta.height`; `alt_m` not sent |
| 2 | **Unclassified kind.** `classification.unclassified_kind`: `insufficient` / `ambiguous` / `unlike_library` | classifier-design.md §5.6: the three are different answers | Folded into `classification.reason` |
| 3 | **Class-tree level.** `candidate.level` (e.g. `category`, `type`) | Answers are given at the deepest confident level (§5.4) | Implied by the class id |
| 4 | **Purpose candidates.** `classification.purpose[]`, same shape as candidates | "What for" is a separate output from "what" (§5.5) | Not sent until Phase 5 |
| 5 | **Source claims.** `source_claims[]` on an entity: scheme, category, claimed id, count | ADR-0006: a self-identification broadcast is shown as the source's claim, beside the behaviour | Not sent until Phase 5 |
| 6 | **Coverage per sensor** on the sensor object: bands and listening fraction | A reader needs to see why timing features are or are not trustworthy (§3.2) | Not sent |
| 7 | **The `?` badge collides with 2525's unknown battle dimension.** An entity whose dimension is unknown (`S?ZP…`) renders a `?` inside the frame, which reads as Videns' "unreported: no region drawn" badge | Tracks with measured regions look unreported (seen live, 2026-09-30) | Vigilans keeps `Z`: nothing yet says ground or air. A different badge glyph, or drawing the badge outside the frame, would fix it |
| 8 | **Departures and lineage on the map.** A link from each `split_from` entity to its parent, and a count of departures on the parent | ADR-0009: an entity six others departed from is the owner's "command post" pattern; today it is only in `meta.departures` | Sent as `lineage.split_from` and `meta.departures` |
| 9 | **Motion-model probabilities as a field**, not meta text: `motion: {stationary, moving, manoeuvring}` | They are classifier features and worth a legend | In `meta.motion_model_probabilities` as text |
| 10 | **Re-identification alternatives.** `reidentification.alternatives[]`: `{entity_id \| null, probability}` summing to 1, null being "a different emitter" | ADR-0016: a confidence means little without what else was weighed; twins share it | In `meta.reidentification_weighed` as text |
| 11 | **Operator corrections as their own field** — `classification.operator: {confirmed, rejected[], by, note, declared_in}` | ADR-0015: the reader should see at a glance that a person, not the classifier, set the wording | In the first candidate's reasons |
