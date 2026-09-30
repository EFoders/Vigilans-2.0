# Classification — design

**Status:** built in Phase 5, 2026-09-30 — see [ADR-0013](adr/0013-classifier-as-built.md) for what is built and where it simplifies this design. It was designed
before Phases 2–4, because they produce the features it needs. Decisions:
[ADR-0004](adr/0004-classifier-is-a-library-driven-bayesian-model.md) (the approach) and
[ADR-0005](adr/0005-no-propagation-models-no-pulsed-emitters.md) (the scope), and
[ADR-0006](adr/0006-classifier-inputs-history-maps-corrections.md) (the owner's answers in §11).

Classification answers two questions about an entity, separately: **what it probably is**
(class) and **what it is probably for** (purpose). Both are hedged, both carry their reasons,
and "unclassified" is a first-class, common answer (spec §9.2).

---

## 1. Constraints it must honour

| Rule | Consequence for the classifier |
|---|---|
| Externals only (rule 2) | Features are frequency, bandwidth, timing, power *as received*, geometry and behaviour. No content, ever. |
| Hedged, with reasons (rule 4) | Every candidate carries the features that moved it, for and against, in words a reader can check. |
| Uncertainty carried, never invented (rule 5) | Every feature has an uncertainty and a basis. Missing is missing — never zero, never a default. |
| Knowledge is data, not code (§9.1) | All class knowledge lives in the library. The engine holds no class-specific numbers. |
| No real parameters in the tree (rule 6) | The shipped library is synthetic. Real value comes from a private library loaded from outside the tree. |
| No propagation models (ADR-0005) | No transmit power (EIRP), no path-loss ranging, no radio-horizon height, no "should have been heard" reasoning. |
| Communications emitters only (ADR-0005) | No pulse-train analysis. Pulsed emitters are out of scope for now. |

## 2. Shape of the system

```
observations ─▶ Phases 2–4: fixes, tracks, entities
                     │
                     ▼
              feature extraction   (§3–4; each feature: value, sigma, n, basis)
                     │
                     ▼
              evidence per class   (§5; library likelihoods, measurement uncertainty folded in)
                     │
                     ▼
              accumulation         (§6; over time, with decay and change-point resets)
                     │
                     ▼
              decision             (§5.6; sequential test, hierarchy, open-set)
                     │
                     ▼
              class + purpose, hedged wording, reasons ─▶ picture.v0 classification
```

## 3. Features

Every feature is produced as a **feature record**:

| Field | Meaning |
|---|---|
| `value` | The estimate |
| `sigma` | 1-sigma uncertainty of the estimate, in the feature's unit |
| `n` | Samples behind it (transmissions, intervals, fixes) |
| `window_s` | The span of picture time it summarises |
| `basis` | `measured`, `assumed`, `mixed` or `unreported`, inherited from the inputs (rule 5) |
| `status` | `ok`, `insufficient` (below the feature's minimum `n`), or `unavailable` (the input does not exist — e.g. no elevation-capable sensor) |

Only `ok` features contribute evidence. `insufficient` and `unavailable` are reported, so the
inspector can say "period not yet estimable: 2 of 3 intervals needed".

### 3.1 Catalogue

| Family | Feature | Algorithm | Produced by |
|---|---|---|---|
| **Position** | lat, lon | Fix / track estimate with covariance | Phase 2–3 |
| | Location context | Membership of library-declared regions (private library only; synthetic regions in the repo), and site priors (§6.3) | Phase 5 |
| **Height** | Height above ellipsoid | From elevation angles (sensors that measure them) or from position sources that report altitude. Nothing is inferred when neither exists: `unavailable` | Phase 2 |
| | Height above ground | Height minus terrain elevation, if terrain data is supplied (§11 Q3) | Phase 3 |
| | Climb rate, hover | Tracker vertical state; hover = airborne *and* stationary-model probability high | Phase 3 |
| **Motion** | Stationary / moving / road-constrained | Probabilities from an interacting-multiple-model tracker (stationary, constant-velocity, coordinated-turn, road-constrained) | Phase 3 |
| | Ground speed, max speed, acceleration, turn rate | Tracker state with covariance | Phase 3 |
| | On a road | HMM map matching (Newson & Krumm), scored against the chance that a random track with the same ellipses would match as well — "near a road" is meaningless in a dense network otherwise. Needs road data (§11 Q3) | Phase 3 |
| | On water | Region membership against water polygons, if supplied | Phase 3 |
| | Stop–start pattern | Durations of stationary-model episodes | Phase 5 |
| **Frequency** | Centre frequency | Weighted mean over transmissions | Phase 5 |
| | Occupied bandwidth | Weighted mean and spread | Phase 5 |
| | Channel raster | Residue test of centre frequencies against candidate channel spacings | Phase 5 |
| | Stability / drift | Least-squares drift rate; Allan deviation once `n` allows. Also a fingerprint input (Phase 6) | Phase 5 |
| | Frequency hopping | Transmissions grouped by timing coherence (no overlap, constant dwell) and geometry, *not* frequency; yields hop set, dwell, hop rate. Runs before entity resolution, or one hopper becomes dozens of entities | Phase 4 |
| | Modulation hint | As reported by the source; categorical | Phase 1 (already carried) |
| **Time** | Transmission length | Distribution of `duration_s`: median, spread | Phase 5 |
| | Transmission period | Histogram of gaps between transmission starts, cumulative over gap orders so a missed transmission does not halve the estimate; confirmed by folding start times at the candidate period | Phase 5 |
| | Regularity | Coefficient of variation of the gaps and the burstiness parameter: ≈0 clockwork, ≈1 random, >1 bursty | Phase 5 |
| | Time-slot grid | Start times folded over candidate slot lengths; Rayleigh test for phase clustering | Phase 5 |
| | Duty cycle | Time on air ÷ time **observed** (§3.2) | Phase 5 |
| | Persistence | Fraction of observed time the entity is active | Phase 5 |
| | Time-of-day pattern | Circular statistics over the 24-hour day | Phase 6 |
| | Behaviour change | Bayesian online change-point detection (Adams & MacKay) over period, duration, speed; a change restarts the affected feature windows | Phase 5 |
| **Power** | Received power, as reported | Per sensor, over time. Used only as a *relative* and *temporal* quantity: its stability at one sensor for a stationary emitter, and whether it is near a sensor's reported ceiling. Never converted to transmit power or range (ADR-0005) | Phase 5 |
| **Interference** | Jamming indicators | §7 | Phase 5 |

### 3.2 Coverage: what "observed" means

Period, duty cycle, persistence and time-of-day are only honest if we know when each sensor
was actually listening on the entity's frequency. A scanning sensor that visits a channel
20 % of the time makes a 10 s emitter look irregular and slow. So these features are computed
over **covered time** only, from coverage records (§8). A source that does not report coverage
has its timing features marked `basis: unreported` and weakened (§5.3) — never assumed to
have listened continuously.

## 4. Minimum evidence per feature

Each feature has a minimum `n` below which it is `insufficient` (library defaults, overridable
per class): e.g. period needs ≥ 3 gaps, regularity ≥ 5, time-slot grid ≥ 8 starts, speed ≥ 2
fixes over ≥ 30 s, time-of-day ≥ 24 h of covered time. These are engine defaults, not class
knowledge, so they may live in the repository.

## 5. The classifier

### 5.1 Model

A generative Bayesian classifier whose likelihoods are **written in the library**, not trained.
For entity features **x** and class *c*:

```
score(c) = log P(c | prior) + Σ_blocks  τ · w_b · log [ P(x_b | c) / P(x_b | background) ]
```

- **Likelihoods** `P(x | c)` come from the library as soft shapes: a range with soft edges
  (uniform convolved with a Gaussian edge), a Gaussian, or a categorical table. No hard cut-offs:
  a feature just outside a range lowers the score smoothly rather than vetoing the class.
- **Background** `P(x | background)` is an explicit "none of the above" model (broad, from the
  library's band-level defaults). Scores are evidence *against* it, so unclassified is a real
  outcome, not the lowest score.
- **Blocks.** Features that depend on each other are scored together as one block through a
  small, hand-structured Bayesian network, so the same evidence is not counted twice:
  *motion* (speed, stationary/moving/road probabilities, turn rate), *height* (height, climb,
  hover), *timing* (length, period, regularity, slot grid, duty cycle), *spectrum* (frequency,
  bandwidth, raster, hopping, modulation hint). Remaining features are independent terms.
- **Tempering** `τ ≤ 1` removes the residual over-confidence naive combination always has. It
  is 1 until replay with adjudicated labels exists to fit it (spec §13 level 2).
- **Block weights** `w_b` default to 1; the library may lower them per class.

### 5.2 Measurement uncertainty

A feature estimate `x̂ ± σ` is compared with the library shape by integrating over it:
`P(x̂ | c) = ∫ p(x | c) · N(x; x̂, σ) dx`. A fuzzy measurement therefore gives weaker evidence
for *and* against, automatically. Tracker outputs that are already probabilities (stationary
0.93) enter as soft ("virtual") evidence.

### 5.3 Unreported and assumed inputs

A feature whose inputs include `unreported` uncertainty contributes with weight
`w_unreported` (default 0.5, engine-level) and is flagged in the reasons. `assumed` inputs
contribute fully but are named as assumptions in the reasons. `mixed` reports the breakdown.
Nothing is ever given a sigma it did not have.

### 5.4 Hierarchy

Library classes form a tree (e.g. *ground emitter → mobile → vehicle net radio*). Posteriors
are summed up the tree; the answer is the **deepest node** that clears the decision threshold.
A reader sees "likely ground mobile emitter" when the fine type is still ambiguous, rather than
a coin-flip between two leaves.

### 5.5 Purpose

Purpose is a separate output with its own library tables (*what for*: e.g. synthetic
"command net", "telemetry", "interference"). It takes the class posterior as input, plus
behaviour, plus — from Phase 7 — the entity's role in groups (controller of a fast mover, hub
of a peer group). Group context feeds **purpose only, never class**, so classification and
grouping cannot reinforce each other in a loop.

**Departure pattern (project owner, 2026-09-30).** When several entities split away from one
that stays stationary, the stationary one is likely a command post, forward base or similar
fixed site, and the departed ones are likely subordinate to it. Resolution (Phase 4) records
every split with its lineage and keeps a per-entity list of departures, with the parent's
stationary-model probability at the time of each (ADR-0009). Phase 7 turns that into a
`superior/subordinate` group with the departures as evidence; purpose (Phase 5) may then say,
hedged, "possible command post or fixed site — 6 entities departed from it while it stayed
stationary". The departure count alone never asserts a role: it is weighed like any other
evidence, and a car park that six vehicles leave is the negative control.

### 5.6 Deciding, and the three kinds of unclassified

Evidence accumulates (§6); a candidate is promoted by a **sequential probability ratio test**
(Wald), which commits only when the evidence crosses a threshold with a stated error rate:

| Posterior | Wording |
|---|---|
| ≥ 0.70 and SPRT committed | "likely …" |
| ≥ 0.40 | "possible …" |
| below | not shown as a candidate |

Thresholds are engine defaults, recorded in every assessment's provenance.

"Unclassified" always says which of three things it is:

1. **Insufficient evidence** — not enough yet; the reasons say what is missing ("period needs
   one more interval").
2. **Ambiguous** — two or more candidates close together; both are named.
3. **Unlike the library** — every class, *and* the background, explains the features poorly
   (a typicality test on the per-feature fit). This is how something new gets noticed rather
   than forced into the nearest box.

### 5.7 Reasons

Reasons are generated from the arithmetic, not written separately: each block's contribution to
the log score, largest first, for and against, with the measured value and what the library
expected. For example:

> likely synthetic wideband data link (0.78) — bandwidth 210 ± 15 kHz, within 100–500 kHz
> (strongly for); gaps 9.8 s, regular, CV 0.05 (for); stationary 0.93 (weakly for).
> Against: received power varies more than a fixed site usually shows (weakly).
> Library synthetic 0.2.0. Bearing uncertainty for LEGACY-DF is an operator assumption.

Library name and version, and any assumed or unreported inputs, are always in the reasons.

## 6. History

### 6.1 Within an entity

The per-class log scores are accumulated with **exponential forgetting** (half-life in picture
time, engine default, library-overridable per class), so behaviour that changes is not outvoted
by stale history. A detected change point (§3.1) restarts the affected feature windows and
reduces the accumulated score for the affected block.

### 6.2 Across gaps

When re-identification (spec §8, Phase 6) says a new entity is consistent with an earlier one
at confidence *r*, the new entity starts from a mixture prior
`r · posterior(old) + (1 − r) · prior(default)`. Re-identification confidence therefore bounds
how much of the past carries forward, and the reasons say "carried forward from E17 (consistent
with, 0.6)".

### 6.3 Across runs — deferred (§11, decision 2)

Not built yet. When it is: site and band base rates held as Dirichlet counts in a store **outside the
repository** (it is real observed data), updated **only from operator-confirmed labels**. The
system never learns from its own unconfirmed conclusions — otherwise its early guesses become
self-fulfilling. Used as `log P(c | prior)`.

## 7. Jamming and interference

Without propagation models, transmit power cannot be estimated, so "high power" is not
available as evidence. Interference is recognised by behaviour instead, as a library
behavioural-signature class, from indicators that need no propagation:

- duty cycle near 1 over covered time, and long or ongoing transmissions;
- wide occupied bandwidth; noise-like modulation hint if the source gives one;
- a rise in reported noise floor at **several sensors at once** in the affected band (§8);
- a change point in the detection rates of *other* entities in that band, coinciding in time.

Each is evidence, not a verdict; the wording stays hedged ("possible interference source").

## 8. Contract changes this needs

**observation.v1** — additive and optional, made before any third-party adapter exists:

- `transmission: "complete" | "ongoing"` (and a later record closing it), so continuous
  emitters are representable;
- a new record `kind: "coverage"`: which sensor listened to which frequency range, from when to
  when (dwell/scan windows) — the denominator for every timing feature (§3.2);
- a new record `kind: "occupancy"`: reported noise floor per sensor per band, over time (§7);
- optional `hop` hints on a transmission, where a source already groups hops itself;
- optional `source_claim: {scheme, category, claimed_id?}` for self-identification broadcasts an
  adapter decoded (§11, decision 1). Fixtures use invented identifiers only (rule 1).

**library.v0 → library.v1**: soft likelihood shapes instead of hard ranges; feature blocks;
the class tree; a purpose section; per-class minimum-evidence and forgetting overrides;
background band models; optional regions (private libraries only hold real ones).

**picture.v0** (Videns' contract — proposed to Videns, not edited here): the unclassified kind
(insufficient / ambiguous / unlike library); the class-tree level of a candidate; purpose
candidates alongside class candidates.

## 9. Which phase builds what

| Phase | Classifier-relevant output |
|---|---|
| 2 Geolocation | Fix with covariance; height from elevation or reported altitude; basis carried |
| 3 Tracking | IMM mode probabilities (stationary / moving / turning / road), speed and turn rate with covariance, vertical state; map matching if road data is available |
| 4 Resolution | Hop grouping; one entity per emitter across sources |
| 5 Classification | Timing, spectrum and power features; classifier; interference indicators; coverage-aware timing |
| 6 Fingerprinting | Re-identification prior carry-forward; time-of-day; cross-run priors if allowed |
| 7 Grouping | Group roles as inputs to purpose |

## 10. Validation

- **Synthetic** (now): per-class confusion matrices on simulated scenarios — internal
  consistency only, and labelled as such.
- **Honesty tests** (always): missing features contribute nothing; unreported inputs are
  weakened and flagged; no candidate without reasons; group context never reaches class;
  "unlike library" fires on a deliberately novel emitter; negative controls stay unclassified.
- **Calibration** (needs adjudicated replay): reliability diagrams and expected calibration
  error; fits `τ`. Until then every published confidence carries the spec §13 caveat.
- **Trained models** may later be added as an extra, private evidence source (loaded like a
  private library), still scored against the background, tempered, and explained. Never the core.

## 11. Decisions (project owner, 2026-09-30) — [ADR-0006](adr/0006-classifier-inputs-history-maps-corrections.md)

1. **Self-identification broadcasts: allowed, labelled.** Vigilans decodes nothing. An adapter
   may attach a `source_claim` to an observation (scheme, claimed category, optional claimed
   identifier). It is evidence of the *source's claim*, weighed like any other feature against
   the entity's behaviour, never trusted outright, and always worded as a claim ("the source
   reports a UAS identity broadcast"). A claim that contradicts behaviour is itself a reason.
2. **History: within a run only, for now.** §6.1 and §6.2 are built; §6.3 (cross-run priors) is
   not, until operator confirmation has been used in practice.
3. **Map data: roads, water and terrain elevation**, all offline, mounted from outside the tree
   and named in configuration (`[maps] roads`, `water`, `terrain`). Terrain gives height above
   ground only — never radio horizon (ADR-0005). Formats are chosen in Phase 3. A map not
   configured makes its features `unavailable`, and the run header says so.
4. **Corrections: an operator declarations file first.** A TOML file Vigilans reads and reloads
   when it changes, like source declarations today: confirm or reject a classification for an
   entity in the current run (later: "not a group"). Each correction is recorded in provenance
   and shown as operator-basis evidence. A Videns write path may follow; it needs authentication.
