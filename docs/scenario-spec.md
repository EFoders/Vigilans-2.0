# scenario.v2 — specification for scenario generators

**Status:** stable enough to generate against (contract 0.3.0, 2026-09-30). Machine-readable
definition: [`contract/schemas/scenario.v2.schema.json`](../contract/schemas/scenario.v2.schema.json).
Validator: `vigilans_contract.validate_scenario`. Runner: `vigilans-hub` (ADR-0010, ADR-0011).

A scenario is a synthetic world: sensors, the systems they report through, and emitters that
move and transmit. The hub simulates it and sends Vigilans exactly what those systems would
report (`observation.v1`); Vigilans never sees the scenario or its truth.

**scenario.v2 is a superset of the Videns editor's scenario.v1.** Every v1 file is valid and
means the same thing. A generator that writes v1 today needs only to set `schema:
scenario.v2` to start using the additions; nothing it writes now has to change.

---

## 1. Conventions

- YAML or JSON. Unknown keys are **errors**, at every level: a typo fails loudly.
- Positions: `[east_m, north_m]` from `origin`, azimuthal equidistant on WGS84 (range and
  bearing from the origin are exact).
- Heights `alt_m`: metres above the WGS84 ellipsoid.
- Frequencies in Hz; times in seconds from the scenario start; the simulator's epoch is
  2026-01-01T00:00:00Z unless the hub is told to restamp to wall time.
- Ids: `[A-Za-z0-9._:-]`, 1–64 characters, unique within their list.
- `label`: free text up to 64 characters, for people. Never interpreted.
- **Rule 1 of the repository:** scenarios committed to it are synthetic — neutral origin,
  invented names, arbitrary frequencies. A scenario with real places, units or equipment is
  fine to run, but keep it outside the repository and mount it into the hub (§9).

## 2. Top level

| Key | Type | Required | Default | Meaning |
|---|---|---|---|---|
| `schema` | `"scenario.v2"` or `"scenario.v1"` | yes | | |
| `name` | text ≤ 64 | yes | | Shown in run headers |
| `description` | text ≤ 4000 | | `""` | |
| `seed` | integer | | 42 | Every random draw derives from it: same seed, same records, byte for byte |
| `duration_s` | number > 0, ≤ 86400 | yes | | |
| `tick_s` | number > 0, ≤ 60 | | 1 | Simulation step |
| `origin` | `{lat, lon}` | yes | | |
| `propagation` | object (§3) | | | |
| `sources` | list (§4) | | `[]` | **v2** |
| `sensors` | list (§5), ≥ 1 | yes | | |
| `nets` | list (§6) | | `[]` | |
| `emitters` | list (§7), ≥ 1 | yes | | |
| `operator` | `{declarations: [...]}` (§8) | | | |
| `truth_relations` | list (§8) | | `[]` | Scorer only |

## 3. `propagation` — what a sensor can hear

Used **only by the simulator** to decide detections and fill `power_dbm`; the engine has no
propagation model (ADR-0005).

| Key | Default | Meaning |
|---|---|---|
| `model` | `log_distance` | The only model |
| `exponent` | 2.7 | Path-loss exponent: 2 is free space, higher is cluttered ground |
| `shadowing_sigma_db` | 4 | Log-normal shadowing per detection; 0 for a noise-free world |

Received power = `eirp_dbm` − (free-space loss at 1 m + 10·exponent·log10(range) + shadowing).
A sensor detects when received power − `noise_floor_dbm` ≥ `threshold_db`.

## 4. `sources` — the systems that report (v2)

A **source** is one reporting system — a DF net, an integrated geolocating system, a legacy
box. Every sensor belongs to one via `sensor.source_id`. A sensor naming a source not in this
list belongs to an implicit bearing source of that id with the defaults below. A v1 file has
no sources: all its sensors report as source `SIM`.

| Key | Default | Meaning |
|---|---|---|
| `id` | required | The `source_id` its observations carry |
| `label` | | |
| `kind` | required | `bearing`: each sensor reports lines of bearing. `position`: the system reports the emitter's position (its sensors are receivers: they decide what it hears) |
| `reports_uncertainty` | `true` | `false`: observations say basis `unreported` and carry no sigma, whatever noise was applied — the honest way to model equipment that gives no error figure |
| `position_sigma_m` | 150 | `position` sources: 1-sigma error applied to each reported position |
| `region_confidence` | 0.95 | `position` sources: confidence of the circular ellipse reported (0.5 models a CEP50 system) |
| `freq_bias_hz` | 0 | Every frequency it reports is high by this much (Vigilans must learn it) |
| `clock_offset_s` | 0 | Every time it reports is late by this much (Vigilans must learn it) |
| `emits_coverage` | `false` | Also send `coverage` records: what each sensor listened to, and when |
| `fault` | none | `{kind, every}`: every `every`-th record is broken on purpose — `flat_sigma`, `radians`, `local_time` or `missing_provenance` — so ingest has something to reject |

## 5. `sensors`

| Key | Default | Meaning |
|---|---|---|
| `id` | required | |
| `label` | | |
| `source_id` | `SIM` | **v2**. §4 |
| `pos_m` | required | Receiver position |
| `alt_m` | 0 | |
| `bearing_sigma_deg` | required for bearing sources | 1-sigma bearing error applied |
| `elevation_sigma_deg` | null | null or absent: cannot measure elevation (the common case). A number: reports elevation from exact geometry plus this noise. Not allowed on position-source receivers |
| `freq_range_hz` | `[20e6, 3e9]` | What it can tune to, `[low, high]` with low < high |
| `noise_floor_dbm` | -110 | |
| `threshold_db` | 6 | SNR needed to detect |
| `report_interval_s` | 1 | At most one report per emitter per interval |
| `p_miss` | 0 | Probability a detectable report is dropped |
| `p_outlier` | 0 | Probability a bearing is replaced by a uniformly random one |
| `false_alarm_rate_hz` | 0 | Clutter: Poisson reports at random frequency and bearing (bearing sources) |
| `scan` | none | **v2**. `{dwell_s, revisit_s}`: a scanning receiver listens to any one frequency for `dwell_s` out of every `revisit_s` (dwell ≤ revisit), and misses what it is not tuned to |

## 6. `nets` — emitters that take turns

Members share a channel and at most one transmits at a time. When the net is quiet, a
non-hub member calls (duration ~ exponential(`mean_tx_s`)); with probability
`hub_reply_prob` the hub replies after a uniform `turnaround_s` gap; then the net is quiet
for ~ exponential(`mean_gap_s`).

| Key | Default |
|---|---|
| `id` | required |
| `label` | |
| `mean_tx_s` | 6 |
| `mean_gap_s` | 20 |
| `hub_reply_prob` | 0.8 |
| `turnaround_s` | `[1, 3]` (low ≤ high) |

## 7. `emitters`

| Key | Default | Meaning |
|---|---|---|
| `id` | required | |
| `label` | | |
| `truth` | | `{class_id, role_id, side}` — scorer only, **never** in an observation. `side`: `friend`, `hostile`, `neutral`, `civilian`, `unknown` or null |
| `path_m` | required | Waypoints. One point: static. Each point is `[east_m, north_m]` or, **v2**, `[east_m, north_m, hold_s]`: wait `hold_s` there before moving on |
| `alt_m` | 0 | |
| `speed_mps` | 0 | Must be > 0 if the path has more than one point |
| `loop` | `false` | After the last waypoint, return to the first and repeat |
| `freq_hz`, `bandwidth_hz` | required | Centre and occupied bandwidth |
| `eirp_dbm` | 40 | |
| `activity` | continuous | §7.1 |
| `active_windows_s` | always | **v2**. `[[start_s, stop_s], ...]`: switched on only inside these (start < stop) |
| `source_claim` | none | **v2**. `{scheme, category, claimed_id?}`: a self-identification broadcast (`remote_id`, `ads_b`, `ais`, `other`) that sources decode and pass on as the source's claim (ADR-0006) |

### 7.1 `activity`

| `type` | Also needs | Behaviour |
|---|---|---|
| `continuous` | | Always transmitting while switched on |
| `periodic` | `period_s`, `on_s` (on ≤ period); **v2** `offset_s` | On for `on_s` at the start of every `period_s`, from `offset_s` |
| `bursty` | `mean_on_s`, `mean_off_s` | Exponentially distributed on and off spells |
| `net` | `net_id` (must exist); `hub` (default false) | §6 |

## 8. `operator` and `truth_relations`

**`operator.declarations`** — what the operator tells Vigilans, and takes responsibility for.
Each becomes that source's `[sources.<id>]` configuration in the engine (not sent over the
network: it is engine configuration). `source_id` must be one some sensor reports as.

| Key | Meaning |
|---|---|
| `source_id` | required |
| `label` | |
| `affiliation` | A 2525 standard identity for the source's sensors: `pending`, `unknown`, `assumed_friend`, `friend`, `neutral`, `suspect`, `hostile` |
| `assumed_bearing_sigma_deg`, `assumed_elevation_sigma_deg`, `assumed_position_sigma_m` | Declared uncertainty, applied only where the source reports none |
| `assumption_note` | **Required** with any `assumed_*`: why this number |

**`truth_relations`** — which emitters really operate together, for scoring grouping
(Phase 7). Scorer only. `{kind, members: [{emitter_id, role?}], note?}`; `kind` is
`controller/controlled`, `superior/subordinate`, `peer` or `co-sited`; at least two members,
each an existing emitter.

## 9. Running a scenario

```bash
# Locally, no containers: simulate to a file, or send to a running Vigilans
uv run vigilans-hub write my.scenario.yaml --out out/my.observations.jsonl --truth out/my.truth.jsonl
uv run vigilans run --ingest 127.0.0.1:8092            # terminal 1
uv run vigilans-hub sim my.scenario.yaml --rate 4       # terminal 2

# In containers, with Videns
HUB_SCENARIOS=/path/to/exports HUB_SCENARIO="my scenario" HUB_RATE=4 \
  docker compose --profile videns up --build
```

`HUB_SCENARIO` is a file name in `HUB_SCENARIOS` (with or without `.scenario.yaml`) or a
built-in: `mixed`, `cp_departure`.

## 10. Minimal example

```yaml
schema: scenario.v2
name: minimal
duration_s: 300
origin: { lat: 50.0, lon: -105.0 }
sources:
  - { id: DF, kind: bearing }
sensors:
  - { id: A, source_id: DF, pos_m: [-8000, -6000], bearing_sigma_deg: 2, freq_range_hz: [100e6, 1e9] }
  - { id: B, source_id: DF, pos_m: [9000, -5000], bearing_sigma_deg: 2, freq_range_hz: [100e6, 1e9] }
emitters:
  - { id: E, path_m: [[1000, 2000]], freq_hz: 401e6, bandwidth_hz: 12500,
      activity: { type: periodic, period_s: 10, on_s: 1 } }
```

Worked examples: [`hub/scenarios/mixed.scenario.yaml`](../hub/scenarios/mixed.scenario.yaml)
(three sources, all three uncertainty stories, a deliberate adapter fault) and
[`hub/scenarios/cp_departure.scenario.yaml`](../hub/scenarios/cp_departure.scenario.yaml)
(holds, a biased and late position system, `truth_relations`).

## 11. Validation errors a generator should expect

The validator reports every problem with a JSON pointer. Besides the schema itself it checks:
duplicate ids; `activity.net_id` naming no net; a multi-point path with no speed; `on_s` over
`period_s`; `turnaround_s` or `freq_range_hz` out of order; `dwell_s` over `revisit_s`; a
bearing-source sensor without `bearing_sigma_deg`; elevation on a position receiver; an
active window whose start is not before its stop; a declaration for a source no sensor uses;
an assumption without a note; a truth relation naming an unknown emitter.
