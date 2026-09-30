# Vigilans 2.0

Vigilans turns RF observations from several independent sources into a small number of
**entities** — who is transmitting, probably what they are, and which of them are working
together — and publishes that picture to [Videns](../Videns) (the viewer) and to TAK.

Receive-only. Externals only, never content. Synthetic and unclassified in the repository.
The specification is [`VIGILANS_SPEC.md`](../specs/VIGILANS_SPEC.md); the viewer's is
[`VIDENS_SPEC.md`](../specs/VIDENS_SPEC.md).

```
observations (observation.v1)
   ↓  ingest + normalise + validate            ← Phase 1 ✔
   ↓  cross-source entity resolution           ← Phase 4 ✔
   ↓  association + geolocation                ← Phase 2 ✔
   ↓  tracking · fingerprinting                ← Phase 3a ✔ (3b: maps), 6
   ↓  classification · grouping                ← Phase 5 ✔, 7
   ↓  dissemination (CoT)                      Phase 8
picture.v0 ──SSE──▶ Videns                     ← Phase 1 ✔
```

## Status

| Phase | Deliverable | Gate | |
|---|---|---|---|
| 0 | Repository, contract package, synthetic library schema, CI | Contract validates both observation flavours | ✔ |
| 1 | Ingest, normalisation, validation, provenance — plus the picture sink | A malformed or unlabelled observation fails loudly | ✔ |
| 2 | Geolocation from bearings; direct use of positions | Both paths give a fix with honest uncertainty | ✔ |
| 3a | Tracking (no map data) | Identity survives motion and silence | ✔ |
| 4 | Cross-source entity resolution | Two sources, one emitter, one entity | ✔ |
| 3b | Map data: roads, water, terrain ([ADR-0014](docs/adr/0014-offline-maps.md)) | Road-constrained motion; height above ground | ✔ (synthetic maps; bring your own files) |
| 5 | Classification ([design](docs/classifier-design.md), [as built](docs/adr/0013-classifier-as-built.md)) | Private library loads; nothing real in the repo | ✔ |
| 6 | Fingerprinting and re-identification ([ADR-0016](docs/adr/0016-reidentification.md)) | Re-identification confidence is carried downstream | ✔ |
| 7 | Grouping | | next |

The picture now holds **entities**: tracked across sources, through silence, with motion-model
probabilities. Emitters on one channel too close to separate are one entity until they move
apart, which is published as a split with lineage; an entity that stays while others leave
keeps a record of the departures ([ADR-0009](docs/adr/0009-tracking-and-resolution-cluster-doctrine.md)).
Try it: `VIGILANS_SCENARIO=cp_departure VIGILANS_RATE=4 docker compose --profile videns up --build`. Every fix's
uncertainty keeps its basis, and the regions are calibrated against simulated truth
([ADR-0007](docs/adr/0007-fix-uncertainty-basis-and-calibration.md)). Classification is
designed ahead of the phases that feed it: [docs/classifier-design.md](docs/classifier-design.md).

## Quickstart

Docker only:

```bash
docker compose --profile videns up --build
```

Three containers: a **sensor hub** simulating the `mixed` scenario and sending
`observation.v1` over HTTP, **Vigilans** taking it on :8092, and **Videns** on
http://127.0.0.1:8081. The hub is swappable for real equipment
([ADR-0010](docs/adr/0010-inputs-outside-the-engine.md)). Another scenario, faster:
`HUB_SCENARIO=cp_departure HUB_RATE=4 docker compose --profile videns up --build`.
Scenarios are `scenario.v2` — the spec for generators is [docs/scenario-spec.md](docs/scenario-spec.md).

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run vigilans run --scenario mixed                      # live; picture on 127.0.0.1:8091
uv run vigilans run --scenario mixed --fast --record out/mixed.picture.jsonl
uv run vigilans run --config vigilans/config/assumed.toml # an operator-declared uncertainty
uv run vigilans record --scenario mixed --out out/mixed.observations.jsonl
uv run vigilans validate out/mixed.observations.jsonl     # exits 1: 4 deliberate adapter mistakes
```

Every run prints what it is about to do — inputs, sources, library, clock, where the
picture goes, and which flag, variable or file each setting came from — and a summary of
what was accepted, rejected and why.

## The `mixed` scenario

Three sources over four emitters, laid out ~40 km wide around the neutral origin:

- **DF-NET** — bearings from three sensors, with measured uncertainty.
- **LEGACY-DF** — bearings that report **no** uncertainty. Every 40th record carries a
  flat, unlabelled `bearing_sigma_deg` — a realistic adapter mistake — and is rejected.
- **GEO-SYS** — emitter positions with 95 % ellipses.

## Layout

```
contract/        vigilans-contract: observation.v1, picture.v0, library.v1, scenario.v2 + validator + fixtures
vigilans/        the engine (Python): src/vigilans, tests, libraries, config, Dockerfile
hub/             vigilans-hub: the simulated sensor hub, and scenario.v2 worlds
docs/adr/        decisions
compose.yaml     vigilans; videns (profile); test (profile)
```

## License

Apache-2.0.
