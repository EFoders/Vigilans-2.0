# vigilans-hub

A stand-in for a sensor hub. It sends `observation.v1` to Vigilans over HTTP, exactly as a
real hub or adapter would, from one of:

- `vigilans-hub sim SCENARIO` — a `scenario.v2` world (docs/scenario-spec.md), simulated
  and paced in picture time;
- `vigilans-hub replay FILE` — a recorded `.observations.jsonl`.

It runs in its own container so it can be swapped for real equipment without touching
Vigilans (ADR-0010). Truth — which emitter made which observation — never leaves this side
except to a file you ask for with `--truth`, for scoring.
