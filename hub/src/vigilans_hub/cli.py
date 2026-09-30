"""The ``vigilans-hub`` command.

    vigilans-hub sim SCENARIO --to http://vigilans:8092 [--rate 4] [--truth out/x.truth.jsonl]
    vigilans-hub replay FILE.observations.jsonl --to http://vigilans:8092 [--rate 1]
    vigilans-hub write SCENARIO --out FILE.observations.jsonl [--truth FILE]

SCENARIO is a scenario.v2 (or v1) YAML path, or a name in hub/scenarios/. ``--wall-time``
restamps records to start now, as a real hub's would; by default they keep the scenario's
fixed epoch, so a run is reproducible.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vigilans_hub.sender import now_utc, restamp, send_paced
from vigilans_hub.world import ScenarioV2Error, load_world, simulate

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def _resolve(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    # HUB_SCENARIO_DIR (a mounted folder of editor exports) first, then the built-in ones.
    folders = [Path(d) for d in (os.environ.get("HUB_SCENARIO_DIR"),) if d] + [SCENARIO_DIR]
    for folder in folders:
        for suffix in (".scenario.yaml", ".yaml", ""):
            candidate = folder / f"{name_or_path}{suffix}"
            if candidate.is_file():
                return candidate
    known = ", ".join(sorted({p.name.split(".")[0] for f in folders for p in f.glob("*.yaml")})) or "(none)"
    raise SystemExit(f"vigilans-hub: no scenario {name_or_path!r}: not a file, and not one of {known}")


def _simulate(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = _resolve(args.scenario)
    try:
        world = load_world(path)
    except ScenarioV2Error as error:
        raise SystemExit(f"vigilans-hub: {error}") from error
    records, truth = simulate(world)
    print(
        f"vigilans-hub: {world.name}: {len(records)} records from {len(world.sensors)} sensors, "
        f"{len(world.emitters)} emitters, {world.duration_s:g} s",
        flush=True,
    )
    if args.truth:
        out = Path(args.truth)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"type": "scenario", "path": str(path), "name": world.name}) + "\n")
            for observation_id, emitter_id in truth.items():
                handle.write(
                    json.dumps({"type": "truth", "observation_id": observation_id, "emitter_id": emitter_id})
                    + "\n"
                )
        print(f"vigilans-hub: truth for {len(truth)} observations written to {out} (never sent)", flush=True)
    return records


def _send(records: list[dict[str, Any]], args: argparse.Namespace) -> int:
    if args.wall_time:
        records = restamp(records, now_utc())
    rate = None if args.rate == "max" else float(args.rate)
    print(
        f"vigilans-hub: sending {len(records)} records to {args.to} at "
        f"{'full speed' if rate is None else f'{rate:g}x'}",
        flush=True,
    )
    report = asyncio.run(send_paced(records, args.to.rstrip("/"), rate=rate))
    print(
        f"vigilans-hub: sent {report.sent} in {report.batches} batches; Vigilans rejected {report.invalid}",
        flush=True,
    )
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    return _send(_simulate(args), args)


def cmd_replay(args: argparse.Namespace) -> int:
    lines = Path(args.path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return _send([r for r in records if r.get("type") != "header"], args)


def cmd_write(args: argparse.Namespace) -> int:
    records = _simulate(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records), encoding="utf-8")
    print(f"vigilans-hub: wrote {len(records)} records to {out}", flush=True)
    return 0


def cmd_declarations(args: argparse.Namespace) -> int:
    """Write a scenario's operator declarations as Vigilans configuration ([sources.<id>] tables).

    Declarations are the operator's, so they belong to the engine's configuration, not the
    observation stream: run Vigilans with ``--config`` on the file this writes.
    """
    world = load_world(_resolve(args.scenario))
    lines = [f"# Operator declarations from scenario {world.name!r}, written by vigilans-hub.", ""]
    for d in world.declarations:
        lines.append(f'[sources."{d["source_id"]}"]')
        for key in ("label", "affiliation", "assumption_note"):
            if key in d:
                lines.append(f"{key} = {json.dumps(d[key])}")
        for key in ("assumed_bearing_sigma_deg", "assumed_elevation_sigma_deg", "assumed_position_sigma_m"):
            if key in d:
                lines.append(f"{key} = {d[key]}")
        lines.append("")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"vigilans-hub: {len(world.declarations)} declaration(s) written to {out}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="vigilans-hub", description="A simulated sensor hub for Vigilans.")
    commands = top.add_subparsers(dest="command", required=True)
    default_url = os.environ.get("HUB_VIGILANS_URL", "http://127.0.0.1:8092")
    default_rate = os.environ.get("HUB_RATE", "1")

    def sending(p: argparse.ArgumentParser) -> None:
        p.add_argument("--to", default=default_url, help=f"Vigilans ingest URL (default {default_url})")
        p.add_argument("--rate", default=default_rate, help="picture seconds per wall second, or 'max'")
        p.add_argument("--wall-time", action="store_true", help="restamp records to start now")

    sim = commands.add_parser("sim", help="simulate a scenario and send it")
    sim.add_argument("scenario", nargs="?", default=os.environ.get("HUB_SCENARIO", "mixed"))
    sim.add_argument("--truth", default=os.environ.get("HUB_TRUTH") or None)
    sending(sim)
    sim.set_defaults(func=cmd_sim)

    replay = commands.add_parser("replay", help="send a recorded .observations.jsonl")
    replay.add_argument("path")
    sending(replay)
    replay.set_defaults(func=cmd_replay)

    write = commands.add_parser("write", help="simulate a scenario to a file, sending nothing")
    write.add_argument("scenario")
    write.add_argument("--out", required=True)
    write.add_argument("--truth")
    write.set_defaults(func=cmd_write)

    declarations = commands.add_parser(
        "declarations", help="write the scenario's operator declarations as Vigilans TOML"
    )
    declarations.add_argument("scenario")
    declarations.add_argument("--out", required=True)
    declarations.set_defaults(func=cmd_declarations)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    args = parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
