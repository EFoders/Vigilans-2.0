"""The ``vigilans`` command.

    vigilans run --scenario mixed                    live: serve the picture for Videns
    vigilans run --scenario mixed --rate 4           four picture seconds per wall second
    vigilans run --scenario mixed --fast --record out/mixed.picture.jsonl
    vigilans run --file out/mixed.observations.jsonl --fast
    vigilans run --config vigilans/config/assumed.toml
    vigilans validate FILE...                        check observation.v1 records; exit 1 if any fail
    vigilans record --scenario mixed --out FILE      write a scenario's observations as JSONL
    vigilans check-library PATH [--private]

A run prints what it is about to do before it does it (spec §12): inputs, sources,
library, clock, where the picture goes, and where each of those settings came from.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from vigilans import STAGES_BUILT, STAGES_PENDING, __version__
from vigilans.app import Run, RunSummary, make_run_id
from vigilans.config import ConfigError, RunConfig, resolve
from vigilans.env import load_env_file
from vigilans.ingest import Ingestor
from vigilans.library import Library, LibraryError, load_library
from vigilans.picture.sensors import AffiliationError
from vigilans.sources.base import Input
from vigilans.sources.file import FileInput
from vigilans.sources.network import NetworkInput
from vigilans.sources.sim import SimInput

#: Drain an input completely.
END_OF_TIME = datetime.max.replace(tzinfo=UTC)


def _err(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _say(text: str) -> None:
    print(text, flush=True)


def _build_inputs(config: RunConfig) -> list[Input]:
    inputs: list[Input] = []
    for spec in config.inputs:
        if spec.kind == "network":
            assert spec.listen is not None
            inputs.append(NetworkInput(spec.listen))
        elif spec.kind == "sim":
            assert spec.scenario is not None
            inputs.append(SimInput(spec.scenario, config.seed))
        else:
            assert spec.path is not None
            if not spec.path.is_file():
                raise ConfigError(f"input file {spec.path} does not exist")
            inputs.append(FileInput(spec.path))
    return inputs


def _load_library(config: RunConfig) -> tuple[Library | None, str]:
    if config.library is None:
        return None, "none configured -- every entity would be unclassified"
    try:
        library = load_library(config.library, private=config.library_private)
    except LibraryError as error:
        # Spec §9.1: an unavailable library degrades to unclassified, loudly, never to a guess.
        return None, f"UNAVAILABLE: {error}"
    return library, library.describe()


def _header(config: RunConfig, run: Run, library_text: str) -> list[str]:
    o = config.origins

    def src(key: str) -> str:
        return f"  [{o.get(key, 'default')}]"

    lines = [f"vigilans {__version__} -- run {run.header.run_id} (contracts picture.v0, observation.v1)"]
    if config.config_path:
        lines.append(f"  config:   {config.config_path}{src('config')}")
    for index, item in enumerate(run.inputs):
        lines.append(
            f"  {'inputs:' if index == 0 else '':9} {item.description}{src('inputs') if index == 0 else ''}"
        )
    sources = []
    for source_id, label in sorted(run.header.source_labels.items()):
        declared = run.board._declarations.get(source_id)
        extra = f", {declared.identity} declared in {declared.declared_in}" if declared else ""
        settings = config.sources.get(source_id)
        if settings and any(
            v is not None
            for v in (
                settings.assumed_bearing_sigma_deg,
                settings.assumed_elevation_sigma_deg,
                settings.assumed_position_sigma_m,
            )
        ):
            extra += ", has operator-assumed uncertainty"
        detail = f"{label or ''}{extra}".removeprefix(", ")
        sources.append(f"{source_id} ({detail})" if detail else source_id)
    lines.append(f"  sources:  {'; '.join(sources) if sources else '(learned as observations arrive)'}")
    lines.append(f"  library:  {library_text}{src('library')}")
    for index, (kind, status) in enumerate(run.maps.status().items()):
        lines.append(f"  {'maps:' if index == 0 else '':9} {kind} {status}{src('map_' + kind)}")
    lines.append(f"  corrections: {run.corrections.describe()}{src('corrections')}")
    rate = "as fast as possible (offline)" if config.rate is None else f"{config.rate:g}x"
    lines.append(
        f"  clock:    sim from {run.start.isoformat()}, {rate}, step {config.step_s:g} s{src('rate')}"
    )
    lines.append(f"  seed:     {config.seed}{src('seed')}")
    if config.listen:
        host, port = config.listen
        lines.append(f"  picture:  http://{host}:{port}/picture/v0/stream  (SSE){src('listen')}")
    else:
        lines.append(f"  picture:  not served{src('listen')}")
    lines.append(f"  record:   {config.record or 'none'}{src('record')}")
    lines.append("  CoT:      not built or published (Phase 8); nothing reaches TAK from this run")
    lines.append(f"  stages:   built {', '.join(STAGES_BUILT)}")
    lines.append(f"            pending {', '.join(STAGES_PENDING)}")
    lines.append(
        f"  entities: tracked across sources; co-located emitters on one channel are one entity "
        f"until they separate (a split){src('tracking')}"
    )
    return lines


def _summary(summary: RunSummary) -> list[str]:
    ended = "interrupted" if summary.interrupted else "finished"
    lines = [
        f"run {summary.run_id} {ended} at picture time {summary.ended.isoformat()}",
        f"  observations: {summary.raw} received, {summary.accepted} accepted, {summary.rejected} rejected",
    ]
    for source_id, s in summary.by_source.items():
        notes = []
        if s["unreported"]:
            notes.append(f"{s['unreported']} with unreported uncertainty")
        if s["assumed"]:
            notes.append(f"{s['assumed']} with operator-assumed uncertainty")
        if s["duplicates"]:
            notes.append(f"{s['duplicates']} duplicates")
        tail = f"  ({'; '.join(notes)})" if notes else ""
        lines.append(
            f"    {source_id:<14} {s['bearing']:>6} bearing {s['position']:>6} position "
            f"{s['rejected']:>5} rejected{tail}"
        )
    if summary.reasons:
        lines.append("  rejection reasons:")
        lines.extend(f"    {count:>5} x {reason}" for reason, count in summary.reasons)
    fixes = sum(summary.fixes.values())
    bases = ", ".join(f"{n} {b}" for b, n in sorted(summary.fix_bases.items())) or "none"
    lines.append(
        f"  geolocation: {fixes} fixes ({summary.fixes.get('bearings', 0)} from bearings, "
        f"{summary.fixes.get('reported_position', 0)} as reported); uncertainty: {bases}"
    )
    if summary.unlocated:
        lines.append(
            "  not located: "
            + ", ".join(f"{n} {r.replace('_', ' ')}" for r, n in sorted(summary.unlocated.items()))
        )
    lines.append(
        f"  picture: {summary.messages} messages; {summary.sensors} sensors; "
        f"{summary.entities} entities at the end"
    )
    states = ", ".join(f"{n} {s}" for s, n in sorted(summary.entity_states.items())) or "none"
    t = summary.tracking
    lines.append(
        f"  entities: {states}; {t.get('split', 0)} split(s), {t.get('merged', 0)} merge(s), "
        f"{t.get('unmerged', 0)} unmerge(s), {t.get('relocated', 0)} relocation(s), "
        f"{t.get('retired', 0)} retired"
    )
    for label, n in sorted(summary.departures.items(), key=lambda kv: -kv[1]):
        lines.append(f"    {label}: {n} entit{'y' if n == 1 else 'ies'} departed from it while it stayed")
    for est in summary.nuisance:
        if est["freq_n"] or est["clock_n"]:
            lines.append(
                f"  {est['source_id']} vs {est['reference']}: frequency {est['freq_bias_hz']:+.0f} Hz "
                f"(n={est['freq_n']}), clock {est['clock_offset_s']:+.3f} s (n={est['clock_n']})"
            )
    if summary.recording:
        lines.append(f"  recorded to {summary.recording}")
    return lines


def cmd_run(args: argparse.Namespace) -> int:
    flags = {
        k: getattr(args, k)
        for k in (
            "config",
            "scenario",
            "files",
            "ingest",
            "seed",
            "rate",
            "listen",
            "record",
            "library",
            "corrections",
            "map_roads",
            "map_water",
            "map_terrain",
            "run_id",
            "duration_s",
            "fast",
            "exit_when_done",
            "strict",
        )
    }
    try:
        config = resolve(flags, os.environ)
        inputs = _build_inputs(config)
        library, library_text = _load_library(config)
    except (ConfigError, AffiliationError, ValueError) as error:
        _err(f"vigilans: {error}")
        return 2
    network = [i for i in inputs if isinstance(i, NetworkInput)]
    if network:
        return asyncio.run(_run_networked(config, network, library, library_text))
    try:
        name = next((i.scenario for i in inputs if i.scenario), None)
        run = Run(config, inputs, library, run_id=config.run_id or make_run_id(name), out=_say)
    except (ConfigError, AffiliationError, ValueError) as error:
        _err(f"vigilans: {error}")
        return 2
    for line in _header(config, run, library_text):
        _say(line)
    if library is None and config.library is not None:
        _err(f"vigilans: WARNING: library {library_text}")
    summary = asyncio.run(run.execute())
    for line in _summary(summary):
        _say(line)
    if config.strict and summary.rejected:
        _err(f"vigilans: {summary.rejected} observation(s) rejected (--strict)")
        return 1
    return 0


async def _run_networked(
    config: RunConfig, network: list[NetworkInput], library: Library | None, library_text: str
) -> int:
    """Listen for observations, wait for the first, then run with picture time following the stream."""
    from vigilans.picture.server import serve_app

    runners = []
    for item in network:
        host, port = item.listen
        runners.append(await serve_app(item.app(), host, port))
        _say(f"vigilans {__version__}: listening for observation.v1 on http://{host}:{port}/observations/v1")
    _say(
        "  waiting for the first observation: the picture starts when a source does "
        "(origin and time come from it)"
    )
    try:
        while not any(i.first_time is not None for i in network):
            await asyncio.sleep(0.2)
        run = Run(config, network, library, run_id=config.run_id or make_run_id("network"), out=_say)
        for line in _header(config, run, library_text):
            _say(line)
        summary = await run.execute()
        for line in _summary(summary):
            _say(line)
        return 0
    except (ConfigError, AffiliationError, ValueError) as error:
        _err(f"vigilans: {error}")
        return 2
    finally:
        for runner in runners:
            await runner.cleanup()


def cmd_validate(args: argparse.Namespace) -> int:
    ingestor = Ingestor()
    failures = 0
    total = 0
    for name in args.paths:
        path = Path(name)
        if not path.is_file():
            _err(f"{path}: no such file")
            failures += 1
            continue
        source = FileInput(path)
        total += source.lines
        batch = ingestor.ingest(source.drain(END_OF_TIME))
        for rejection in batch.rejected:
            failures += 1
            _say(f"FAIL {rejection.origin}: {'; '.join(rejection.issues)}")
    accepted = ingestor.accepted
    _say(f"{accepted} of {total} observation(s) valid; {failures} failed")
    return 1 if failures else 0


def cmd_record(args: argparse.Namespace) -> int:
    try:
        source = SimInput(args.scenario, args.seed)
    except ValueError as error:
        _err(f"vigilans: {error}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in source.records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    _say(f"wrote {len(source.records)} observations from {source.description} to {out}")
    return 0


def cmd_adapt(args: argparse.Namespace) -> int:
    from vigilans.adapters.detection_v2 import adapt_file

    report = adapt_file(Path(args.path), Path(args.out), source_id=args.source_id)
    _say(f"read {report.read}, wrote {report.written} observation.v1 records to {args.out}")
    for reason, count in report.dropped.items():
        _say(f"  dropped {count}: {reason}")
    for problem in report.invalid[:20]:
        _err(f"  INVALID (not written): {problem}")
    return 1 if report.invalid else 0


def cmd_check_library(args: argparse.Namespace) -> int:
    try:
        library = load_library(Path(args.path), private=args.private)
    except LibraryError as error:
        _err(f"vigilans: {error}")
        return 1
    _say(f"ok: {library.describe()}")
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="vigilans", description="Vigilans RF observation fusion (Phase 1).")
    top.add_argument("--version", action="version", version=f"vigilans {__version__}")
    top.add_argument("--env-file", default=".env", help="environment file to load if present (default .env)")
    top.add_argument("-v", "--verbose", action="store_true")
    commands = top.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run inputs through the engine and publish the picture")
    run.add_argument("--config", help="TOML configuration file")
    run.add_argument("--scenario", help="a built-in scenario name, or a scenario YAML path")
    run.add_argument("--file", dest="files", action="append", default=[], help="an .observations.jsonl input")
    run.add_argument("--ingest", help="HOST:PORT to receive observations over HTTP (a sensor hub or adapter)")
    run.add_argument("--seed", type=int)
    run.add_argument("--rate", help="picture seconds per wall second, or 'max'")
    run.add_argument("--fast", action="store_true", help="offline: as fast as possible, no server")
    run.add_argument("--listen", help="HOST:PORT for the picture feed, or 'none' (default 127.0.0.1:8091)")
    run.add_argument("--record", help="write the picture to this .picture.jsonl")
    run.add_argument("--library", help="classification library path, or 'none'")
    run.add_argument("--corrections", help="operator corrections TOML, reread when it changes (ADR-0015)")
    run.add_argument("--map-roads", help="roads GeoJSON (ADR-0014)")
    run.add_argument("--map-water", help="water GeoJSON (ADR-0014)")
    run.add_argument("--map-terrain", help="terrain elevation ESRI ASCII grid (ADR-0014)")
    run.add_argument("--run-id")
    run.add_argument("--duration", dest="duration_s", type=float, help="stop after this many picture seconds")
    run.add_argument(
        "--exit-when-done", action="store_true", help="do not keep serving after the run completes"
    )
    run.add_argument("--strict", action="store_true", help="exit 1 if any observation was rejected")
    run.set_defaults(func=cmd_run)

    validate = commands.add_parser("validate", help="validate observation.v1 JSONL files")
    validate.add_argument("paths", nargs="+")
    validate.set_defaults(func=cmd_validate)

    record = commands.add_parser("record", help="write a scenario's observations as JSONL")
    record.add_argument("--scenario", required=True)
    record.add_argument("--seed", type=int, default=1)
    record.add_argument("--out", required=True)
    record.set_defaults(func=cmd_record)

    adapt = commands.add_parser("adapt", help="translate another system's records into observation.v1")
    adapt.add_argument(
        "format", choices=["detection-v2"], help="detection-v2: the Vigilans prototype's output"
    )
    adapt.add_argument("path")
    adapt.add_argument("--out", required=True)
    adapt.add_argument(
        "--source-id", default="PROTO-SIM", help="source_id for every record (default PROTO-SIM)"
    )
    adapt.set_defaults(func=cmd_adapt)

    library = commands.add_parser("check-library", help="validate a classification library")
    library.add_argument("path")
    library.add_argument("--private", action="store_true", help="treat as private (must be outside the repo)")
    library.set_defaults(func=cmd_check_library)
    return top


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    applied = load_env_file(Path(args.env_file))
    if applied:
        _say(f"loaded {', '.join(sorted(applied))} from {args.env_file}")
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
