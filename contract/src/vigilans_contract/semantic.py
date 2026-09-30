"""Checks the schemas cannot express.

Each one guards an honesty rule: an ellipse that is not an ellipse, a symbol that
contradicts its stated affiliation, a group more confident than its members' identities
allow. The picture checks mirror Videns' ``src/contract/semantic.ts`` so that the two
languages reject the same messages for the same reasons; the shared invalid fixtures are
what keeps them honest.

Every function here assumes a schema-valid record.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from vigilans_contract.identity import IDENTITY_LABEL, sidc_identity
from vigilans_contract.validate import Issue

#: Covariance determinant tolerance, square metres squared: rounding, not a real negative.
PSD_TOLERANCE = 1e-6

#: Targeting vocabulary. Vigilans rule 4: never in an assessment, a label, or a UI string.
FORBIDDEN_WORDS = re.compile(r"\b(target\w*|strike\w*|engag\w*|kill\w*)\b", re.IGNORECASE)

#: A library label is a noun phrase; the engine adds the hedge (library.v1).
HEDGE_WORDS = re.compile(
    r"^\s*(likely|possible|possibly|probable|probably|maybe|certain\w*)\b", re.IGNORECASE
)


def forbidden_words(text: str) -> list[str]:
    """Every targeting word in ``text``, as written."""
    return [match.group(0) for match in FORBIDDEN_WORDS.finditer(text)]


# --- shared ------------------------------------------------------------------------------


def uncertainty_issues(uncertainty: Mapping[str, Any], path: str) -> list[Issue]:
    issues: list[Issue] = []
    ellipse = uncertainty.get("ellipse")
    if ellipse is not None and ellipse["semi_minor_m"] > ellipse["semi_major_m"]:
        issues.append(Issue(f"{path}/ellipse", "semi_minor_m is larger than semi_major_m"))
    cov = uncertainty.get("cov_en_m2")
    if cov is not None and cov["ee"] * cov["nn"] - cov["en"] ** 2 < -PSD_TOLERANCE:
        issues.append(Issue(f"{path}/cov_en_m2", "covariance is not positive semi-definite"))
    return issues


def _unique(ids: Iterable[str], path: str) -> list[Issue]:
    seen: set[str] = set()
    issues: list[Issue] = []
    for identifier in ids:
        if identifier in seen:
            issues.append(Issue(path, f'duplicate id "{identifier}"'))
        seen.add(identifier)
    return issues


# --- observation.v1 ----------------------------------------------------------------------


def observation_issues(record: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    kind = record["kind"]
    if kind == "position":
        issues.extend(uncertainty_issues(record["position_uncertainty"], "/position_uncertainty"))
    if kind in ("coverage", "occupancy") and record["band"]["min_hz"] > record["band"]["max_hz"]:
        issues.append(Issue("/band", "min_hz is greater than max_hz"))
    if kind == "coverage" and datetime.fromisoformat(record["t_end"]) < datetime.fromisoformat(
        record["t_start"]
    ):
        issues.append(Issue("/t_end", "t_end is before t_start"))
    if kind == "coverage" and "dwell_s" in record and record["dwell_s"] > record["revisit_s"]:
        issues.append(Issue("/dwell_s", "dwell_s is longer than revisit_s"))
    return issues


# --- picture.v0 --------------------------------------------------------------------------


def _symbol_issues(item: Mapping[str, Any], path: str) -> list[Issue]:
    sidc = item["symbol"]["sidc"]
    identity = item["affiliation"]["identity"]
    coded = sidc_identity(sidc)
    if coded is None:
        return [Issue(f"{path}/symbol/sidc", f"symbol code {sidc} does not carry a standard identity")]
    if coded != identity:
        return [
            Issue(
                f"{path}/symbol/sidc",
                f"symbol code {sidc} says {IDENTITY_LABEL[coded]} "
                f"but affiliation says {IDENTITY_LABEL[identity]}",
            )
        ]
    return []


def _entity_issues(entity: Mapping[str, Any], path: str) -> list[Issue]:
    issues = _symbol_issues(entity, path)
    if "position_uncertainty" in entity:
        issues.extend(uncertainty_issues(entity["position_uncertainty"], f"{path}/position_uncertainty"))
    # Compare instants, not strings: "…00Z" and "…00.000Z" are the same moment.
    first = datetime.fromisoformat(entity["first_seen"])
    last = datetime.fromisoformat(entity["last_seen"])
    if last < first:
        issues.append(Issue(f"{path}/last_seen", "last_seen is before first_seen"))
    for index, evidence in enumerate(entity.get("fix_evidence", {}).get("positions", [])):
        issues.extend(
            uncertainty_issues(evidence["uncertainty"], f"{path}/fix_evidence/positions/{index}/uncertainty")
        )
    return issues


def _group_issues(group: Mapping[str, Any], path: str) -> list[Issue]:
    issues: list[Issue] = []
    if group["confidence"] > group["member_identity_bound"]:
        issues.append(
            Issue(
                f"{path}/confidence",
                f"confidence {group['confidence']} exceeds the member identity bound "
                f"{group['member_identity_bound']}",
            )
        )
    issues.extend(_unique((m["entity_id"] for m in group["members"]), f"{path}/members"))
    return issues


def picture_issues(message: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    if message["type"] == "snapshot":
        containers: Mapping[str, Any] = message
        prefix = ""
    elif message["type"] == "delta":
        containers = message.get("upsert", {})
        prefix = "/upsert"
    else:
        return issues

    entities = containers.get("entities", [])
    groups = containers.get("groups", [])
    sensors = containers.get("sensors", [])
    for index, entity in enumerate(entities):
        issues.extend(_entity_issues(entity, f"{prefix}/entities/{index}"))
    for index, group in enumerate(groups):
        issues.extend(_group_issues(group, f"{prefix}/groups/{index}"))
    for index, sensor in enumerate(sensors):
        issues.extend(_symbol_issues(sensor, f"{prefix}/sensors/{index}"))
    issues.extend(_unique((e["entity_id"] for e in entities), f"{prefix}/entities"))
    issues.extend(_unique((g["group_id"] for g in groups), f"{prefix}/groups"))
    issues.extend(_unique((s["sensor_id"] for s in sensors), f"{prefix}/sensors"))
    return issues


# --- library.v1 --------------------------------------------------------------------------


def _texts(value: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _texts(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _texts(item, f"{path}/{index}")


def scenario_issues(scenario: Mapping[str, Any]) -> list[Issue]:
    """Cross-references and orderings a scenario schema cannot check."""
    issues: list[Issue] = []
    sensors, emitters = scenario["sensors"], scenario["emitters"]
    nets, sources = scenario.get("nets", []), scenario.get("sources", [])
    for name, items in (("sensors", sensors), ("emitters", emitters), ("nets", nets), ("sources", sources)):
        issues.extend(_unique((i["id"] for i in items), f"/{name}"))
    net_ids = {n["id"] for n in nets}
    source_kinds = {s["id"]: s["kind"] for s in sources}
    used_sources = {s.get("source_id", "SIM") for s in sensors}
    emitter_ids = {e["id"] for e in emitters}

    for i, s in enumerate(sensors):
        path = f"/sensors/{i}"
        low, high = s.get("freq_range_hz", (20e6, 3e9))
        if not low < high:
            issues.append(Issue(f"{path}/freq_range_hz", "low must be below high"))
        scan = s.get("scan")
        if scan and scan["dwell_s"] > scan["revisit_s"]:
            issues.append(Issue(f"{path}/scan", "dwell_s is longer than revisit_s"))
        if (
            source_kinds.get(s.get("source_id", "SIM"), "bearing") == "bearing"
            and "bearing_sigma_deg" not in s
        ):
            issues.append(Issue(path, "a sensor in a bearing source needs bearing_sigma_deg"))
        if source_kinds.get(s.get("source_id", "SIM"), "bearing") == "position" and s.get(
            "elevation_sigma_deg"
        ):
            issues.append(
                Issue(
                    f"{path}/elevation_sigma_deg",
                    "a position source's receiver reports positions, not angles",
                )
            )
    for i, n in enumerate(nets):
        turnaround = n.get("turnaround_s", (1.0, 3.0))
        if turnaround[0] > turnaround[1]:
            issues.append(Issue(f"/nets/{i}/turnaround_s", "low must not exceed high"))
    for i, e in enumerate(emitters):
        path = f"/emitters/{i}"
        if len(e["path_m"]) > 1 and e.get("speed_mps", 0) <= 0:
            issues.append(
                Issue(f"{path}/speed_mps", f"a path of {len(e['path_m'])} waypoints needs speed_mps > 0")
            )
        activity = e.get("activity", {"type": "continuous"})
        if activity["type"] == "net" and activity.get("net_id") not in net_ids:
            issues.append(Issue(f"{path}/activity/net_id", f"no net {activity.get('net_id')!r}"))
        if activity["type"] == "periodic" and activity["on_s"] > activity["period_s"]:
            issues.append(Issue(f"{path}/activity/on_s", "on_s exceeds period_s"))
        for j, (start, stop) in enumerate(e.get("active_windows_s", [])):
            if not start < stop:
                issues.append(Issue(f"{path}/active_windows_s/{j}", "start must be before stop"))
    for i, d in enumerate(scenario.get("operator", {}).get("declarations", [])):
        path = f"/operator/declarations/{i}"
        if d["source_id"] not in used_sources:
            issues.append(Issue(f"{path}/source_id", f"no sensor reports as {d['source_id']!r}"))
        if any(k.startswith("assumed_") for k in d) and "assumption_note" not in d:
            issues.append(Issue(path, "an assumed uncertainty needs an assumption_note saying why"))
    for i, r in enumerate(scenario.get("truth_relations", [])):
        for j, m in enumerate(r["members"]):
            if m["emitter_id"] not in emitter_ids:
                issues.append(Issue(f"/truth_relations/{i}/members/{j}", f"no emitter {m['emitter_id']!r}"))
    return issues


_SCALE_FEATURES = {"freq_hz", "bandwidth_hz", "duration_s", "period_s"}


def _range_issues(feature: Mapping[str, Any], path: str, name: str) -> list[Issue]:
    issues = []
    if feature["min"] > feature["max"]:
        issues.append(Issue(path, f"min {feature['min']} is greater than max {feature['max']}"))
    if name in _SCALE_FEATURES and feature["min"] <= 0:
        issues.append(Issue(path, f"{name} is compared on a log scale: min must be positive"))
    return issues


def library_issues(library: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    classes = library["classes"]
    ids = [c["class_id"] for c in classes]
    issues.extend(_unique(ids, "/classes"))
    for name, feature in library["background"].items():
        issues.extend(_range_issues(feature, f"/background/{name}", name))
    parents = {c["class_id"]: c.get("parent") for c in classes}
    for index, cls in enumerate(classes):
        path = f"/classes/{index}"
        if HEDGE_WORDS.match(cls["label"]):
            issues.append(Issue(f"{path}/label", "a label is a plain noun phrase; the engine adds the hedge"))
        parent = cls.get("parent")
        if parent is not None and parent not in parents:
            issues.append(Issue(f"{path}/parent", f"no class {parent!r}"))
        seen, node = {cls["class_id"]}, parent
        while node is not None and node in parents:
            if node in seen:
                issues.append(Issue(f"{path}/parent", "the class tree has a cycle"))
                break
            seen.add(node)
            node = parents[node]
        for name, feature in cls.get("features", {}).items():
            if isinstance(feature, Mapping) and "min" in feature:
                issues.extend(_range_issues(feature, f"{path}/features/{name}", name))
                if name not in library["background"]:
                    issues.append(
                        Issue(f"{path}/features/{name}", f"the background has no {name} to weigh it against")
                    )
    for path, text in _texts(library, ""):
        for word in forbidden_words(text):
            issues.append(Issue(path, f'"{word}" is targeting language (Vigilans rule 4)'))
    return issues
