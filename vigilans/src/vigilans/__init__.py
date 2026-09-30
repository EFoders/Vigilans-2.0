"""Vigilans: RF observations from many sources, fused into entities and groups.

Receive-only, externals only, synthetic in the repository. See VIGILANS_SPEC.md.
"""

ENGINE_NAME = "vigilans"
__version__ = "2.0.0.dev0"

#: Which stages exist yet. Printed in the run header and published in a notice, so a
#: picture with no entities says why rather than looking broken (rule 9).
STAGES_BUILT: tuple[str, ...] = (
    "ingest",
    "normalise",
    "validate",
    "geolocation",
    "tracking",
    "entity resolution",
    "classification",
    "operator corrections",
    "offline maps",
    "fingerprinting and re-identification",
    "picture",
)
STAGES_PENDING: tuple[str, ...] = (
    "grouping",
    "CoT dissemination",
)
