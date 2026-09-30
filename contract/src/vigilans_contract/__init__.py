"""The Vigilans contracts: JSON Schemas and a thin validator for each.

- ``observation.v1`` -- what sources and adapters send Vigilans.
- ``picture.v0``     -- what Vigilans publishes and Videns shows. Unstable.
- ``library.v1``     -- the format of a classification library. Unstable.
- ``scenario.v2``    -- a synthetic world for the simulated sensor hub.

Versioned and released independently of Vigilans, Videns and Audiens (VIGILANS_SPEC.md
section 5.1): this package is the only coupling between them.
"""

from vigilans_contract.identity import (
    IDENTITIES,
    IDENTITY_SIDC_CHAR,
    StandardIdentity,
    sidc_identity,
    sidc_with_identity,
)
from vigilans_contract.semantic import FORBIDDEN_WORDS, forbidden_words
from vigilans_contract.validate import (
    CONTRACTS,
    LIBRARY_V1,
    OBSERVATION_KINDS,
    OBSERVATION_V1,
    PICTURE_TYPES,
    PICTURE_V0,
    SCENARIO_V2,
    SIGNAL_KINDS,
    ContractJSONError,
    Issue,
    Result,
    date_time_format_is_checked,
    load_schema,
    parse_json,
    schema_dir,
    validate_library,
    validate_observation,
    validate_picture,
    validate_scenario,
)

__version__ = "0.4.0"

__all__ = [
    "CONTRACTS",
    "FORBIDDEN_WORDS",
    "IDENTITIES",
    "IDENTITY_SIDC_CHAR",
    "LIBRARY_V1",
    "OBSERVATION_KINDS",
    "OBSERVATION_V1",
    "PICTURE_TYPES",
    "PICTURE_V0",
    "SCENARIO_V2",
    "SIGNAL_KINDS",
    "ContractJSONError",
    "Issue",
    "Result",
    "StandardIdentity",
    "__version__",
    "date_time_format_is_checked",
    "forbidden_words",
    "load_schema",
    "parse_json",
    "schema_dir",
    "sidc_identity",
    "sidc_with_identity",
    "validate_library",
    "validate_observation",
    "validate_picture",
    "validate_scenario",
]
