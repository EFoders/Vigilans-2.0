"""MIL-STD-2525 standard identities and the 2525C letter-code character for each.

The same table as Videns' ``src/contract/identity.ts``. Nothing here chooses an identity;
it only says how one is written.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

StandardIdentity = Literal["pending", "unknown", "assumed_friend", "friend", "neutral", "suspect", "hostile"]

IDENTITIES: Final[tuple[StandardIdentity, ...]] = get_args(StandardIdentity)

#: The 2525C identity character, position 2 of a letter symbol code.
IDENTITY_SIDC_CHAR: Final[dict[StandardIdentity, str]] = {
    "pending": "P",
    "unknown": "U",
    "assumed_friend": "A",
    "friend": "F",
    "neutral": "N",
    "suspect": "S",
    "hostile": "H",
}


#: The standard's own names. Not synonyms: "Hostile", never anything stronger.
IDENTITY_LABEL: Final[dict[StandardIdentity, str]] = {
    "pending": "Pending",
    "unknown": "Unknown",
    "assumed_friend": "Assumed friend",
    "friend": "Friend",
    "neutral": "Neutral",
    "suspect": "Suspect",
    "hostile": "Hostile",
}


def sidc_identity(sidc: str) -> StandardIdentity | None:
    """The identity a 2525C letter code encodes, or None for exercise and joker codes."""
    char = sidc[1:2]
    for identity, code in IDENTITY_SIDC_CHAR.items():
        if code == char:
            return identity
    return None


def sidc_with_identity(template: str, identity: StandardIdentity) -> str:
    """Fill the identity position of a 2525C code or template (``S*GP…``)."""
    if len(template) != 15:
        raise ValueError(f"a 2525C letter code has 15 characters, got {len(template)}: {template!r}")
    return template[0] + IDENTITY_SIDC_CHAR[identity] + template[2:]
