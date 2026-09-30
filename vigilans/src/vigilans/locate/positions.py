"""Direct use of position observations: the source already located the emitter.

Nothing is recomputed and nothing is improved. The fix is the source's position with the
source's uncertainty, in the form the source gave it (ellipse at its stated confidence, or
covariance), and the source's basis. A source that reported no uncertainty yields a fix with
no region; an operator's declared assumption, applied at ingest, is carried as ``assumed``.
"""

from __future__ import annotations

from vigilans.locate.fix import Fix, Height, Region
from vigilans.observation import PositionObservation


def fix_from_position(observation: PositionObservation, *, fix_id: str) -> Fix:
    u = observation.position_uncertainty
    assumptions = (u.assumption,) if u.assumption is not None else ()
    region = Region(
        basis=u.basis,
        cov_en_m2=u.cov_en_m2,
        ellipse=u.ellipse,
        assumptions=assumptions,
        mixture={k: int(k == u.basis) for k in ("measured", "assumed", "unreported")},
    )
    height = None
    if observation.alt_m is not None:
        a = observation.alt_uncertainty
        sigma = a.sigma if a is not None else None
        height = Height(
            observation.alt_m,
            sigma,
            a.basis if a is not None else "unreported",
            0,
            f"reported by {observation.source_id}",
        )
    return Fix(
        fix_id=fix_id,
        method="reported_position",
        t=observation.t,
        lat=observation.lat,
        lon=observation.lon,
        region=region,
        freq_hz=observation.freq_hz,
        bandwidth_hz=observation.bandwidth_hz,
        position=observation,
        height=height,
        notes=(f"Position as reported by {observation.source_id}; Vigilans has not recomputed it.",),
    )
