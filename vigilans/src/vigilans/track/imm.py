"""An interacting-multiple-model (IMM) filter over east/north position and velocity.

Five motion models run side by side and are weighed by how well each explains the fixes:

- **stationary** — velocity held at zero, position allowed a slow random walk;
- **moving** — constant velocity, gentle white-acceleration noise (a vehicle on its way);
- **manoeuvring** — constant velocity with large acceleration noise (stopping, starting, jinking);
- **turning_left** / **turning_right** — a coordinated turn: constant speed, the velocity
  rotating at a fixed rate +ω (anticlockwise, seen from above) or -ω, with modest acceleration
  noise for speed changes and for a real turn rate that is not exactly ω (Phase 3b).

Their probabilities are outputs in their own right: the classifier's "assumed stationary" and
"moving" features are these numbers (classifier-design.md §3.1). :meth:`IMM.turn_rate` reads a
turn-rate estimate, with its uncertainty, off the same mixture. A road-constrained model needs
map data and is not here.

Why fixed-rate turn models rather than turn rate as a fifth state: every model keeps the same
4-D state, so IMM mixing (a weighted sum of states and covariances) stays exact; a turn-rate
state would exist in some models and not others. Two signed rates bracket the common case —
a turn is left or right — and the acceleration noise covers the mismatch in magnitude.

State ``[east_m, north_m, east_mps, north_mps]`` in the run's local frame. Measurements are
positions with a 2x2 covariance in the same frame. Pure: numpy only, no clock, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]

#: Index 0 is and stays "stationary": the tracker reads ``mu[0]`` and gates on mode 0.
MODES: tuple[str, ...] = ("stationary", "moving", "manoeuvring", "turning_left", "turning_right")
STATIONARY, MOVING, MANOEUVRING, TURNING_LEFT, TURNING_RIGHT = range(len(MODES))
_TURNS = (TURNING_LEFT, TURNING_RIGHT)
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_I4 = np.eye(4)


@dataclass(frozen=True, slots=True)
class MotionSettings:
    """Engine defaults, not class knowledge. Tuned on the synthetic scenarios; stated, not hidden."""

    #: Position random walk of the stationary model, m²/s: an emitter that "stays put" can
    #: still shuffle a few metres (a vehicle repositioning in a harbour).
    q_stationary_m2ps: float = 0.5
    sigma_a_moving: float = 0.5
    sigma_a_manoeuvring: float = 4.0
    #: Turn rate of the coordinated-turn models, rad/s (they turn at +ω and -ω). 0.1 rad/s is
    #: about 6°/s: a small uncrewed aircraft at 15-25 m/s on a 150-250 m radius, or a vehicle
    #: at 10 m/s round a 100 m bend; twice a crewed aircraft's "standard rate" (3°/s), well
    #: below a multirotor pivoting in place. Real rates up to about 1.7 ω are still picked up
    #: (the acceleration noise below absorbs the difference); much tighter turns read as
    #: "manoeuvring". 0.15 fitted the synthetic figure-eight (``uas_loop``) a little better
    #: but perturbed the command-post scenario's anchors; 0.1 did not. An engine default, not
    #: a measured property of any class.
    turn_rate_radps: float = 0.1
    #: Acceleration noise of the turn models, m/s². Covers speed changes and a real turn rate
    #: that is not exactly ω: at 25 m/s, a rate 0.05 rad/s off is 1.25 m/s² of lateral error.
    sigma_a_turning: float = 1.5
    #: Mean time spent in a mode before switching, seconds.
    tau_mode_s: float = 120.0
    #: Mean time spent turning before switching: turns are short (180° at ω takes ~31 s).
    tau_turn_s: float = 30.0
    #: How often a switch between moving models goes *into* a turn, relative to into any one
    #: other moving model. See :func:`_generator` for why stationary is unaffected.
    turn_entry_weight: float = 1.0
    #: Speed uncertainty at birth, m/s: nothing is known about velocity from one fix.
    v0_sigma_mps: float = 25.0
    initial_probabilities: tuple[float, ...] = (0.45, 0.45, 0.10, 0.0, 0.0)


def _generator(s: MotionSettings) -> FloatArray:
    """Continuous-time mode-switching rates, per second: q[i, j] for i != j, rows sum to 0.

    Built so that "stationary" against "on the move" switches exactly as it did with three
    models — the classifier's stationary feature must not drift because turn models exist:

    - stationary leaves (every ``tau_mode_s`` on average) for moving or manoeuvring, never
      straight into a turn: nothing turns from a standstill, it starts moving first;
    - moving and manoeuvring leave every ``tau_mode_s``: half the time to stationary (as
      before), half to another moving model, turns weighted by ``turn_entry_weight``;
    - a turn ends every ``tau_turn_s``, into moving, manoeuvring or the opposite turn.
    """
    n = len(MODES)
    weight = np.ones(n)
    weight[list(_TURNS)] = s.turn_entry_weight
    q: FloatArray = np.zeros((n, n))

    def spread(i: int, rate: float, into: list[int]) -> None:
        total = float(sum(weight[j] for j in into))
        for j in into:
            q[i, j] += rate * weight[j] / total

    moving_family = [MOVING, MANOEUVRING, *_TURNS]
    spread(STATIONARY, 1.0 / s.tau_mode_s, [MOVING, MANOEUVRING])
    for i in (MOVING, MANOEUVRING):
        q[i, STATIONARY] = 0.5 / s.tau_mode_s
        spread(i, 0.5 / s.tau_mode_s, [j for j in moving_family if j != i])
    for i in _TURNS:
        spread(i, 1.0 / s.tau_turn_s, [j for j in moving_family if j != i])
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def _expm(a: FloatArray) -> FloatArray:
    """Matrix exponential by scaling and squaring a Taylor series (small matrices only)."""
    norm = float(np.abs(a).sum(axis=1).max())
    squarings = max(0, math.ceil(math.log2(norm)) + 1) if norm > 0.5 else 0
    scaled = a / (2.0**squarings)
    out: FloatArray = np.eye(a.shape[0])
    term: FloatArray = np.eye(a.shape[0])
    for k in range(1, 16):
        term = term @ scaled / k
        out = out + term
    for _ in range(squarings):
        out = out @ out
    return out


def _transition(dt: float, s: MotionSettings) -> FloatArray:
    """pi[i, j] = P(mode j after ``dt`` | mode i now), from the switching rates."""
    if dt <= 0:
        return np.eye(len(MODES))
    matrix = np.maximum(_expm(_generator(s) * dt), 0.0)
    out: FloatArray = matrix / matrix.sum(axis=1, keepdims=True)
    return out


def _white_acceleration(sigma: float, dt: float) -> FloatArray:
    a, b, c = dt**3 / 3.0, dt**2 / 2.0, dt
    q: FloatArray = sigma**2 * np.array(
        [[a, 0, b, 0], [0, a, 0, b], [b, 0, c, 0], [0, b, 0, c]], dtype=np.float64
    )
    return q


def _coordinated_turn(omega: float, dt: float) -> FloatArray:
    """Transition of a turn at ``omega`` rad/s (positive anticlockwise, i.e. to the left)."""
    if abs(omega * dt) < 1e-9:
        return np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
    sin, cos = math.sin(omega * dt), math.cos(omega * dt)
    return np.array(
        [
            [1.0, 0.0, sin / omega, -(1.0 - cos) / omega],
            [0.0, 1.0, (1.0 - cos) / omega, sin / omega],
            [0.0, 0.0, cos, -sin],
            [0.0, 0.0, sin, cos],
        ]
    )


def _model(mode: int, dt: float, s: MotionSettings) -> tuple[FloatArray, FloatArray]:
    if mode == STATIONARY:
        f = np.diag([1.0, 1.0, 0.0, 0.0])
        q = np.diag([s.q_stationary_m2ps * dt, s.q_stationary_m2ps * dt, 1e-6, 1e-6])
        return f, q
    if mode in _TURNS:
        omega = s.turn_rate_radps if mode == TURNING_LEFT else -s.turn_rate_radps
        return _coordinated_turn(omega, dt), _white_acceleration(s.sigma_a_turning, dt)
    sigma = s.sigma_a_moving if mode == MOVING else s.sigma_a_manoeuvring
    return _coordinated_turn(0.0, dt), _white_acceleration(sigma, dt)


def _symmetric(p: FloatArray) -> FloatArray:
    out: FloatArray = (p + p.T) / 2.0
    return out


class IMM:
    def __init__(self, z: FloatArray, r: FloatArray, settings: MotionSettings) -> None:
        if len(settings.initial_probabilities) != len(MODES):
            raise ValueError(f"initial_probabilities needs {len(MODES)} values, one per mode {MODES}")
        self.s = settings
        p = np.zeros((4, 4))
        p[:2, :2] = r
        p[2:, 2:] = np.eye(2) * settings.v0_sigma_mps**2
        x = np.array([z[0], z[1], 0.0, 0.0])
        self.x: FloatArray = np.stack([x.copy() for _ in MODES])
        self.p: FloatArray = np.stack([p.copy() for _ in MODES])
        mu = np.maximum(np.array(settings.initial_probabilities, dtype=np.float64), 1e-6)
        self.mu: FloatArray = mu / mu.sum()

    def copy(self) -> IMM:
        other = IMM.__new__(IMM)
        other.s, other.x, other.p, other.mu = self.s, self.x.copy(), self.p.copy(), self.mu.copy()
        return other

    def predict(self, dt: float) -> None:
        """Mix the models, then move each forward ``dt`` seconds (0 allowed)."""
        dt = max(dt, 0.0)
        n = len(MODES)
        pi = _transition(dt, self.s)
        c = pi.T @ self.mu
        c = np.maximum(c, 1e-300)
        mixing = (pi * self.mu[:, None]) / c[None, :]  # mixing[i, j] = P(was i | now j)
        x0 = mixing.T @ self.x
        p0 = np.zeros_like(self.p)
        for j in range(n):
            for i in range(n):
                d = self.x[i] - x0[j]
                p0[j] += mixing[i, j] * (self.p[i] + np.outer(d, d))
        for j in range(n):
            f, q = _model(j, dt, self.s)
            self.x[j] = f @ x0[j]
            self.p[j] = _symmetric(f @ p0[j] @ f.T + q)
        self.mu = c / c.sum()

    def combined(self) -> tuple[FloatArray, FloatArray]:
        """Moment-matched mean and covariance over the models."""
        x: FloatArray = self.mu @ self.x
        p: FloatArray = np.zeros((4, 4))
        for j in range(len(MODES)):
            d = self.x[j] - x
            p += self.mu[j] * (self.p[j] + np.outer(d, d))
        return x, _symmetric(p)

    def mahalanobis2(self, z: FloatArray, r: FloatArray | None) -> float:
        """Squared distance of a measurement from the combined prediction (``r`` None: track region only)."""
        x, p = self.combined()
        s = p[:2, :2] + (r if r is not None else 0.0)
        nu = z - x[:2]
        return float(nu @ np.linalg.solve(s, nu))

    def mahalanobis2_mode(self, z: FloatArray, r: FloatArray, mode: int) -> float:
        """Squared distance under one model's prediction alone (0: stationary)."""
        s = self.p[mode][:2, :2] + r
        nu = z - self.x[mode][:2]
        return float(nu @ np.linalg.solve(s, nu))

    def update(self, z: FloatArray, r: FloatArray) -> float:
        """Kalman update of every model, re-weighting the models by their likelihood."""
        n = len(MODES)
        log_l = np.zeros(n)
        for j in range(n):
            s = _H @ self.p[j] @ _H.T + r
            nu = z - _H @ self.x[j]
            k = self.p[j] @ _H.T @ np.linalg.inv(s)
            self.x[j] = self.x[j] + k @ nu
            a = _I4 - k @ _H
            self.p[j] = _symmetric(a @ self.p[j] @ a.T + k @ r @ k.T)  # Joseph form
            _, logdet = np.linalg.slogdet(s)
            log_l[j] = -0.5 * (float(nu @ np.linalg.solve(s, nu)) + logdet + 2 * math.log(2 * math.pi))
        weights = np.log(np.maximum(self.mu, 1e-300)) + log_l
        weights -= weights.max()
        mu = np.exp(weights)
        self.mu = mu / mu.sum()
        return float(np.max(log_l))

    def turn_rate(self) -> tuple[float, float] | None:
        """Turn rate in rad/s (positive: turning left, anticlockwise) and its 1-sigma, or None.

        Read off the moving models, weighted by their probabilities *given that the emitter
        is moving* (a stationary emitter has no heading to turn). Each model says the rate is
        near its own — 0 for moving and manoeuvring, ±ω for the turns — give or take what its
        acceleration noise allows at the current speed (sigma_a / speed). The estimate is that
        mixture's mean and standard deviation: between -ω and +ω, and honest about a real rate
        beyond ω only through the spread. None when the models give no speed to divide by, or
        when the moving models together hold under 1 % of the probability.
        """
        moving = [j for j in range(len(MODES)) if j != STATIONARY]
        weights = self.mu[moving]
        total = float(weights.sum())
        if total < 0.01:
            return None
        x, _ = self.combined()
        speed = math.hypot(float(x[2]), float(x[3]))
        if speed < 1e-3:
            return None
        s = self.s
        rates = {
            MOVING: 0.0,
            MANOEUVRING: 0.0,
            TURNING_LEFT: s.turn_rate_radps,
            TURNING_RIGHT: -s.turn_rate_radps,
        }
        noise = {
            MOVING: s.sigma_a_moving,
            MANOEUVRING: s.sigma_a_manoeuvring,
            TURNING_LEFT: s.sigma_a_turning,
            TURNING_RIGHT: s.sigma_a_turning,
        }
        w = weights / total
        mean = float(sum(wj * rates[j] for wj, j in zip(w, moving, strict=True)))
        second = float(
            sum(
                wj * ((noise[j] / speed) ** 2 + (rates[j] - mean) ** 2)
                for wj, j in zip(w, moving, strict=True)
            )
        )
        return mean, math.sqrt(second)

    @property
    def probabilities(self) -> dict[str, float]:
        return {name: float(p) for name, p in zip(MODES, self.mu, strict=True)}
