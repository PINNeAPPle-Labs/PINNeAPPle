"""CPU orbital reference solver and explicit external backend contract.

SI units throughout. No upstream code is vendored.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp


def propagate(mu, initial, times, *, backend=None):
    """Return Cartesian states (N, 6); optional backend uses same SI contract."""
    initial = np.asarray(initial, dtype=float)
    times = np.asarray(times, dtype=float)
    if not np.isfinite(mu) or mu <= 0 or initial.shape != (6,) or not np.isfinite(initial).all() or np.linalg.norm(initial[:3]) == 0:
        raise ValueError("Invalid gravitational parameter or initial state")
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Expected strictly increasing finite times")
    if backend is not None:
        result = np.asarray(backend(mu, initial.copy(), times.copy()), dtype=float)
    else:
        def rhs(t, y):
            r = y[:3]
            return np.r_[y[3:], -mu*r/np.linalg.norm(r)**3]
        sol = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times,
                        method="DOP853", rtol=1e-10, atol=1e-9)
        if not sol.success:
            raise RuntimeError(sol.message)
        result = sol.y.T
    if result.shape != (len(times), 6) or not np.isfinite(result).all():
        raise ValueError("Invalid backend trajectory")
    if not np.allclose(result[0], initial, rtol=1e-10, atol=1e-9):
        raise ValueError("Backend did not preserve initial state")
    return result


def hohmann(mu, r1, r2):
    """Signed tangential burns (m/s), coast time (s), total delta-v (m/s)."""
    if not np.isfinite([mu, r1, r2]).all() or min(mu, r1, r2) <= 0:
        raise ValueError("Expected positive finite SI inputs")
    a = (r1+r2)/2
    dv1 = np.sqrt(mu*(2/r1-1/a))-np.sqrt(mu/r1)
    dv2 = np.sqrt(mu/r2)-np.sqrt(mu*(2/r2-1/a))
    return dict(dv1=float(dv1), dv2=float(dv2), coast_time=float(np.pi*np.sqrt(a**3/mu)),
                delta_v=float(abs(dv1)+abs(dv2)))


def invariants(mu, states):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 6 or not np.isfinite(states).all() or not np.isfinite(mu) or mu <= 0:
        raise ValueError("Invalid trajectory")
    r, v = states[:, :3], states[:, 3:]
    radii = np.linalg.norm(r, axis=1)
    if np.any(radii == 0):
        raise ValueError("Singular radius")
    return np.sum(v*v, axis=1)/2-mu/radii, np.cross(r, v)


@dataclass(frozen=True)
class ScientificIntent:
    """Explicit hypothesis and acceptance checks; not a VeriPhysics certificate."""
    objective: str
    assumptions: tuple[str, ...]
    energy_tolerance: float
    momentum_tolerance: float

    def evaluate(self, mu, states):
        tolerances = np.asarray([self.energy_tolerance, self.momentum_tolerance])
        if not np.isfinite(tolerances).all() or np.any(tolerances < 0):
            raise ValueError("Invalid tolerances")
        e, h = invariants(mu, states)
        energy = float(np.max(np.abs(e-e[0])))
        momentum = float(np.max(np.linalg.norm(h-h[0], axis=1)))
        return dict(energy_error=energy, momentum_error=momentum,
                    passed=bool(energy <= self.energy_tolerance and momentum <= self.momentum_tolerance))


def generate_dataset(mu, initial_states, times, *, backend=None):
    """Stack trajectories for supervised surrogate learning."""
    return np.stack([propagate(mu, s, times, backend=backend) for s in initial_states])
