"""Verification of unforced two-body trajectories in SI units."""
import numpy as np
from pinneapple_simulation.orbital_intent import invariants
from .surrogate_gates import SurrogateGateReport, SurrogateGateResult

def verify_orbit(mu, states, *, energy_tolerance, momentum_tolerance,
                 reference=None, position_tolerance=None, velocity_tolerance=None,
                 model="", bundle_id=""):
    states = np.asarray(states, dtype=float)
    if len(states) < 2:
        raise ValueError("At least two states required")
    e, h = invariants(mu, states)
    report = SurrogateGateReport(model=model, bundle_id=bundle_id, field="orbital_state")
    def gate(name, value, tolerance, detail):
        if tolerance is None or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("A finite nonnegative tolerance is required")
        report.results.append(SurrogateGateResult(name, bool(np.isfinite(value) and value <= tolerance),
                              float(value), float(tolerance), detail, __name__))
    gate("orbital_energy", np.max(np.abs(e-e[0])), energy_tolerance, "Specific energy drift (m²/s²)")
    gate("orbital_momentum", np.max(np.linalg.norm(h-h[0], axis=1)), momentum_tolerance, "Specific angular momentum drift (m²/s)")
    if reference is None:
        for name in ("orbital_position", "orbital_velocity"):
            report.results.append(SurrogateGateResult(name, None, detail="No reference supplied", tool=__name__))
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != states.shape or not np.isfinite(ref).all():
            raise ValueError("Reference shape or values invalid")
        gate("orbital_position", np.max(np.linalg.norm(states[:,:3]-ref[:,:3], axis=1)), position_tolerance, "Maximum position error (m)")
        gate("orbital_velocity", np.max(np.linalg.norm(states[:,3:]-ref[:,3:], axis=1)), velocity_tolerance, "Maximum velocity error (m/s)")
    return report
