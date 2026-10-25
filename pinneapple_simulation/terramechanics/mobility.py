"""Wheel mobility screening with the ``bekker_wong_surrogate_2d`` PINN: train it, find the sinkage that carries a
given load, read drawbar pull against slip, and re-check each physics constraint on its own.

The preset trains (slip, sinkage) -> (Fx drawbar pull, Fz normal force, My torque) for a rigid wheel under
constraints that are inequalities, not a PDE residual:

* R1 ``Fx(slip = 0) = 0`` (an initial condition),
* R2 the Mohr-Coulomb traction limit ``Fx <= c A + Fz tan(phi)`` with the rigid-wheel contact area
  ``A = b R theta1``, ``theta1 = arccos(1 - z / R)`` (Bekker 1956/1969; Wong & Reece 1967),
* R3 ``dFx/ds >= 0`` for slip <= 0.4,
* R4 ``My >= R Fx``.

:func:`mohr_coulomb_traction_limit` is that R2 bound in closed form, written independently of the compiled
residual; ``tests/test_terramechanics_mobility.py`` checks that the two agree (the source app's
``verify_analytical_crosscheck.py``). :func:`analyze_wheel_surrogate` reports R1-R4 per slip point next to the
guardrail, because the guardrail's ``pde_residual`` (a) pools R2-R5 into one mean, so one violated constraint is
diluted by the others, and (b) never evaluates conditions given as ``InitialCondition`` (R1). A pooled residual
of 0.0 can therefore sit next to a surrogate that collapsed to Fx ~ 0 (the constraints bound relations between
Fx, Fz and My but never anchor Fz's scale): read ``max_drawbar_pull_N`` and ``sinkage_solve_reliable`` too.

The sinkage for a target load is found on a dense grid of the surrogate's own Fz(slip, sinkage), so it inherits
the surrogate's error (reported per row as ``sinkage_solve_error_N``). Rigid wheel, 2-D, single wheel.

Moved here from the PINNeAPPle-apps ``rover_mobility_analysis`` product.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

__all__ = ["contact_entry_angle", "mohr_coulomb_traction_limit", "train_wheel_surrogate", "analyze_wheel_surrogate",
           "mobility_analysis"]


def contact_entry_angle(R_m: float, sinkage_m: float) -> float:
    """theta1 = arccos(1 - z/R) of a rigid wheel of radius R sunk to depth z (0 < z < R)."""
    if not 0.0 < sinkage_m < R_m:
        raise ValueError(f"sinkage {sinkage_m} must be in (0, R={R_m})")
    return math.acos(1.0 - sinkage_m / R_m)


def mohr_coulomb_traction_limit(c_Pa: float, phi_deg: float, R_m: float, b_m: float, sinkage_m: float,
                                Fz_N: float) -> float:
    """Largest drawbar pull the soil shear strength allows: c A + Fz tan(phi), A = b R theta1 (N)."""
    return c_Pa * b_m * R_m * contact_entry_angle(R_m, sinkage_m) + Fz_N * math.tan(math.radians(phi_deg))


def _out(y):
    return y.y if hasattr(y, "y") else y


def train_wheel_surrogate(*, c_Pa: float, phi_deg: float, R_m: float, b_m: float, epochs: int = 2000,
                          n_collocation: int = 1024, architecture: str = "modified_mlp", hidden_dim: int = 64,
                          n_layers: int = 4, seed: int = 0):
    """Train the ``bekker_wong_surrogate_2d`` PINN; returns ``(spec, model, final_training_loss)``."""
    import torch

    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry

    spec = pp.get_preset("bekker_wong_surrogate_2d", c_Pa=c_Pa, phi_deg=phi_deg, R_m=R_m, b_m=b_m)
    torch.manual_seed(seed)
    model = ModelRegistry.build(architecture, in_dim=2, out_dim=3, hidden_dim=hidden_dim, n_layers=n_layers)
    result = pp.solve_pde(spec, model, epochs=epochs, n_collocation=n_collocation, seed=seed)
    return spec, model, float(result["history"]["loss"][-1])


def analyze_wheel_surrogate(model, spec, *, c_Pa: float, phi_deg: float, R_m: float, b_m: float, Fz_N: float,
                            slip_min: float = 0.0, slip_max: float = 0.7, n_slip: int = 15,
                            n_sinkage: int = 400) -> dict[str, Any]:
    """Drawbar pull against slip at the sinkage that carries ``Fz_N``, with R1-R4 checked one by one."""
    import torch

    if c_Pa < 0 or not 0 < phi_deg < 90 or R_m <= 0 or b_m <= 0 or Fz_N <= 0:
        raise ValueError("need c_Pa >= 0, 0 < phi_deg < 90, R_m > 0, b_m > 0, Fz_N > 0")
    if not 0 <= slip_min < slip_max or n_slip < 2:
        raise ValueError("need 0 <= slip_min < slip_max and n_slip >= 2")
    (s_lo, s_hi), (z_lo, z_hi) = spec.domain_bounds["slip"], spec.domain_bounds["sinkage"]
    warnings = []
    if slip_min < s_lo or slip_max > s_hi:
        warnings.append(f"slip range [{slip_min}, {slip_max}] clipped to the surrogate's domain [{s_lo}, {s_hi}]")
        slip_min, slip_max = max(slip_min, s_lo), min(slip_max, s_hi)
        if slip_min >= slip_max:
            raise ValueError(f"slip range does not overlap the domain [{s_lo}, {s_hi}]")

    model.eval()
    slips = np.linspace(slip_min, slip_max, n_slip)
    sink = np.linspace(z_lo, z_hi, n_sinkage)
    S, Z = np.meshgrid(slips, sink, indexing="ij")
    with torch.no_grad():
        Y = _out(model(torch.as_tensor(np.stack([S.ravel(), Z.ravel()], 1), dtype=torch.float32))).numpy()
    Fx, Fz, My = (Y[:, k].reshape(n_slip, n_sinkage) for k in range(3))
    j = np.argmin(np.abs(Fz - Fz_N), axis=1)
    rows = np.arange(n_slip)
    z_sol, fx, fz, my = sink[j], Fx[rows, j], Fz[rows, j], My[rows, j]
    err = np.abs(fz - Fz_N)
    reliable = bool(err.max() <= 0.2 * Fz_N)
    if not reliable:
        warnings.append(f"sinkage solve: worst Fz error {err.max():.3g} N for a {Fz_N:.3g} N load (> 20 %); "
                        "the surrogate's Fz surface may not reach this load")

    limit = np.array([mohr_coulomb_traction_limit(c_Pa, phi_deg, R_m, b_m, float(z), float(f))
                      for z, f in zip(z_sol, fz, strict=True)])
    r2_ok = fx <= limit + 1e-6
    r4_ok = my >= R_m * fx - 1e-6
    r3_mask = slips[:-1] <= 0.4
    r3_viol = int(np.sum((np.diff(fx) < -1e-6) & r3_mask))
    z_probe = np.linspace(z_lo, z_hi, 25)
    with torch.no_grad():
        fx0 = _out(model(torch.as_tensor(np.stack([np.zeros_like(z_probe), z_probe], 1),
                                         dtype=torch.float32))).numpy()[:, 0]
    r1_scale = c_Pa * b_m * R_m * math.pi + Fz_N * math.tan(math.radians(phi_deg))
    r1_tol = max(0.02 * r1_scale, 1e-3)
    r1_max = float(np.abs(fx0).max())
    k = int(np.argmax(fx))
    per_slip = [{"slip": float(slips[i]), "sinkage_m": float(z_sol[i]), "Fz_N": float(fz[i]),
                 "sinkage_solve_error_N": float(err[i]), "Fx_drawbar_pull_N": float(fx[i]), "My_Nm": float(my[i]),
                 "mohr_coulomb_limit_N": float(limit[i]), "r2_within_limit": bool(r2_ok[i]),
                 "r4_torque_covers_thrust": bool(r4_ok[i])} for i in range(n_slip)]
    checks = {"r1_zero_pull_at_zero_slip": {"passed": r1_max <= r1_tol, "max_abs_Fx_N": r1_max, "tolerance_N": r1_tol},
              "r2_mohr_coulomb": {"violations": int(np.sum(~r2_ok)), "out_of": n_slip},
              "r3_monotone_up_to_slip_0_4": {"violations": r3_viol, "out_of": int(r3_mask.sum())},
              "r4_torque_covers_thrust": {"violations": int(np.sum(~r4_ok)), "out_of": n_slip}}
    consistent = (checks["r1_zero_pull_at_zero_slip"]["passed"] and not checks["r2_mohr_coulomb"]["violations"]
                  and not r3_viol and not checks["r4_torque_covers_thrust"]["violations"])
    return {"per_slip": per_slip, "max_drawbar_pull_N": float(fx[k]), "max_drawbar_pull_at_slip": float(slips[k]),
            "sinkage_solve_reliable": reliable, "max_sinkage_solve_error_N": float(err.max()),
            "constraint_checks": checks, "constraints_consistent": bool(consistent), "warnings": warnings}


def mobility_analysis(*, c_Pa: float = 1400.0, phi_deg: float = 30.0, R_m: float = 0.125, b_m: float = 0.060,
                      Fz_N: float = 50.0, slip_min: float = 0.0, slip_max: float = 0.7, n_slip: int = 15,
                      epochs: int = 2000, n_collocation: int = 1024, architecture: str = "modified_mlp",
                      hidden_dim: int = 64, n_layers: int = 4, residual_threshold: float = 1e-2,
                      seed: int = 0) -> dict[str, Any]:
    """Train, guardrail-check and analyse the wheel surrogate for one soil and wheel (blocking)."""
    import pinneapple_llm as pl
    from pinneapple_analysis.verification.pipeline import guardrail_checks_as_dicts

    t0 = time.time()
    spec, model, loss = train_wheel_surrogate(c_Pa=c_Pa, phi_deg=phi_deg, R_m=R_m, b_m=b_m, epochs=epochs,
                                              n_collocation=n_collocation, architecture=architecture,
                                              hidden_dim=hidden_dim, n_layers=n_layers, seed=seed)
    report = pl.PhysicsGuardrail(spec, residual_threshold=residual_threshold).check(model)
    out = analyze_wheel_surrogate(model, spec, c_Pa=c_Pa, phi_deg=phi_deg, R_m=R_m, b_m=b_m, Fz_N=Fz_N,
                                  slip_min=slip_min, slip_max=slip_max, n_slip=n_slip)
    out["inputs"] = {"c_Pa": c_Pa, "phi_deg": phi_deg, "R_m": R_m, "b_m": b_m, "Fz_N": Fz_N}
    out["training"] = {"epochs": epochs, "architecture": architecture, "final_training_loss": loss,
                       "seconds": round(time.time() - t0, 2)}
    out["guardrail"] = {"trustworthy": bool(report.trustworthy), "checks": guardrail_checks_as_dicts(report),
                        "skipped": list(report.skipped)}
    out["verdict"] = ("physically consistent" if report.trustworthy and out["constraints_consistent"]
                      else "not fully consistent: see constraint_checks and guardrail")
    return out
