"""Rocket nozzle screening by two methods that do not share code: a PINN and the ideal 1-D isentropic nozzle.

:func:`screen_nozzle` runs, on the same inputs,

1. :func:`pinneapple_physics.closed_form.rocket_nozzle.nozzle_performance` (exit Mach, pressure, temperature,
   C_F, Isp of an ideal nozzle), and
2. a PINN on the ``rocket_nozzle_cfd`` preset (axisymmetric compressible Euler), trained with
   :func:`pinneapple_physics.solve_pde` on the preset's own tagged geometry
   (:func:`~pinneapple_physics.pde_environment.presets.tag_geometry.build_tag_batch`: the divergent cone from the
   throat to the exit, no invented chamber), re-checked by :class:`pinneapple_llm.PhysicsGuardrail`, and read
   at the exit plane (section average over the radius),

then reports the relative difference of exit Mach number and exit pressure and whether the two agree within a
tolerance. Disagreement is the expected outcome today, not a failure of this function: the preset's residual is
in raw SI units (pressure ~1e6 Pa next to velocity ~1e3 m/s; the source app measured residuals of 1e22-1e26
against the guardrail's 1e-2 threshold and 73-4764 % disagreement), its wall condition is no-slip on inviscid
equations, and its domain box extends past the cone. The network output is rescaled per field
(:func:`scaled_field_head`) so each field starts at its physical magnitude; that changes units, not physics, and
does not make a short run converge.

Moved here from the PINNeAPPle-apps ``rocket_nozzle_screening`` product. Its PINN half used its own
convergent-divergent wall (a chamber 1.6 x the throat radius, the throat at 30 % of the length); this version uses
the library's tagged geometry for the preset instead.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["scaled_field_head", "train_nozzle_pinn", "screen_nozzle"]


def _torch():
    import torch
    return torch


def scaled_field_head(net, scale, offset):
    """Wrap ``net`` so that ``y = net(x) * scale + offset`` per output field (fixed, non-trainable buffers)."""
    torch = _torch()

    class ScaledFieldHead(torch.nn.Module):
        """``y = net(x) * scale + offset`` per field, with fixed (non-trainable) scale and offset."""

        def __init__(self, net, scale, offset):
            super().__init__()
            self.net = net
            self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
            self.register_buffer("offset", torch.as_tensor(offset, dtype=torch.float32))

        def forward(self, x):
            y = self.net(x)
            y = y.y if hasattr(y, "y") else y
            return y * self.scale + self.offset

    return ScaledFieldHead(net, scale, offset)


def train_nozzle_pinn(*, throat_radius: float, exit_radius: float, nozzle_length: float, p_chamber: float,
                      T_chamber: float, gamma: float, R_gas: float, epochs: int = 300, n_collocation: int = 1024,
                      n_bc_per_face: int = 300, architecture: str = "modified_mlp", hidden_dim: int = 64,
                      n_layers: int = 4, residual_threshold: float = 1e-2, seed: int = 0) -> dict[str, Any]:
    """Train the ``rocket_nozzle_cfd`` PINN, run the guardrail and return the exit-plane section averages."""
    torch = _torch()
    import pinneapple_llm as pl
    import pinneapple_neural.architectures  # noqa: F401  registers the zoo
    import pinneapple_physics as pp
    from pinneapple_neural.architectures.registry import ModelRegistry
    from pinneapple_physics.pde_environment.presets.tag_geometry import (
        build_tag_batch,
        solve_pde_kwargs_from_batch,
    )

    from .pipeline import guardrail_checks_as_dicts

    spec = pp.get_preset("rocket_nozzle_cfd", gamma=gamma, R_gas=R_gas, T_inlet=T_chamber, p_inlet=p_chamber,
                         throat_radius=throat_radius, exit_radius=exit_radius, nozzle_length=nozzle_length)
    fields = tuple(spec.fields)                        # ("rho", "u", "v", "p", "T")
    torch.manual_seed(seed)
    net = ModelRegistry.build(architecture, in_dim=len(spec.coords), out_dim=len(fields), hidden_dim=hidden_dim,
                              n_layers=n_layers)
    rho0, a0 = p_chamber / (R_gas * T_chamber), math.sqrt(gamma * R_gas * T_chamber)
    model = scaled_field_head(net, [rho0, a0, a0, p_chamber, T_chamber], [rho0, 0.2 * a0, 0.0, p_chamber,
                                                                            T_chamber])
    batch = build_tag_batch("rocket_nozzle_cfd", spec, n_col=n_collocation, n_bc_per_face=n_bc_per_face, seed=seed)
    result = pp.solve_pde(spec, model, epochs=epochs, n_collocation=n_collocation, seed=seed, device="cpu",
                          **solve_pde_kwargs_from_batch(batch))
    trained = result.get("model", model)
    report = pl.PhysicsGuardrail(spec, residual_threshold=residual_threshold).check(trained)

    trained.eval()
    r = np.linspace(0.0, 0.95 * exit_radius, 16)
    x = torch.as_tensor(np.stack([r, np.full_like(r, nozzle_length)], axis=1), dtype=torch.float32)
    with torch.no_grad():
        y = trained(x)
        y = (y.y if hasattr(y, "y") else y).numpy()
    i = {f: k for k, f in enumerate(fields)}
    T = np.clip(y[:, i["T"]], 1.0, None)
    mach = np.hypot(y[:, i["u"]], y[:, i["v"]]) / np.sqrt(np.maximum(gamma * R_gas * T, 1e-6))
    return {"mach_exit": float(mach.mean()), "p_exit_pa": float(y[:, i["p"]].mean()), "T_exit_k": float(T.mean()),
            "final_training_loss": float(result["history"]["loss"][-1]),
            "guardrail_trustworthy": bool(report.trustworthy), "guardrail_checks": guardrail_checks_as_dicts(report),
            "geometry": "rocket_nozzle_cfd tagged geometry: divergent cone, throat (inlet) to exit"}


def screen_nozzle(*, throat_radius: float, exit_radius: float, p_chamber: float, T_chamber: float,
                  gamma: float = 1.2, R_gas: float = 287.0, nozzle_length: float | None = None,
                  p_atm: float = 101325.0, tolerance: float = 0.30, run_pinn: bool = True,
                  **pinn_options: Any) -> dict[str, Any]:
    """Ideal-nozzle estimate and, with ``run_pinn``, the PINN estimate and their disagreement.

    ``nozzle_length`` defaults to 8 throat radii (a placeholder, as in the source app). ``tolerance`` is the
    relative difference of exit Mach and exit pressure above which the methods are reported as disagreeing.
    """
    from pinneapple_physics.closed_form.rocket_nozzle import nozzle_performance

    if exit_radius <= throat_radius:
        raise ValueError("exit_radius must be larger than throat_radius")
    length = float(nozzle_length if nozzle_length is not None else 8.0 * throat_radius)
    eps = (exit_radius / throat_radius) ** 2
    analytic = nozzle_performance(gamma=gamma, R_gas=R_gas, T_chamber=T_chamber, p_chamber=p_chamber,
                                  area_ratio_exit=eps, p_atm=p_atm)
    out: dict[str, Any] = {"inputs": {"throat_radius_m": throat_radius, "exit_radius_m": exit_radius,
                                      "nozzle_length_m": length, "area_ratio": eps, "p_chamber_pa": p_chamber,
                                      "T_chamber_k": T_chamber, "gamma": gamma, "R_gas": R_gas, "p_atm_pa": p_atm},
                           "analytic": analytic}
    if not run_pinn:
        return out
    pinn = train_nozzle_pinn(throat_radius=throat_radius, exit_radius=exit_radius, nozzle_length=length,
                             p_chamber=p_chamber, T_chamber=T_chamber, gamma=gamma, R_gas=R_gas, **pinn_options)
    dm = abs(pinn["mach_exit"] - analytic["mach_exit"]) / max(analytic["mach_exit"], 1e-12)
    dp = abs(pinn["p_exit_pa"] - analytic["p_exit_pa"]) / max(analytic["p_exit_pa"], 1e-12)
    agree = bool(dm <= tolerance and dp <= tolerance)
    reasons = []
    if not agree:
        reasons.append(f"exit Mach / exit pressure differ by {dm:.0%} / {dp:.0%} (tolerance {tolerance:.0%})")
    if not pinn["guardrail_trustworthy"]:
        reasons.append("the guardrail failed at least one check of the PINN")
    out.update({"pinn": pinn,
                "disagreement": {"mach_exit_relative": dm, "p_exit_relative": dp, "tolerance": tolerance,
                                 "methods_agree": agree},
                "verdict": "agree" if not reasons else "do not trust: " + "; ".join(reasons)})
    return out
