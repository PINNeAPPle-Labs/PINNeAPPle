"""Terramechanics physics residuals and preset for Bekker-Wong PINN surrogate.

Usage
-----
from pinneapple_physics.pde_environment.presets.terramechanics import (
    TerramechanicsResiduals,
    bekker_wong_surrogate_2d,
)

residuals = TerramechanicsResiduals()
r_dict = residuals(model, norm_x, norm_y)
loss = sum(v.mean() for v in r_dict.values())
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .registry import register_preset
from ..spec import PDETermSpec, ProblemSpec
from ..conditions import InitialCondition


class TerramechanicsResiduals:
    """Physics residuals for the Bekker-Wong PINN surrogate.

    Only constraints that were checked numerically against the Bekker-Wong solver
    (``pinneapple_simulation.numerical_solvers.bekker_wong``) on the preset domain
    s in [0, 0.75], z in [0.002, 0.058] m (GRC-1 defaults) are imposed:

    R2 -- Mohr-Coulomb traction limit: F_x <= c*A + F_z*tan(phi), A = b*R*theta_1   (ReLU^2)
    R3 -- Pre-peak monotonicity: dF_x/ds >= 0 for s in [0, 0.4]                      (autograd)
    R4 -- Torque coupling: M_y >= R * F_x
    R5 -- Load monotonicity: dF_z/dz >= 0

    Removed: the former "R1: F_x(s=0) = 0". It is not a property of the Bekker-Wong model:
    at zero slip the shear displacement j(theta) = R[(theta_f-theta) - (sin theta_f - sin theta)]
    is non-zero, and the solver returns F_x(0, z) between about -4.2 N and +5.0 N on the
    preset domain (compaction resistance vs. rear-region shear). Imposing it biased the
    surrogate against its own training data. R3 is only valid on the domain above -- it is
    violated for sinkages beyond ~0.06 m, so re-check it before widening the domain.

    Parameters
    ----------
    c_Pa : soil cohesion [Pa]
    phi_deg : internal friction angle [degrees]
    R_m : wheel radius [m]
    b_m : wheel width [m]
    n_phys : number of collocation points per residual
    R_factor : scale factor converting normalised M_y vs F_x for R4
    slip_range, sink_range : collocation domain (must stay inside the verified domain)
    """

    def __init__(
        self,
        c_Pa: float = 1_400.0,
        phi_deg: float = 30.0,
        R_m: float = 0.125,
        b_m: float = 0.060,
        n_phys: int = 256,
        R_factor: float = 0.06,
        slip_range: Tuple[float, float] = (0.0, 0.75),
        sink_range: Tuple[float, float] = (0.002, 0.058),
    ):
        self.c = c_Pa
        self.tan_phi = math.tan(math.radians(phi_deg))
        self.R = R_m
        self.b = b_m
        self.n_phys = n_phys
        self.R_factor = R_factor
        self.slip_range = slip_range
        self.sink_range = sink_range

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        model: nn.Module,
        norm_x: Any,
        norm_y: Any,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all physics residuals.

        Parameters
        ----------
        model : PINN model, maps normalised (s, z) -> normalised (Fx, Fz, My)
        norm_x : normaliser for inputs  (must have transform_torch / inverse)
        norm_y : normaliser for outputs (must have transform_torch / inverse)
        device : target device (defaults to first model parameter device)

        Returns
        -------
        dict with keys "r2", "r3", "r4", "r5" -- each a scalar loss tensor
        """
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        n = self.n_phys
        s_lo, s_hi = self.slip_range
        z_lo, z_hi = self.sink_range

        def sample(k=n, s_max=None):
            sm = s_hi if s_max is None else min(s_max, s_hi)
            s_ = torch.rand(k, 1, device=device) * (sm - s_lo) + s_lo
            z_ = torch.rand(k, 1, device=device) * (z_hi - z_lo) + z_lo
            return s_, z_

        # R2: Mohr-Coulomb traction limit (soft, one-sided), A = b*R*theta_1
        s_r2, z_r2 = sample()
        phy_r2 = norm_y.inverse_torch(model(norm_x.transform_torch(torch.cat([s_r2, z_r2], dim=1))))
        theta1 = torch.acos(torch.clamp(1.0 - z_r2 / self.R, -1.0 + 1e-6, 1.0 - 1e-6))
        limit = self.c * self.b * self.R * theta1 + phy_r2[:, 1:2].detach() * self.tan_phi
        r2 = (torch.nn.functional.relu(phy_r2[:, 0:1] - limit) ** 2).mean()

        # R3: dF_x/ds >= 0 for s in [0, 0.4]
        s_r3, z_r3 = sample(s_max=0.4)
        s_r3 = s_r3.detach().requires_grad_(True)
        fx_r3_n = model(norm_x.transform_torch(torch.cat([s_r3, z_r3], dim=1)))[:, 0:1]
        dfx_ds = torch.autograd.grad(fx_r3_n, s_r3, grad_outputs=torch.ones_like(fx_r3_n),
                                     create_graph=True, retain_graph=True)[0]
        r3 = (torch.nn.functional.relu(-dfx_ds) ** 2).mean()

        # R4: M_y >= R * F_x (in normalised units via R_factor)
        s_r4, z_r4 = sample()
        pred_r4 = model(norm_x.transform_torch(torch.cat([s_r4, z_r4], dim=1)))
        r4 = (torch.nn.functional.relu(self.R_factor * pred_r4[:, 0:1] - pred_r4[:, 2:3]) ** 2).mean()

        # R5: dF_z/dz >= 0
        s_r5, z_r5 = sample()
        z_r5 = z_r5.detach().requires_grad_(True)
        fz_n = model(norm_x.transform_torch(torch.cat([s_r5, z_r5], dim=1)))[:, 1:2]
        dfz_dz = torch.autograd.grad(fz_n, z_r5, grad_outputs=torch.ones_like(fz_n), create_graph=True)[0]
        r5 = (torch.nn.functional.relu(-dfz_dz) ** 2).mean()

        return {"r2": r2, "r3": r3, "r4": r4, "r5": r5}


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

@register_preset("bekker_wong_surrogate_2d")
def bekker_wong_surrogate_2d(
    c_Pa: float = 1_400.0,
    phi_deg: float = 30.0,
    R_m: float = 0.125,
    b_m: float = 0.060,
) -> ProblemSpec:
    """Return a ProblemSpec for the 2-D Bekker-Wong PINN surrogate.

    Inputs : (slip_ratio, sinkage_m)
    Outputs: (F_x, F_z, M_y)
    """
    pde = PDETermSpec(
        kind="bekker_wong_terramechanics",
        fields=("Fx", "Fz", "My"),
        coords=("slip", "sinkage"),
        params={
            "c_Pa": float(c_Pa),
            "phi_deg": float(phi_deg),
            "R_m": float(R_m),
            "b_m": float(b_m),
        },
        meta={
            "description": "Bekker-Wong rigid-wheel / deformable-soil surrogate",
            "physics_constraints": [
                "R2: Fx <= c*A + Fz*tan(phi)",
                "R3: dFx/ds >= 0 for s in [0, 0.4]",
                "R4: My >= R*Fx",
                "R5: dFz/dz >= 0",
            ],
            "removed_constraints": {"R1: Fx(s=0) = 0": "false for Bekker-Wong (Fx(0,z) ranges -4.2..+5.0 N)"},
        },
    )
    # No F_x(slip=0) = 0 condition: it contradicts the Bekker-Wong model itself (see
    # TerramechanicsResiduals' docstring for the numerical evidence).
    return ProblemSpec(
        name="bekker_wong_surrogate_2d",
        dim=2,
        coords=("slip", "sinkage"),
        fields=("Fx", "Fz", "My"),
        pde=pde,
        conditions=(),
        domain_bounds={"slip": (0.0, 0.75), "sinkage": (0.002, 0.058)},
        meta={"description": "Bekker-Wong rigid-wheel terramechanics surrogate for rover simulation"},
    )


__all__ = ["TerramechanicsResiduals", "bekker_wong_surrogate_2d"]
