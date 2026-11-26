"""Pipeline presets: a whole configuration, named, for the three shapes of run that work today.

A preset is nothing but a configuration with the numbers filled in — every field can be overridden:
``SurrogatePipeline.from_preset("burgers_in_viscosity", training={"epochs": 1500})``. They exist so that the first
run is one line, and so that the examples have something to compare against.

==============================  ==========================================================  ==================
preset                          what it builds                                              backend
==============================  ==========================================================  ==================
``burgers_in_viscosity``        a parametric surrogate: viscosity in, the velocity field     supervised
                                at t = 1 out, from the preset's own FDM reference solver
``burgers_physics_only``        a PINN trained on the Burgers equation alone, scored         physics-only
                                against the Cole-Hopf closed form
``heat_2d_symbolic``            a PINN on a PDE written as text (heat equation), trained     symbolic
                                and validated by the PINN validation suite
``burgers_fdm_reference``       a PINN on Burgers, scored against the preset's own FDM         physics-only
                                reference solution (a grid, so the discretisation gate runs)
==============================  ==========================================================  ==================
"""
from __future__ import annotations

from typing import Any

__all__ = ["PRESETS", "preset_config"]

PRESETS: dict[str, dict[str, Any]] = {
    "burgers_in_viscosity": {
        "name": "burgers_in_viscosity",
        "description": "parametric surrogate of the 1-D Burgers solution at t = 1 as a function of the viscosity",
        "problem": {"source": "preset", "preset": "burgers_1d"},
        "parameters": {"parameters": {"nu": [0.005, 0.06]}, "budget": 24},
        "sampling": {"design_sampling": {"method": "lhs", "n_samples": 18},
                     "validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"},
                     "seed": 3},
        "data": {"source": "simulation", "probes": {"x": 7}, "probe_at": {"t": 1.0}},
        "solver": {"method": "reference"},
        "architecture": {"family": "auto", "candidates": ["gp", "mlp"]},
        "training": {"budget_seconds": 2.0, "epochs": 400, "seed": 3},
        "verification": {"mode": "balanced", "fail_policy": "block",
                         "acceptance": {"nrmse": 0.10, "refinement": 0.05}},
        "export": {"formats": ["pytorch"], "verify_export": True},
    },
    "burgers_physics_only": {
        "name": "burgers_physics_only",
        "description": "PINN of the Burgers equation with no data, scored against the Cole-Hopf solution",
        "problem": {"source": "preset", "preset": "burgers_1d", "preset_kwargs": {"nu": 0.01}},
        "data": {"source": "physics_only"},
        "solver": {"method": "analytic"},
        "architecture": {"family": "modified_mlp"},
        "training": {"epochs": 1200, "lr": 1e-3, "n_collocation": 2048, "hidden_dim": 48, "n_layers": 4, "seed": 0,
                     "device": "cpu"},
        "loss": {"weights": {"w_pde": 1.0, "w_bc": 10.0, "w_ic": 10.0, "w_data": 1.0}},
        "verification": {"mode": "balanced", "fail_policy": "warn",
                         "acceptance": {"relative_l2": 0.10, "bc": 5e-2}},
        "export": {"formats": ["pytorch", "onnx"], "verify_export": True},
    },
    "heat_2d_symbolic": {
        "name": "heat_2d_symbolic",
        "description": "PINN on a heat equation written as text, trained and validated by the PINN validation suite",
        "problem": {
            "source": "symbolic",
            "coords": ["x", "y", "t"],
            "fields": ["u"],
            "domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.05]},
            "equations": ["d(u, t) - alpha * lap(u)"],
            "constants": {"alpha": 0.1},
            "conditions": [
                {"kind": "initial", "on": "t_min", "equation": "u = sin(pi*x) * sin(pi*y)"},
                {"kind": "dirichlet", "on": "boundary", "equation": "u = 0"},
            ],
        },
        "data": {"source": "physics_only"},
        "architecture": {"family": "pinn_problem_mlp"},
        "training": {"epochs": 2000, "lr": 2e-3, "n_collocation": 2000, "lbfgs_steps": 400, "seed": 0},
        "loss": {"weights": {"w_pde": 1.0, "w_bc": 50.0, "w_ic": 50.0, "w_data": 1.0}},
        "verification": {"mode": "balanced", "fail_policy": "warn"},
        "export": {"enabled": False, "formats": []},
        "sampling": {"validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"}},
    },
    "burgers_fdm_reference": {
        "name": "burgers_fdm_reference",
        "description": "PINN on Burgers, scored against the preset's own finite-difference reference solution",
        "problem": {"source": "preset", "preset": "burgers_1d", "preset_kwargs": {"nu": 0.01}},
        "data": {"source": "physics_only"},
        "solver": {"method": "reference"},
        "architecture": {"family": "modified_mlp"},
        "training": {"epochs": 1200, "lr": 1e-3, "n_collocation": 2048, "hidden_dim": 48, "n_layers": 4, "seed": 0},
        "verification": {"mode": "balanced", "fail_policy": "warn", "acceptance": {"relative_l2": 0.10}},
        "export": {"formats": ["pytorch"]},
    },
}


def _merge(base: dict, overrides: dict) -> dict:
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def preset_config(preset: str, **overrides: Any):
    """The :class:`~pinneapple_pipeline.configs.SurrogateConfig` of a preset, with ``overrides`` merged in.

    The preset's key is not the run's name: ``name="..."`` in ``overrides`` renames the run.
    """
    from .configs import SurrogateConfig
    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; the presets are {sorted(PRESETS)}")
    data = _merge(PRESETS[preset], overrides)
    data.setdefault("name", preset)
    return SurrogateConfig.from_dict(data)
