"""How much faster is a trained surrogate than a numerical solve, and how far apart are they? A measured sweep over
seven plate-fin heat sinks.

Three models of the same heat sink (80 x 80 mm aluminium base, 3 mm thick, straight fins, uniform film coefficient):

* closed form: :mod:`pinneapple_physics.closed_form.fin_array_conduction` (1-D fins with adiabatic-tip correction +
  1-D base conduction, Incropera & DeWitt);
* surrogate: an MLP (:class:`pinneapple_design.design_optimizer.surrogate.PhysicsSurrogate`, 3 x 48 tanh) trained on
  6,000 samples of the closed form over (fin count, fin height, fin thickness, power, h);
* numerical reference: the finite-volume unit cell of
  :func:`pinneapple_simulation.numerical_solvers.fin_cell_fv.solve_fin_cell` (2-D cross-section, which is the same
  problem as the 3-D cell because nothing depends on the fin depth).

For each design the surrogate forward pass is timed (mean of ``timing_reps`` calls after one warm-up call) and the
reference solve is timed (wall clock of the solve), and the agreement is the difference of the temperature rise
above ambient, ``|dT_a - dT_b| / dT_ref``. The speed ratio depends on the machine and on the reference solver: this
reference is a sub-second 2-D solve, so the ratios are much smaller than against a meshed 3-D FEM run.

The seven designs stay at 55 W or more and inside the manufacturable box (fin spacing >= 1.2 mm, height 5-45 mm,
thickness 0.4-3 mm, 3-60 fins); the surrogate is trained over 15-150 W, so the relative error of a small temperature
rise is larger at the low-power end.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from pinneapple_physics.closed_form import fin_array_conduction as fac

__all__ = ["K_ALUMINUM", "T_AMBIENT_C", "BASE_W_M", "BASE_D_M", "BASE_THICKNESS_M", "COOLING_PROFILES",
           "FinArrayDesign", "junction_temperature_c", "constraint_violations", "CURATED_DESIGNS", "design_by_name",
           "train_surrogate", "time_surrogate", "reference_solve", "measure_design", "summarize", "run_sweep",
           "thermal_field", "AGREEMENT_THRESHOLD_PCT"]

K_ALUMINUM = 205.0                   # W/(m K)
T_AMBIENT_C = 25.0
BASE_W_M = BASE_D_M = 0.08
BASE_THICKNESS_M = 0.003
FIN_HEIGHT_BOUNDS_M = (0.005, 0.045)
FIN_THICKNESS_BOUNDS_M = (0.0004, 0.003)
FIN_COUNT_BOUNDS = (3, 60)
MIN_FIN_SPACING_M = 0.0012
COOLING_PROFILES = {"quiet_fan": 18.0, "standard_fan": 35.0, "high_static_pressure_fan": 65.0}   # W/(m2 K)
AGREEMENT_THRESHOLD_PCT = 10.0       # decision bound on the surrogate vs reference difference (temperature rise)

CURATED_DESIGNS: list[dict[str, Any]] = [
    {"name": "baseline", "label": "Baseline", "short": "Baseline", "fin_count": 20, "fin_height_m": 0.025,
     "fin_thickness_m": 0.0010, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "dense_thin_fins", "label": "Dense, thin fins", "short": "Dense fins", "fin_count": 44,
     "fin_height_m": 0.020, "fin_thickness_m": 0.0006, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "sparse_thick_fins", "label": "Sparse, thick fins", "short": "Sparse fins", "fin_count": 8,
     "fin_height_m": 0.030, "fin_thickness_m": 0.0025, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "tall_fins", "label": "Tall fins", "short": "Tall fins", "fin_count": 15, "fin_height_m": 0.035,
     "fin_thickness_m": 0.0010, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "short_fins", "label": "Short, dense fins", "short": "Short fins", "fin_count": 25,
     "fin_height_m": 0.008, "fin_thickness_m": 0.0012, "power_w": 65.0, "cooling_profile": "standard_fan"},
    {"name": "high_power_hs_fan", "label": "High power + strong fan", "short": "High power", "fin_count": 20,
     "fin_height_m": 0.025, "fin_thickness_m": 0.0010, "power_w": 130.0, "cooling_profile": "high_static_pressure_fan"},
    {"name": "lower_power_quiet", "label": "Lower power + quiet fan", "short": "Low power", "fin_count": 20,
     "fin_height_m": 0.025, "fin_thickness_m": 0.0010, "power_w": 55.0, "cooling_profile": "quiet_fan"},
]


@dataclass(frozen=True)
class FinArrayDesign:
    fin_count: float
    fin_height_m: float
    fin_thickness_m: float

    def fin_spacing_m(self) -> float:
        n = max(self.fin_count, 1.0)
        return BASE_W_M if n <= 1 else max((BASE_W_M - n * self.fin_thickness_m) / (n - 1), 1e-6)


def _profile(d: FinArrayDesign, power_w: float, h_conv: float) -> dict[str, float]:
    return fac.fin_array_profile_params(k_material_w_mk=K_ALUMINUM, h_conv_w_m2k=h_conv, n_fins=d.fin_count,
                                        fin_thickness_m=d.fin_thickness_m, fin_length_m=d.fin_height_m,
                                        transverse_extent_m=BASE_D_M, spacing_extent_m=BASE_W_M,
                                        base_thickness_m=BASE_THICKNESS_M, power_w=power_w)


def junction_temperature_c(d: FinArrayDesign, power_w: float, h_conv: float) -> float:
    """Base (junction-side) temperature from the closed form."""
    return fac.fin_array_base_temperature_c(_profile(d, power_w, h_conv), T_AMBIENT_C, power_w)


def constraint_violations(d: FinArrayDesign) -> dict[str, float]:
    """Signed violation of each manufacturability limit (<= 0 is satisfied)."""
    return {"min_spacing": MIN_FIN_SPACING_M - d.fin_spacing_m(),
            "min_height": FIN_HEIGHT_BOUNDS_M[0] - d.fin_height_m, "max_height": d.fin_height_m - FIN_HEIGHT_BOUNDS_M[1],
            "min_thickness": FIN_THICKNESS_BOUNDS_M[0] - d.fin_thickness_m,
            "max_thickness": d.fin_thickness_m - FIN_THICKNESS_BOUNDS_M[1],
            "min_count": FIN_COUNT_BOUNDS[0] - d.fin_count, "max_count": d.fin_count - FIN_COUNT_BOUNDS[1]}


def design_by_name(name: str) -> dict[str, Any]:
    """One of ``CURATED_DESIGNS`` (KeyError otherwise)."""
    for spec in CURATED_DESIGNS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def _design(spec: dict[str, Any]) -> FinArrayDesign:
    return FinArrayDesign(float(spec["fin_count"]), spec["fin_height_m"], spec["fin_thickness_m"])


_SCALES = torch.tensor([30.0, 0.025, 0.0015, 65.0, 40.0])
_T_SCALE = 150.0


class _Scale(torch.nn.Module):
    def __init__(self, factor: torch.Tensor | float, divide: bool) -> None:
        super().__init__()
        self.register_buffer("f", torch.as_tensor(factor, dtype=torch.float32))
        self.divide = divide

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.f if self.divide else x * self.f


def _samples(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    hs = np.array(list(COOLING_PROFILES.values()))
    X = np.column_stack([rng.uniform(*FIN_COUNT_BOUNDS, n), rng.uniform(*FIN_HEIGHT_BOUNDS_M, n),
                         rng.uniform(*FIN_THICKNESS_BOUNDS_M, n), rng.uniform(15.0, 150.0, n),
                         rng.choice(hs, n) * rng.uniform(0.85, 1.15, n)])
    y = np.array([junction_temperature_c(FinArrayDesign(*r[:3]), r[3], r[4]) for r in X])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)[:, None]


def train_surrogate(n_samples: int = 6000, epochs: int = 400, seed: int = 0):
    """MLP (inputs: fin count, height m, thickness m, power W, h W/m2K -> base temperature C) trained full-batch with
    Adam on the closed form; the last 10 % of the samples are held out. Returns ``(surrogate, val_rel_error)`` where
    the error is the mean of |pred - true| / true over the held-out samples (temperatures in C)."""
    from pinneapple_design.design_optimizer.surrogate import PhysicsSurrogate

    torch.manual_seed(seed)
    X, y = _samples(n_samples, seed)
    n_val = max(1, n_samples // 10)
    core = PhysicsSurrogate.build_mlp(in_dim=5, out_dim=1, hidden=48, layers=3)
    model = torch.nn.Sequential(_Scale(_SCALES, True), core.model, _Scale(_T_SCALE, False))
    sur = PhysicsSurrogate(model, core.cfg)
    sur.train_on_dataset({"theta": X[:-n_val], "u": y[:-n_val]}, epochs=epochs, lr=3e-3, verbose=False)
    with torch.no_grad():
        pred = sur.model(X[-n_val:])
    return sur, float((torch.abs(pred - y[-n_val:]) / y[-n_val:]).mean())


def time_surrogate(surrogate, d: FinArrayDesign, power_w: float, h_conv: float, reps: int = 200) -> dict[str, float]:
    """Mean wall time of one forward pass (``reps`` calls after a warm-up call) and the prediction."""
    x = torch.tensor([[d.fin_count, d.fin_height_m, d.fin_thickness_m, power_w, h_conv]], dtype=torch.float32)
    model = surrogate.model
    with torch.no_grad():
        model(x)
        t0 = time.perf_counter()
        for _ in range(reps):
            pred = model(x)
        dt = time.perf_counter() - t0
    return {"predicted_c": float(pred.item()), "time_s": dt / reps}


def reference_solve(d: FinArrayDesign, power_w: float, h_conv: float, refine: int = 1) -> dict[str, Any]:
    """Finite-volume unit-cell solve (cell size = thinnest feature / (3 refine)), timed end to end."""
    from pinneapple_simulation.numerical_solvers.fin_cell_fv import solve_fin_cell

    pitch = d.fin_spacing_m() + d.fin_thickness_m
    t0 = time.perf_counter()
    r = solve_fin_cell(pitch=pitch, t_fin=d.fin_thickness_m, height=d.fin_height_m, t_base=BASE_THICKNESS_M,
                       depth=BASE_D_M, k=K_ALUMINUM, h_conv=h_conv, power_per_fin=power_w / d.fin_count,
                       t_amb=T_AMBIENT_C,
                       cell=min(d.fin_thickness_m, pitch - d.fin_thickness_m, BASE_THICKNESS_M) / (3 * refine))
    return {"junction_c": r["junction_c"], "wall_time_s": time.perf_counter() - t0, "solve_time_s": r["solve_time_s"],
            "n_cells": r["n_cells"], "heat_balance_error": r["heat_balance_error"], "cell_size_m": r["cell_size_m"]}


def _rise_diff_pct(a_c: float, ref_c: float) -> float:
    return abs((a_c - T_AMBIENT_C) - (ref_c - T_AMBIENT_C)) / max(abs(ref_c - T_AMBIENT_C), 1e-6) * 100.0


def measure_design(surrogate, spec: dict[str, Any], timing_reps: int = 200, grid_check: bool = True) -> dict[str, Any]:
    """Closed form, timed surrogate and timed reference for one design; with ``grid_check`` the reference is also
    solved on two finer grids and the grid convergence index (``pinneapple_lab.uq.gci``) of the base temperature
    rise is reported."""
    d = _design(spec)
    power, h = spec["power_w"], COOLING_PROFILES[spec["cooling_profile"]]
    t0 = time.perf_counter()
    analytical = junction_temperature_c(d, power, h)
    analytical_s = time.perf_counter() - t0
    s = time_surrogate(surrogate, d, power, h, timing_reps)
    ref = reference_solve(d, power, h)
    out = {"name": spec["name"], "label": spec["label"], "short": spec["short"],
           "design": {"fin_count": d.fin_count, "fin_height_mm": d.fin_height_m * 1e3,
                      "fin_thickness_mm": d.fin_thickness_m * 1e3, "fin_spacing_mm": d.fin_spacing_m() * 1e3},
           "power_w": power, "cooling_profile": spec["cooling_profile"], "h_conv": h,
           "constraints_ok": all(v <= 1e-9 for v in constraint_violations(d).values()),
           "analytical_c": analytical, "analytical_time_s": analytical_s,
           "surrogate_c": s["predicted_c"], "surrogate_time_s": s["time_s"],
           "fea_c": ref["junction_c"], "fea_solve_time_s": ref["solve_time_s"], "fea_wall_time_s": ref["wall_time_s"],
           "fea_num_cells": ref["n_cells"], "fea_heat_balance_error": ref["heat_balance_error"],
           "surrogate_vs_fea_diff_pct": _rise_diff_pct(s["predicted_c"], ref["junction_c"]),
           "analytical_vs_fea_diff_pct": _rise_diff_pct(analytical, ref["junction_c"]),
           "surrogate_vs_fea_abs_error_c": abs(s["predicted_c"] - ref["junction_c"]),
           "speedup_factor": ref["wall_time_s"] / s["time_s"] if s["time_s"] > 0 else float("inf")}
    if grid_check:
        from pinneapple_lab.uq import gci

        fine = [reference_solve(d, power, h, refine=r) for r in (2, 4)]
        rises = [x["junction_c"] - T_AMBIENT_C for x in (ref, *fine)]
        out["fea_gci"] = gci(rises, [x["cell_size_m"] for x in (ref, *fine)])["gci"]
    return out


def summarize(designs: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [d["surrogate_vs_fea_diff_pct"] for d in designs]
    sp = [d["speedup_factor"] for d in designs]
    ref_t = sum(d["fea_wall_time_s"] for d in designs)
    sur_t = sum(d["surrogate_time_s"] for d in designs)
    ae = [d["surrogate_vs_fea_abs_error_c"] for d in designs]
    return {"n_designs": len(designs), "mean_speedup_factor": float(np.mean(sp)), "min_speedup_factor": min(sp),
            "max_speedup_factor": max(sp), "aggregate_speedup_factor": ref_t / sur_t if sur_t > 0 else float("inf"),
            "mean_agreement_diff_pct": float(np.mean(diffs)), "max_agreement_diff_pct": max(diffs),
            "agreement_threshold_pct": AGREEMENT_THRESHOLD_PCT,
            "all_within_threshold": max(diffs) <= AGREEMENT_THRESHOLD_PCT,
            "total_fea_wall_time_s": ref_t, "total_surrogate_time_s": sur_t, "mean_abs_error_c": float(np.mean(ae)),
            "max_abs_error_c": max(ae),
            "max_analytical_vs_fea_diff_pct": max(d["analytical_vs_fea_diff_pct"] for d in designs)}


def run_sweep(surrogate, progress_cb: Callable[[int, int], None] | None = None, timing_reps: int = 200,
              grid_check: bool = True) -> dict[str, Any]:
    """Measure every design of ``CURATED_DESIGNS``; ``progress_cb(i, n)`` after each."""
    out = []
    for i, spec in enumerate(CURATED_DESIGNS):
        out.append(measure_design(surrogate, spec, timing_reps, grid_check))
        if progress_cb is not None:
            progress_cb(i + 1, len(CURATED_DESIGNS))
    return {"designs": out, "summary": summarize(out)}


def thermal_field(name: str = "baseline") -> dict[str, Any]:
    """Parameters of the closed-form fin temperature profile theta(z) = theta_b cosh(m (Lc - z)) / cosh(m Lc) for a
    viewer that colours the solid per vertex, plus the base geometry."""
    spec = design_by_name(name)
    d = _design(spec)
    p = _profile(d, spec["power_w"], COOLING_PROFILES[spec["cooling_profile"]])
    return {"t_ambient_c": T_AMBIENT_C, "theta_fin_base_c": p["theta_fin_base_c"], "m": p["m"], "lc_m": p["lc_m"],
            "base_thickness_m": BASE_THICKNESS_M, "base_temperature_c": junction_temperature_c(
                d, spec["power_w"], COOLING_PROFILES[spec["cooling_profile"]])}
