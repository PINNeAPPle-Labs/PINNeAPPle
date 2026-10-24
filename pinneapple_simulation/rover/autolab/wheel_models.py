"""Wheel-soil laws at the operating point a rover needs, and the three ways to get them.

The planner needs, per soil and grade, the slip a wheel must run at to climb (and whether any slip below the
limit suffices), its sinkage, the motor torque and the drive energy per metre. :func:`grade_table` computes these
from the Wong-Reece model (quasi-static rover: ``n`` equally loaded wheels, load ``m g cos(a) / n``, drawbar pull
``m g sin(a) / n``; no load transfer, no cross-slope, no acceleration).

Three approximations of the same wheel-soil law are compared by :func:`compare_contact_models`:

* ``analytic`` -- :class:`~pinneapple_simulation.terramechanics.WongReeceModel` with the soil the rover believes in;
* ``surrogate`` -- any ``predict(slip, sinkage) -> (Fx, Fz, My)`` model, e.g. the model-zoo network
  ``rover_contact_grc1_viper`` (trained on GRC-1 only; its validity box is checked per point);
* ``measured`` -- operating points recorded by the simulated rover (the multibody simulation on the true soil, with
  sensor noise), the reference both are scored against.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...terramechanics.soils import Soil
from ...terramechanics.wheel import Wheel
from ...terramechanics.wong_reece import WongReeceModel

SLIPS = np.r_[np.linspace(-0.3, -0.02, 8), np.linspace(0.0, 0.9, 46)]
GRADES = np.arange(-24.0, 24.01, 2.0)


@dataclass
class GradeTable:
    """Operating points of a quasi-static rover on one soil, per grade [deg] (arrays over ``grades``)."""
    grades: np.ndarray
    slip: np.ndarray            # slip needed to climb; nan where no slip in the table suffices
    sinkage: np.ndarray         # [m]
    torque: np.ndarray          # motor torque per wheel [N m]
    energy_per_m: np.ndarray    # drive energy per metre of travel, all wheels [J/m] (no regeneration)
    max_dp_over_w: float        # largest drawbar pull / load on level ground over the slip range

    def at(self, grade_deg, name: str = "slip"):
        """Interpolate a quantity at grades (nan where the rover cannot climb)."""
        return np.interp(np.asarray(grade_deg, float), self.grades, getattr(self, name), left=np.nan, right=np.nan)


def grade_table(soil: Soil, wheel: Wheel, mass: float, g: float = 1.62, n_wheels: int = 4, grades=GRADES,
                slips=SLIPS, n_gl: int = 12) -> GradeTable:
    """Slip, sinkage, torque and energy per metre a ``mass`` kg rover on ``n_wheels`` needs on each grade."""
    model = WongReeceModel(soil, wheel, g=g, n_gl=n_gl)
    G, S = np.meshgrid(np.asarray(grades, float), np.asarray(slips, float), indexing="ij")
    a = np.radians(G)
    W = mass * g * np.cos(a) / n_wheels
    need = mass * g * np.sin(a) / n_wheels
    ss = model.steady_state(W, S)
    dp, z, My = ss["drawbar_pull"], ss["sinkage"], ss["torque"]
    ok = dp >= need
    need = need[:, 0]
    out = {k: np.full(len(grades), np.nan) for k in ("slip", "sinkage", "torque", "energy")}
    for i in range(len(grades)):
        j = np.flatnonzero(ok[i])
        if not len(j):
            continue
        j = int(j[0])
        j0, f = j, 1.0
        if j > 0:                                   # linear interpolation between the bracketing slips
            d0, d1 = dp[i, j - 1] - need[i], dp[i, j] - need[i]
            j0, f = j - 1, float(np.clip(-d0 / max(d1 - d0, 1e-12), 0.0, 1.0))
        s, tq, zz = ((1 - f) * A[i, j0] + f * A[i, j] for A in (S, My, z))
        out["slip"][i], out["sinkage"][i], out["torque"][i] = s, zz, tq
        # omega = v / (R (1 - s)); energy per metre = n My omega / v, no regeneration on negative torque
        out["energy"][i] = n_wheels * max(tq, 0.0) / (wheel.radius * (1.0 - s))
    lvl = int(np.argmin(np.abs(np.asarray(grades, float))))
    return GradeTable(np.asarray(grades, float), out["slip"], out["sinkage"], out["torque"], out["energy"],
                      float(np.nanmax(dp[lvl] / W[lvl, 0])))


def surrogate_steady_state(predict, load, slip, z_max: float = 0.2, iters: int = 50):
    """Sinkage with ``Fz(slip, z) = load`` for a ``predict(slip, z) -> (N, 3)`` contact model (bisection), and the
    forces there. Returns dict like :meth:`WongReeceModel.steady_state`."""
    W, s = (np.asarray(a, float).ravel() for a in np.broadcast_arrays(load, slip))
    lo, hi = np.full_like(W, 1e-4), np.full_like(W, z_max)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = predict(s, mid)[:, 1] < W
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    z = 0.5 * (lo + hi)
    F = predict(s, z)
    return {"slip": s, "sinkage": z, "drawbar_pull": F[:, 0], "load": F[:, 1], "torque": F[:, 2]}


def compare_contact_models(measured: dict, soil_believed: Soil, wheel: Wheel, surrogate=None, g: float = 1.62,
                           in_range=None) -> dict:
    """Score the analytic law (believed soil) and a surrogate against measured operating points.

    ``measured``: arrays ``load`` [N], ``slip`` [-], ``sinkage`` [m], ``torque`` [N m] (one entry per sample).
    Each model predicts sinkage and torque from the measured load and slip. Returns RMSE and bias per model and,
    for the surrogate, the share of the measured operating points (measured slip and sinkage) outside its validity
    box (``in_range(slip, z)``); a surrogate trained on another soil is outside its validity whatever the box says."""
    W, s = np.asarray(measured["load"], float), np.asarray(measured["slip"], float)
    zm, tm = np.asarray(measured["sinkage"], float), np.asarray(measured["torque"], float)
    out = {"n": int(len(W))}
    ana = WongReeceModel(soil_believed, wheel, g=g).steady_state(W, s)
    preds = {"analytic": ana}
    if surrogate is not None:
        preds["surrogate"] = surrogate_steady_state(surrogate, W, s)
    for name, p in preds.items():
        ez, et = p["sinkage"] - zm, p["torque"] - tm
        out[name] = {"sinkage_rmse_mm": float(1e3 * np.sqrt(np.mean(ez ** 2))),
                     "sinkage_bias_mm": float(1e3 * np.mean(ez)),
                     "torque_rmse_Nm": float(np.sqrt(np.mean(et ** 2))), "torque_bias_Nm": float(np.mean(et)),
                     "sinkage_rel_err": float(np.sqrt(np.mean(ez ** 2)) / max(np.sqrt(np.mean(zm ** 2)), 1e-12)),
                     "torque_rel_err": float(np.sqrt(np.mean(et ** 2)) / max(np.sqrt(np.mean(tm ** 2)), 1e-12))}
        if name == "surrogate" and in_range is not None:     # the operating points the rover actually met
            inside = np.asarray(in_range(s, zm), bool)
            out[name]["fraction_outside_validity"] = float(1.0 - inside.mean())
        out[name]["pred"] = {"sinkage": p["sinkage"], "torque": p["torque"]}
    return out


def max_climbable_grade(table: GradeTable, slip_limit: float) -> float:
    """Largest grade of the table the rover climbs at a slip at or below ``slip_limit`` [deg] (nan if none)."""
    ok = np.isfinite(table.slip) & (table.slip <= slip_limit) & (table.grades >= 0)
    return float(table.grades[ok].max()) if ok.any() else math.nan


__all__ = ["GradeTable", "grade_table", "surrogate_steady_state", "compare_contact_models", "max_climbable_grade",
           "SLIPS", "GRADES"]
