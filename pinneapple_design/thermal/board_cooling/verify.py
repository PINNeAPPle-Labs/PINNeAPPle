"""Independent verification of the chosen design and its trust report.

The design the loop confirmed is solved again with both models on finer grids (``FIDELITY["verify"]``: board cells
x2 in each direction, heat-sink base grid x2), so the decision does not rest on the grid the surrogate was trained
on. The trust report (:class:`pinneapple_analysis.trust.TrustReport`) collects:

* limits: every junction at or below its limit with pessimistic cooling (critical), fan air power within budget
  (critical), heat sink inside the envelope (critical);
* physics: board energy balance, coupling fixed point converged, the heat-sink engine's own checks (energy balance,
  spreading by two methods, grid convergence, correlation range), grid convergence of the coupled solve
  (train vs verify grid);
* the surrogate: its prediction of the design within its +-2 std band of the fine solve plus the grid difference,
  hold-out error and coverage measured on unseen designs, applicability (inside the design space, no farther from
  a snapshot than the held-out designs whose error was measured).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pinneapple_analysis.trust import Check, TrustReport

from .mission import CoolingDesign, CoolingMission
from .simulate import response, simulate

SUP, CON, NOT = "supports", "contradicts", "not_run"


def _c(name, ok, *, critical=False, value=None, limit=None, source="", reason="") -> Check:
    st = NOT if ok is None else (SUP if ok else CON)
    return Check(name, st, critical, 1.0, None if value is None else float(value),
                 None if limit is None else float(limit), source, reason)


def verify(mission: CoolingMission, design: CoolingDesign, *, coarse: dict[str, Any] | None = None,
           prediction: dict[str, np.ndarray] | None = None, holdout: dict[str, Any] | None = None,
           train_X: np.ndarray | None = None, test_X: np.ndarray | None = None, fidelity: str | dict = "verify"
           ) -> dict[str, Any]:
    """Fine solve of ``design`` and the trust report. ``coarse``: its solve on the training grid; ``prediction``:
    the surrogate's ``{"tj", "tj_std", "map"}`` for it; ``holdout``: :func:`.learn.holdout` output;
    ``train_X`` / ``test_X``: the designs the surrogate was fitted on and the held-out ones its error was measured on
    (applicability: the design must be no farther from its nearest snapshot than the farthest held-out design);
    ``fidelity``: grids of the verification solve."""
    fine = simulate(mission, design, fidelity=fidelity)
    names = [c.name for c in mission.components]
    lim = mission.limits()
    n = len(names)
    y = response(mission, fine)
    checks: list[Check] = []
    for nm in names:
        t = fine["tj_pessimistic_c"][nm]
        checks.append(_c(f"limit_tj_{nm}", t <= lim[nm], critical=True, value=t, limit=lim[nm],
                         source="fine coupled solve, pessimistic cooling",
                         reason=f"{nm}: {t:.2f} °C vs {lim[nm]:.0f} °C"))
    checks.append(_c("fan_power_budget", fine["fan_power_w"] <= mission.fan_power_budget_w, critical=True,
                     value=fine["fan_power_w"], limit=mission.fan_power_budget_w, source="closed form (dp x flow)"))
    env_ok = (design.base_width_mm <= mission.max_width_mm + 1e-9 and design.base_depth_mm <= mission.max_depth_mm
              + 1e-9 and design.height_mm <= mission.max_height_mm + 1e-9)
    checks.append(_c("envelope", env_ok, critical=True, value=design.height_mm, limit=mission.max_height_mm,
                     source="geometry", reason=f"{design.base_width_mm:.1f} x {design.base_depth_mm:.1f} x "
                     f"{design.height_mm:.1f} mm in {mission.max_width_mm:g} x {mission.max_depth_mm:g} x "
                     f"{mission.max_height_mm:g} mm"))
    checks.append(_c("energy_balance_board", fine["energy_balance_board"] < 1e-6, value=fine["energy_balance_board"],
                     limit=1e-6, source="pcb.solve: heat out / heat in - 1"))
    checks.append(_c("coupling_converged", fine["coupling_converged"], source="fixed point board <-> heat sink",
                     reason=f"{fine['coupling_iterations']} iterations"))
    for ch in fine["sink"]["checks"]:
        ok = None if ch["status"] == "n/a" else ch["status"] in ("pass", "warn")
        checks.append(_c(f"heat_sink_{ch['name']}", ok, value=ch["value"] if isinstance(ch["value"], (int, float))
                         else None, source="heatsink engine", reason=ch["detail"]))
    grid = None
    if coarse is not None:
        grid = float(np.abs(response(mission, coarse) - y).max())
        checks.append(_c("grid_convergence_tj", grid <= 2.0, value=grid, limit=2.0,
                         source="train grid vs verify grid (max |dTj| over parts, nominal and pessimistic)"))
    if prediction is not None:
        err = np.abs(prediction["tj"] - y)
        band = 2 * prediction["tj_std"] + (grid or 0.0)
        checks.append(_c("surrogate_within_band", bool(np.all(err <= band + 0.05)), value=float(err.max()),
                         limit=float(band[np.argmax(err)] + 0.05),
                         source="surrogate vs fine solve, 2 std + grid difference"))
        if "map" in prediction:
            em = float(np.abs(prediction["map"] - fine["map_c"]).max())
            checks.append(_c("surrogate_map_max_error", em <= 2.0 + (grid or 0.0), value=em,
                             limit=2.0 + (grid or 0.0), source="board-top map, surrogate vs fine solve [K]"))
    if holdout is not None:
        s = holdout["surrogate"]
        checks.append(_c("holdout_tj_max_error", s["tj_max_abs_k"] <= 3.0, value=s["tj_max_abs_k"], limit=3.0,
                         source=f"{holdout['n_test']} unseen designs [K]"))
        checks.append(_c("holdout_beats_nearest", s["tj_mae_k"] < holdout["nearest_snapshot"]["tj_mae_k"],
                         value=s["tj_mae_k"], limit=holdout["nearest_snapshot"]["tj_mae_k"],
                         source="mean |error| vs nearest-snapshot baseline [K]"))
        checks.append(_c("holdout_coverage_2std", holdout["coverage_2std"] >= 0.9, value=holdout["coverage_2std"],
                         limit=0.9, source="fraction of unseen errors inside +-2 std"))
    sp = mission.space
    x = design.vector()
    checks.append(_c("inside_design_space", sp.contains(x), source="box the snapshots were drawn from"))
    if train_X is not None and test_X is not None:
        span = np.maximum(sp.hi - sp.lo, 1e-12)
        nn = lambda Q: np.sqrt((((Q[:, None, :] - train_X[None]) / span) ** 2).sum(-1)).min(1)  # noqa: E731
        d, d_test = float(nn(x[None])[0]), nn(np.atleast_2d(test_X))
        checks.append(_c("nearest_snapshot_distance", d <= float(d_test.max()), value=d, limit=float(d_test.max()),
                         source="scaled distance to the nearest training design vs the held-out designs"))
    rep = TrustReport(checks, model=f"cooling design for {mission.hot}")
    return {"fine": fine, "checks": checks, "report": rep, "decision": rep.decisao,
            "grid_max_dtj_k": grid, "tj_fine": dict(zip(names, y[:n].tolist(), strict=True)),
            "tj_fine_pessimistic": dict(zip(names, y[n:].tolist(), strict=True))}
